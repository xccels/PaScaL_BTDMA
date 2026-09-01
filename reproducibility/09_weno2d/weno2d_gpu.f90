!==============================================================================
! 2D Zalesak Slotted Disk / Solid-Body Rotation -- PaScaL_BTDMA
!   SDIRK4 (5-stage, L-stable, gamma = 1/4) + 5th-order WENO (frozen weights,
!   signed upwind) + Beam-Warming ADI + m = 3 block tridiagonal via PaScaL_BTDMA.
!
! Public reproduction driver derived from the development source documented in
! SOURCE_PROVENANCE.md.  It accepts exactly one of the three manuscript cases:
!   cfl1p0 -> CFL 1.0, 1621 steps
!   cfl1p5 -> CFL 1.5, 1080 steps
!   cfl2p0 -> CFL 2.0,  810 steps
!
! Why m = 3 (not 2)
! -----------------
!   WENO5 semi-discrete stencil is 6-wide and asymmetric (upwind):
!     c >= 0 : offsets [-3, -2, -1, 0, 1, 2]   (left-biased)
!     c <  0 : offsets [-2, -1, 0, 1, 2, 3]    (right-biased)
!   With m = 2, row 2I-1 (for c>=0) uses index 2I-4, which sits in block I-2,
!   violating the block tridiagonal (I-1, I, I+1) connectivity.
!   With m = 3 the 6-wide stencil fits exactly into three super-blocks.
!
! Grid size constraint
! --------------------
!   n1sub, n2sub must both be multiples of m = 3.  Since n_sub = n / np, the
!   global n1, n2 must be multiples of 3 * np.  The manuscript case is fixed to
!     N1 = N2 = 516, np = 2 * 2  ->  n_sub = 258, nblk = 86.
!
! SDIRK4 (Hairer-Wanner, gamma=1/4, stiffly accurate, L-stable)
!   All stages share M = I - gamma*dt*L (frozen spatial operator).
!   For ADI: M ~= (I - gamma*dt*L_x)(I - gamma*dt*L_y).
!   Stage i:  rhs_i = U^n + dt * sum_{j<i} A_ij K_j
!             solve M U_i = rhs_i  (via x then y BTDMA sweeps)
!             K_i = L U_i  (computed via (U_i - rhs_i)/(gamma*dt), ADI-consistent)
!   U^{n+1} = U_s  (stiffly accurate)
!
! Build-time constants (set via -D):
!   N1, N2     grid size, multiples of 3 * np_i
!   NP1, NP2   MPI decomposition
!   M          fixed = 3
!==============================================================================

module params
    use mpi
    use cudafor
    implicit none

    integer, parameter :: n1 = N1
    integer, parameter :: n2 = N2
    integer, parameter :: np1 = NP1
    integer, parameter :: np2 = NP2
    integer, parameter :: m = M                ! = 3 (WENO5 block grouping)
    integer, parameter :: nhalo = 3            ! WENO5 stencil half-width

    real(8), parameter :: pi = 3.141592653589793d0

    real(8), parameter :: Lx = 1.0d0
    real(8), parameter :: Ly = 1.0d0

    ! ---- Solid-body rotation ----
    real(8), parameter :: xc_rot = 0.5d0
    real(8), parameter :: yc_rot = 0.5d0
    real(8), parameter :: T_period = 1.0d0
    real(8), parameter :: omega = 2.0d0 * pi / T_period

    ! ---- Zalesak slotted disk ----
    real(8), parameter :: zal_x0 = 0.5d0
    real(8), parameter :: zal_y0 = 0.75d0
    real(8), parameter :: zal_R  = 0.15d0
    real(8), parameter :: zal_slot_hw  = 0.025d0
    real(8), parameter :: zal_slot_top = 0.85d0

    ! ---- SDIRK4 ----
    integer, parameter :: s_stages = 5
    real(8), parameter :: gamma_rk = 0.25d0
    real(8), device :: Ac_d(s_stages, s_stages)

    ! ---- WENO5 ideal linear weights (left-biased) ----
    real(8), parameter :: d_w0 = 1.0d0/10.0d0
    real(8), parameter :: d_w1 = 6.0d0/10.0d0
    real(8), parameter :: d_w2 = 3.0d0/10.0d0
    real(8), parameter :: weno_eps = 1.0d-6

    integer, public :: mpi_world_cart
    integer, public :: np_dim(0:1)
    logical, public :: period(0:1)

    type, public :: cart_comm_1d
        integer :: myrank
        integer :: nprocs
        integer :: west_rank
        integer :: east_rank
        integer :: mpi_comm
    end type cart_comm_1d

    type(cart_comm_1d), public :: comm_1d_x
    type(cart_comm_1d), public :: comm_1d_y

contains

    subroutine mpi_topology_clean()
        implicit none
        integer :: ierr
        call MPI_Comm_free(mpi_world_cart, ierr)
    end subroutine mpi_topology_clean

    subroutine mpi_topology_make()
        implicit none
        logical :: remain(0:1)
        integer :: ierr
        call MPI_Cart_create(MPI_COMM_WORLD, 2, np_dim, period, .false., mpi_world_cart, ierr)
        remain(0)=.true.;  remain(1)=.false.
        call MPI_Cart_sub(mpi_world_cart, remain, comm_1d_x%mpi_comm, ierr)
        call MPI_Comm_rank(comm_1d_x%mpi_comm, comm_1d_x%myrank, ierr)
        call MPI_Comm_size(comm_1d_x%mpi_comm, comm_1d_x%nprocs, ierr)
        call MPI_Cart_shift(comm_1d_x%mpi_comm, 0, 1, comm_1d_x%west_rank, comm_1d_x%east_rank, ierr)
        remain(0)=.false.; remain(1)=.true.
        call MPI_Cart_sub(mpi_world_cart, remain, comm_1d_y%mpi_comm, ierr)
        call MPI_Comm_rank(comm_1d_y%mpi_comm, comm_1d_y%myrank, ierr)
        call MPI_Comm_size(comm_1d_y%mpi_comm, comm_1d_y%nprocs, ierr)
        call MPI_Cart_shift(comm_1d_y%mpi_comm, 0, 1, comm_1d_y%west_rank, comm_1d_y%east_rank, ierr)
    end subroutine mpi_topology_make

    !--------------------------------------------------------------------------
    ! 2D halo exchange with nhalo layers (for WENO5).
    ! Uses contiguous device buffers + CUDA-aware MPI.
    !--------------------------------------------------------------------------
    subroutine exchange_halo_2d_gpu(U_d, n1sub, n2sub)
        use mpi; use cudafor
        implicit none
        integer, intent(in) :: n1sub, n2sub
        real(8), device, intent(inout) :: U_d(1-nhalo:n1sub+nhalo, 1-nhalo:n2sub+nhalo)
        integer :: ierr
        integer :: reqs(4), stats(MPI_STATUS_SIZE, 4)
        real(8), device, allocatable :: sb_w(:,:), sb_e(:,:), rb_w(:,:), rb_e(:,:)
        real(8), device, allocatable :: sb_s(:,:), sb_n(:,:), rb_s(:,:), rb_n(:,:)
        integer :: ix_nh
        integer :: i, j

        ix_nh = n2sub + 2*nhalo

        ! x direction
        allocate(sb_w(nhalo, ix_nh), sb_e(nhalo, ix_nh))
        allocate(rb_w(nhalo, ix_nh), rb_e(nhalo, ix_nh))
        !$cuf kernel do(2) <<<*,*>>>
        do j = 1-nhalo, n2sub+nhalo
          do i = 1, nhalo
            sb_w(i, j+nhalo) = U_d(i,            j)
            sb_e(i, j+nhalo) = U_d(n1sub-nhalo+i,j)
          end do
        end do
        ierr = cudaDeviceSynchronize()
        call MPI_Isend(sb_e(1,1), nhalo*ix_nh, MPI_DOUBLE_PRECISION, comm_1d_x%east_rank, 0, comm_1d_x%mpi_comm, reqs(1), ierr)
        call MPI_Irecv(rb_w(1,1), nhalo*ix_nh, MPI_DOUBLE_PRECISION, comm_1d_x%west_rank, 0, comm_1d_x%mpi_comm, reqs(2), ierr)
        call MPI_Isend(sb_w(1,1), nhalo*ix_nh, MPI_DOUBLE_PRECISION, comm_1d_x%west_rank, 1, comm_1d_x%mpi_comm, reqs(3), ierr)
        call MPI_Irecv(rb_e(1,1), nhalo*ix_nh, MPI_DOUBLE_PRECISION, comm_1d_x%east_rank, 1, comm_1d_x%mpi_comm, reqs(4), ierr)
        call MPI_Waitall(4, reqs, stats, ierr)
        !$cuf kernel do(2) <<<*,*>>>
        do j = 1-nhalo, n2sub+nhalo
          do i = 1, nhalo
            U_d(1-nhalo+i-1,    j) = rb_w(i, j+nhalo)
            U_d(n1sub+i,        j) = rb_e(i, j+nhalo)
          end do
        end do
        ierr = cudaDeviceSynchronize()
        deallocate(sb_w, sb_e, rb_w, rb_e)

        ! y direction
        allocate(sb_s(n1sub+2*nhalo, nhalo), sb_n(n1sub+2*nhalo, nhalo))
        allocate(rb_s(n1sub+2*nhalo, nhalo), rb_n(n1sub+2*nhalo, nhalo))
        !$cuf kernel do(2) <<<*,*>>>
        do j = 1, nhalo
          do i = 1-nhalo, n1sub+nhalo
            sb_s(i+nhalo, j) = U_d(i, j)
            sb_n(i+nhalo, j) = U_d(i, n2sub-nhalo+j)
          end do
        end do
        ierr = cudaDeviceSynchronize()
        call MPI_Isend(sb_n(1,1), (n1sub+2*nhalo)*nhalo, MPI_DOUBLE_PRECISION, &
                       comm_1d_y%east_rank, 2, comm_1d_y%mpi_comm, reqs(1), ierr)
        call MPI_Irecv(rb_s(1,1), (n1sub+2*nhalo)*nhalo, MPI_DOUBLE_PRECISION, &
                       comm_1d_y%west_rank, 2, comm_1d_y%mpi_comm, reqs(2), ierr)
        call MPI_Isend(sb_s(1,1), (n1sub+2*nhalo)*nhalo, MPI_DOUBLE_PRECISION, &
                       comm_1d_y%west_rank, 3, comm_1d_y%mpi_comm, reqs(3), ierr)
        call MPI_Irecv(rb_n(1,1), (n1sub+2*nhalo)*nhalo, MPI_DOUBLE_PRECISION, &
                       comm_1d_y%east_rank, 3, comm_1d_y%mpi_comm, reqs(4), ierr)
        call MPI_Waitall(4, reqs, stats, ierr)
        !$cuf kernel do(2) <<<*,*>>>
        do j = 1, nhalo
          do i = 1-nhalo, n1sub+nhalo
            U_d(i, 1-nhalo+j-1) = rb_s(i+nhalo, j)
            U_d(i, n2sub+j)     = rb_n(i+nhalo, j)
          end do
        end do
        ierr = cudaDeviceSynchronize()
        deallocate(sb_s, sb_n, rb_s, rb_n)
    end subroutine exchange_halo_2d_gpu

    !--------------------------------------------------------------------------
    ! Exchange one x-block (m doubles per y-row) of an x-block-layout array with
    ! x-direction neighbours, populating ghost slots iblk=0 and iblk=nblk_x+1
    ! (cyclic).  Used by apply_Lx matvec which reads U_xb at iblk-1, iblk+1.
    !--------------------------------------------------------------------------
    subroutine exchange_xblock_halo(U_xb_full, n1sub, n2sub)
        use mpi; use cudafor
        implicit none
        integer, intent(in) :: n1sub, n2sub
        real(8), device, intent(inout) :: U_xb_full(n2sub, 0:n1sub/m+1, m)
        integer :: ierr, reqs(4), stats(MPI_STATUS_SIZE, 4)
        real(8), device, allocatable :: sb_w(:,:), sb_e(:,:), rb_w(:,:), rb_e(:,:)
        integer :: nblk_x, msz, j, mm

        nblk_x = n1sub / m
        msz = n2sub * m
        allocate(sb_w(n2sub, m), sb_e(n2sub, m))
        allocate(rb_w(n2sub, m), rb_e(n2sub, m))

        !$cuf kernel do(2) <<<*,*>>>
        do mm = 1, m
            do j = 1, n2sub
                sb_w(j, mm) = U_xb_full(j, 1,      mm)
                sb_e(j, mm) = U_xb_full(j, nblk_x, mm)
            end do
        end do
        ierr = cudaDeviceSynchronize()

        ! send last block east -> west neighbour's last (= our ghost iblk=0)
        ! send first block west -> east neighbour's first (= our ghost iblk=nblk_x+1)
        call MPI_Isend(sb_e(1,1), msz, MPI_DOUBLE_PRECISION, comm_1d_x%east_rank, 100, comm_1d_x%mpi_comm, reqs(1), ierr)
        call MPI_Irecv(rb_w(1,1), msz, MPI_DOUBLE_PRECISION, comm_1d_x%west_rank, 100, comm_1d_x%mpi_comm, reqs(2), ierr)
        call MPI_Isend(sb_w(1,1), msz, MPI_DOUBLE_PRECISION, comm_1d_x%west_rank, 101, comm_1d_x%mpi_comm, reqs(3), ierr)
        call MPI_Irecv(rb_e(1,1), msz, MPI_DOUBLE_PRECISION, comm_1d_x%east_rank, 101, comm_1d_x%mpi_comm, reqs(4), ierr)
        call MPI_Waitall(4, reqs, stats, ierr)

        !$cuf kernel do(2) <<<*,*>>>
        do mm = 1, m
            do j = 1, n2sub
                U_xb_full(j, 0,        mm) = rb_w(j, mm)
                U_xb_full(j, nblk_x+1, mm) = rb_e(j, mm)
            end do
        end do
        ierr = cudaDeviceSynchronize()

        deallocate(sb_w, sb_e, rb_w, rb_e)
    end subroutine exchange_xblock_halo

end module params


module kernels
    use cudafor
    use params
    implicit none

contains

    !--------------------------------------------------------------------------
    ! IC: Zalesak slotted disk
    !--------------------------------------------------------------------------
    attributes(global) subroutine init_zalesak_kernel(U, x, y, n1sub, n2sub)
        integer, value, intent(in) :: n1sub, n2sub
        real(8), device, intent(out) :: U(1-nhalo:n1sub+nhalo, 1-nhalo:n2sub+nhalo)
        real(8), device, intent(in)  :: x(1-nhalo:n1sub+nhalo), y(1-nhalo:n2sub+nhalo)
        integer :: i, j
        real(8) :: rr
        logical :: in_disk, in_slot

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        if (i > n1sub .or. j > n2sub) return

        rr = sqrt((x(i) - zal_x0)**2 + (y(j) - zal_y0)**2)
        in_disk = (rr < zal_R)
        in_slot = (abs(x(i) - zal_x0) <= zal_slot_hw) .and. (y(j) <= zal_slot_top)
        if (in_disk .and. (.not. in_slot)) then
            U(i, j) = 1.0d0
        else
            U(i, j) = 0.0d0
        end if
    end subroutine init_zalesak_kernel

    !--------------------------------------------------------------------------
    ! Pack scalar U -> x-block layout (n2sub, n1sub/m, m).
    !--------------------------------------------------------------------------
    attributes(global) subroutine pack_scalar_to_xblock_kernel(U_sc, U_xb, n1sub, n2sub)
        integer, value, intent(in) :: n1sub, n2sub
        real(8), device, intent(in)  :: U_sc(1-nhalo:n1sub+nhalo, 1-nhalo:n2sub+nhalo)
        real(8), device, intent(out) :: U_xb(n2sub, n1sub/m, m)
        integer :: i, j, iblk, irow
        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        if (i > n1sub .or. j > n2sub) return
        iblk = (i + m - 1) / m;   irow = i - (iblk - 1) * m
        U_xb(j, iblk, irow) = U_sc(i, j)
    end subroutine pack_scalar_to_xblock_kernel

    attributes(global) subroutine unpack_xblock_to_scalar_kernel(U_xb, U_sc, n1sub, n2sub)
        integer, value, intent(in) :: n1sub, n2sub
        real(8), device, intent(in)    :: U_xb(n2sub, n1sub/m, m)
        real(8), device, intent(inout) :: U_sc(1-nhalo:n1sub+nhalo, 1-nhalo:n2sub+nhalo)
        integer :: i, j, iblk, irow
        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        if (i > n1sub .or. j > n2sub) return
        iblk = (i + m - 1) / m;   irow = i - (iblk - 1) * m
        U_sc(i, j) = U_xb(j, iblk, irow)
    end subroutine unpack_xblock_to_scalar_kernel

    attributes(global) subroutine stage_rhs_xblock_kernel(U_n_xb, K_xb, Ac_d, i_stage, dt, rhs_xb, n1sub, n2sub)
        integer, value, intent(in) :: n1sub, n2sub, i_stage
        real(8), value, intent(in) :: dt
        real(8), device, intent(in)  :: U_n_xb(n2sub, n1sub/m, m)
        real(8), device, intent(in)  :: K_xb(n2sub, n1sub/m, m, s_stages)
        real(8), device, intent(in)  :: Ac_d(s_stages, s_stages)
        real(8), device, intent(out) :: rhs_xb(n2sub, n1sub/m, m)
        integer :: j, iblk, irow, jj
        real(8) :: acc

        iblk = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j    = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        if (iblk > n1sub/m .or. j > n2sub) return
        do irow = 1, m
            acc = U_n_xb(j, iblk, irow)
            do jj = 1, i_stage - 1
                acc = acc + dt * Ac_d(i_stage, jj) * K_xb(j, iblk, irow, jj)
            end do
            rhs_xb(j, iblk, irow) = acc
        end do
    end subroutine stage_rhs_xblock_kernel

    ! NOTE: the old "free K" kernel  K_i = (U_i - rhs_i)/(gamma*dt)  has been
    ! removed.  It is INVALID under ADI splitting (introduces an O(gamma*dt*L_x*L_y)
    ! error term) and was the cause of garbage solutions at moderate CFL.
    ! K_i is now computed via the explicit-matvec sequence
    !   apply_Lx_xblock_kernel + compute_K_correct_kernel
    ! which faithfully reproduces Python's  K[i] = A_mat @ U_i.

    attributes(global) subroutine transpose_xy_d_kernel(Dx, Dy, n1sub, n2sub)
        integer, value, intent(in) :: n1sub, n2sub
        real(8), device, intent(in)  :: Dx(n2sub, n1sub/m, m)
        real(8), device, intent(out) :: Dy(n1sub, n2sub/m, m)
        integer :: i, j, iblk, irow, jblk, jrow
        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        if (i > n1sub .or. j > n2sub) return
        iblk = (i + m - 1)/m;   irow = i - (iblk-1)*m
        jblk = (j + m - 1)/m;   jrow = j - (jblk-1)*m
        Dy(i, jblk, jrow) = Dx(j, iblk, irow)
    end subroutine transpose_xy_d_kernel

    attributes(global) subroutine transpose_yx_d_kernel(Dy, Dx, n1sub, n2sub)
        integer, value, intent(in) :: n1sub, n2sub
        real(8), device, intent(in)  :: Dy(n1sub, n2sub/m, m)
        real(8), device, intent(out) :: Dx(n2sub, n1sub/m, m)
        integer :: i, j, iblk, irow, jblk, jrow
        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        if (i > n1sub .or. j > n2sub) return
        iblk = (i + m - 1)/m;   irow = i - (iblk-1)*m
        jblk = (j + m - 1)/m;   jrow = j - (jblk-1)*m
        Dx(j, iblk, irow) = Dy(i, jblk, jrow)
    end subroutine transpose_yx_d_kernel

    attributes(global) subroutine copy_xblock_kernel(src, dst, n1sub, n2sub)
        integer, value, intent(in) :: n1sub, n2sub
        real(8), device, intent(in)  :: src(n2sub, n1sub/m, m)
        real(8), device, intent(out) :: dst(n2sub, n1sub/m, m)
        integer :: j, iblk, irow
        iblk = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j    = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        if (iblk > n1sub/m .or. j > n2sub) return
        do irow = 1, m
            dst(j, iblk, irow) = src(j, iblk, irow)
        end do
    end subroutine copy_xblock_kernel

    !==========================================================================
    ! WENO5 core helpers (device-callable).
    ! For interface k+1/2 at row i (positive c, left-biased):
    !   uses cells {i-2, i-1, i, i+1, i+2}
    ! For interface k+1/2 (negative c, right-biased):
    !   uses cells {i-1, i, i+1, i+2, i+3}  (mirrored)
    !
    ! Returns 5 stencil coefficients (C_-2..C_+2 for left-biased, or
    ! C_-1..C_+3 for right-biased) that reconstruct f_hat from cell values.
    !==========================================================================
    attributes(device) subroutine weno5_coeffs_left(fm2, fm1, f0, fp1, fp2, c0, c1, c2, c3, c4)
        real(8), intent(in)  :: fm2, fm1, f0, fp1, fp2
        real(8), intent(out) :: c0, c1, c2, c3, c4    ! coefficients of fm2, fm1, f0, fp1, fp2
        real(8) :: b0, b1, b2, a0, a1, a2, asum, w0, w1, w2
        b0 = (13.0d0/12.0d0)*(fm2 - 2.0d0*fm1 + f0)**2  + 0.25d0*(fm2 - 4.0d0*fm1 + 3.0d0*f0)**2
        b1 = (13.0d0/12.0d0)*(fm1 - 2.0d0*f0 + fp1)**2  + 0.25d0*(fm1 - fp1)**2
        b2 = (13.0d0/12.0d0)*(f0 - 2.0d0*fp1 + fp2)**2  + 0.25d0*(3.0d0*f0 - 4.0d0*fp1 + fp2)**2
        a0 = d_w0 / (weno_eps + b0)**2
        a1 = d_w1 / (weno_eps + b1)**2
        a2 = d_w2 / (weno_eps + b2)**2
        asum = a0 + a1 + a2
        w0 = a0/asum;  w1 = a1/asum;  w2 = a2/asum
        c0 =  w0/3.0d0
        c1 = -7.0d0*w0/6.0d0 - w1/6.0d0
        c2 = 11.0d0*w0/6.0d0 + 5.0d0*w1/6.0d0 + w2/3.0d0
        c3 = w1/3.0d0 + 5.0d0*w2/6.0d0
        c4 = -w2/6.0d0
    end subroutine weno5_coeffs_left

    attributes(device) subroutine weno5_coeffs_right(fm1, f0, fp1, fp2, fp3, c0, c1, c2, c3, c4)
        real(8), intent(in)  :: fm1, f0, fp1, fp2, fp3
        real(8), intent(out) :: c0, c1, c2, c3, c4    ! coefficients of fm1, f0, fp1, fp2, fp3
        real(8) :: b0, b1, b2, a0, a1, a2, asum, w0, w1, w2
        ! Mirrored WENO5: reuse left-biased with reversed stencil
        b0 = (13.0d0/12.0d0)*(fp3 - 2.0d0*fp2 + fp1)**2 + 0.25d0*(fp3 - 4.0d0*fp2 + 3.0d0*fp1)**2
        b1 = (13.0d0/12.0d0)*(fp2 - 2.0d0*fp1 + f0)**2  + 0.25d0*(fp2 - f0)**2
        b2 = (13.0d0/12.0d0)*(fp1 - 2.0d0*f0 + fm1)**2  + 0.25d0*(3.0d0*fp1 - 4.0d0*f0 + fm1)**2
        a0 = d_w0 / (weno_eps + b0)**2
        a1 = d_w1 / (weno_eps + b1)**2
        a2 = d_w2 / (weno_eps + b2)**2
        asum = a0 + a1 + a2
        w0 = a0/asum;  w1 = a1/asum;  w2 = a2/asum
        c0 = -w2/6.0d0
        c1 = w1/3.0d0 + 5.0d0*w2/6.0d0
        c2 = 11.0d0*w0/6.0d0 + 5.0d0*w1/6.0d0 + w2/3.0d0
        c3 = -7.0d0*w0/6.0d0 - w1/6.0d0
        c4 = w0/3.0d0
    end subroutine weno5_coeffs_right

    !--------------------------------------------------------------------------
    ! Pre-compute frozen WENO5 stencil coefficients at every interface for
    ! each y-row (x-sweep) / each x-col (y-sweep).
    !
    ! Storage: coef_x(n2sub, 0:n1sub, 5)
    !   coef_x(j, k, 1..5) = 5 stencil coefficients of f_hat at interface k+1/2
    !   in row j.  Ordering of the 5 coefficients depends on sign(c_x(j)):
    !     c >= 0 : coefficients multiply cells at offsets [-2,-1,0,+1,+2]
    !     c <  0 : coefficients multiply cells at offsets [-1, 0,+1,+2,+3]
    !--------------------------------------------------------------------------
    attributes(global) subroutine compute_weno_coeffs_x_kernel(U_d, y, coef_x, n1sub, n2sub)
        integer, value, intent(in) :: n1sub, n2sub
        real(8), device, intent(in)  :: U_d(1-nhalo:n1sub+nhalo, 1-nhalo:n2sub+nhalo)
        real(8), device, intent(in)  :: y(1-nhalo:n2sub+nhalo)
        real(8), device, intent(out) :: coef_x(n2sub, 0:n1sub, 5)
        integer :: j, k
        real(8) :: cx
        real(8) :: c0, c1, c2, c3, c4

        k = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 1  ! 0..n1sub
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        if (k > n1sub .or. j < 1 .or. j > n2sub) return

        cx = -omega * (y(j) - yc_rot)
        ! Interface k+1/2 sits to the right of cell k (so reference cell index = k)
        if (cx >= 0.0d0) then
            call weno5_coeffs_left(U_d(k-2,j), U_d(k-1,j), U_d(k,j), U_d(k+1,j), U_d(k+2,j), &
                                   c0, c1, c2, c3, c4)
        else
            call weno5_coeffs_right(U_d(k-1,j), U_d(k,j), U_d(k+1,j), U_d(k+2,j), U_d(k+3,j), &
                                    c0, c1, c2, c3, c4)
        end if
        coef_x(j, k, 1) = c0
        coef_x(j, k, 2) = c1
        coef_x(j, k, 3) = c2
        coef_x(j, k, 4) = c3
        coef_x(j, k, 5) = c4
    end subroutine compute_weno_coeffs_x_kernel

    attributes(global) subroutine compute_weno_coeffs_y_kernel(U_d, x, coef_y, n1sub, n2sub)
        integer, value, intent(in) :: n1sub, n2sub
        real(8), device, intent(in)  :: U_d(1-nhalo:n1sub+nhalo, 1-nhalo:n2sub+nhalo)
        real(8), device, intent(in)  :: x(1-nhalo:n1sub+nhalo)
        real(8), device, intent(out) :: coef_y(n1sub, 0:n2sub, 5)
        integer :: i, k
        real(8) :: cy
        real(8) :: c0, c1, c2, c3, c4

        k = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 1  ! 0..n2sub
        i = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        if (k > n2sub .or. i < 1 .or. i > n1sub) return

        cy = +omega * (x(i) - xc_rot)
        if (cy >= 0.0d0) then
            call weno5_coeffs_left(U_d(i,k-2), U_d(i,k-1), U_d(i,k), U_d(i,k+1), U_d(i,k+2), &
                                   c0, c1, c2, c3, c4)
        else
            call weno5_coeffs_right(U_d(i,k-1), U_d(i,k), U_d(i,k+1), U_d(i,k+2), U_d(i,k+3), &
                                    c0, c1, c2, c3, c4)
        end if
        coef_y(i, k, 1) = c0
        coef_y(i, k, 2) = c1
        coef_y(i, k, 3) = c2
        coef_y(i, k, 4) = c3
        coef_y(i, k, 5) = c4
    end subroutine compute_weno_coeffs_y_kernel

    !--------------------------------------------------------------------------
    ! Build (I - gamma*dt*L_x) in m=3 block form from pre-computed WENO coefs.
    !
    ! Semi-discrete at row i (for c_x >= 0, left-biased):
    !   L_x[i, i+off] entries from offsets {-3,-2,-1,0,+1,+2}
    !
    !   Let A_{k,1..5} = coef_x(j, k, 1..5) be the 5 coefficients of f_hat at
    !   interface k+1/2 for row j.  For c>=0 they multiply cells at offsets
    !   [-2,-1,0,+1,+2] relative to k.  For c<0 offsets [-1,0,+1,+2,+3].
    !
    !   Row i semi-discrete:
    !     L_x[i, i+off] = -(c/h) * ( contribution_from(f_hat_{i+1/2})
    !                              - contribution_from(f_hat_{i-1/2}) )
    !
    ! Block I covers rows {3I-2, 3I-1, 3I}.  The 6-wide stencil fits into
    ! {block I-1, I, I+1}.  trA, trB, trC are m x m = 3 x 3 blocks.
    !--------------------------------------------------------------------------
    attributes(global) subroutine build_matrix_x_kernel_weno5(coef_x, y, n1sub, n2sub, dx, gdt, trA, trB, trC)
        integer, value, intent(in) :: n1sub, n2sub
        real(8), device, intent(in)  :: coef_x(n2sub, 0:n1sub, 5)
        real(8), device, intent(in)  :: y(1-nhalo:n2sub+nhalo)
        real(8), value, intent(in)   :: dx, gdt
        real(8), device, intent(out) :: trA(n2sub, n1sub/m, m, m)
        real(8), device, intent(out) :: trB(n2sub, n1sub/m, m, m)
        real(8), device, intent(out) :: trC(n2sub, n1sub/m, m, m)

        integer :: j, iblk, irow, icol, i_phys, i_p, i_m
        integer :: off, pos_in_block, block_offset
        real(8) :: cx, coef_base
        real(8) :: Lval(-3:3)    ! L_x[i, i+off] for off in [-3, +3]
        real(8) :: a_p(5), a_m(5)
        integer :: mult
        integer :: k_off, tgt_block
        integer :: iv, jv
        integer :: off_base_p, off_base_m

        iblk = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j    = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        if (iblk > n1sub/m .or. j > n2sub) return

        cx = -omega * (y(j) - yc_rot)
        ! For c>=0, interface coefs multiply cells at offsets [-2,-1,0,+1,+2] relative to the
        !   interface index k; for c<0 they multiply offsets [-1,0,+1,+2,+3].
        if (cx >= 0.0d0) then
            off_base_p = -2;  off_base_m = -2
        else
            off_base_p = -1;  off_base_m = -1
        end if

        ! Initialise blocks to zero
        do jv = 1, m
          do iv = 1, m
            trA(j, iblk, iv, jv) = 0.0d0
            trB(j, iblk, iv, jv) = 0.0d0
            trC(j, iblk, iv, jv) = 0.0d0
          end do
        end do
        ! Identity on B diagonal
        do iv = 1, m
          trB(j, iblk, iv, iv) = 1.0d0
        end do

        coef_base = -gdt * cx / dx     ! prefactor -gamma*dt*c/dx applied per entry

        ! For each row within this block
        do irow = 1, m
            i_phys = (iblk - 1) * m + irow   ! global x-index of this row
            i_p = i_phys                      ! interface i+1/2 uses coef_x at index i_phys
            i_m = i_phys - 1                  ! interface i-1/2 uses coef_x at index i_phys-1

            ! Load 5 coefs for each interface
            a_p(1) = coef_x(j, i_p, 1); a_p(2) = coef_x(j, i_p, 2); a_p(3) = coef_x(j, i_p, 3)
            a_p(4) = coef_x(j, i_p, 4); a_p(5) = coef_x(j, i_p, 5)
            a_m(1) = coef_x(j, i_m, 1); a_m(2) = coef_x(j, i_m, 2); a_m(3) = coef_x(j, i_m, 3)
            a_m(4) = coef_x(j, i_m, 4); a_m(5) = coef_x(j, i_m, 5)

            ! Reset Lval
            do off = -3, 3;  Lval(off) = 0.0d0;  end do

            ! f_hat_{i+1/2} contributes with +1 sign; cells at offsets (off_base_p + 0..4) relative to i_p
            ! Global offset from row i_phys: cell index = i_p + (off_base_p + k) = i_phys + (off_base_p + k)
            do k_off = 1, 5
                off = off_base_p + (k_off - 1)
                Lval(off) = Lval(off) + a_p(k_off)    ! coefficient of f_hat_{i+1/2}
            end do
            ! f_hat_{i-1/2} contributes with -1 sign; cells relative to i_m = i_phys-1
            ! Global offset from row i_phys: (i_m - i_phys) + (off_base_m + k) = -1 + off_base_m + k
            do k_off = 1, 5
                off = -1 + off_base_m + (k_off - 1)
                Lval(off) = Lval(off) - a_m(k_off)
            end do
            ! Multiply by -c/dx to get L[i, i+off] entries
            do off = -3, 3
                Lval(off) = coef_base * Lval(off) * (-1.0d0)
                ! Wait: Lval currently stores (f_hat_{i+1/2} coef - f_hat_{i-1/2} coef).
                ! L_x[i, i+off] = -(c/h) * (f_hat_{i+1/2} - f_hat_{i-1/2}) coefficient
                !               = -(c/h) * Lval(off)  (before adjustment)
                ! So L_x[i, i+off] = (-c/h) * Lval
                ! And M[i, i+off] = -gdt * L_x[i, i+off] = -gdt * (-c/h) * Lval = (gdt*c/h) * Lval
                ! coef_base was defined as -gdt*c/dx, so we want to use -coef_base * Lval.
                ! Re-express: set Lval(off) to represent M[i, i+off] (off-diagonal only; diagonal +1 added separately)
            end do
            ! Actually clean rewrite:
            !   M[i, i+off] = -gdt * L_x[i, i+off] + (1 if off==0 else 0)
            !   L_x[i, i+off] = -(c/dx) * (f_hat_{i+1/2}_coef_of_u_{i+off} - f_hat_{i-1/2}_coef_of_u_{i+off})
            !   M[i, i+off] = (gdt*c/dx) * (...diff...) + delta_{off,0}
            ! Above we set Lval(off) = (diff)*(-1)*coef_base where coef_base = -gdt*c/dx
            !              = diff * (gdt*c/dx)
            ! So Lval(off) now represents M[i, i+off] (off-diagonal); diagonal identity already set.
            ! We just need to distribute Lval into the block structure.

            ! Determine absolute column indices and dispatch into (trA, trB, trC)[irow, local_col]
            do off = -3, 3
                if (Lval(off) == 0.0d0) cycle
                call dispatch_block_entry(trA, trB, trC, j, iblk, irow, off, Lval(off), n1sub, n2sub)
            end do
        end do
    end subroutine build_matrix_x_kernel_weno5

    !--------------------------------------------------------------------------
    ! Dispatch helper: given global stencil offset "off" (column = i_phys + off),
    ! determine which of trA (I-1), trB (I), trC (I+1) to write into, and at
    ! which (irow, icol) inside the 3x3 block.
    !--------------------------------------------------------------------------
    attributes(device) subroutine dispatch_block_entry(trA, trB, trC, j, iblk, irow, off, val, n1sub, n2sub)
        integer, intent(in) :: j, iblk, irow, off
        integer, intent(in) :: n1sub, n2sub
        real(8), intent(in) :: val
        real(8), device, intent(inout) :: trA(n2sub, n1sub/m, m, m)
        real(8), device, intent(inout) :: trB(n2sub, n1sub/m, m, m)
        real(8), device, intent(inout) :: trC(n2sub, n1sub/m, m, m)
        integer :: icol_in_row, tgt_block, local_icol
        integer :: iblk_tgt, iv

        ! Column in local row: physical column = (iblk-1)*m + irow + off
        ! Target block index = ceil(col / m) = ((col-1)/m) + 1
        ! (All offsets from {-3..+2} or {-2..+3} fit within I-1, I, I+1 for m=3)
        icol_in_row = irow + off     ! relative column within "block I local numbering"
        ! icol_in_row can range from irow-3 to irow+3. Mapping to (block, local col):
        if (icol_in_row >= 1 .and. icol_in_row <= m) then
            trB(j, iblk, irow, icol_in_row) = trB(j, iblk, irow, icol_in_row) + val
        else if (icol_in_row < 1) then
            ! In block I-1:  local col = icol_in_row + m
            trA(j, iblk, irow, icol_in_row + m) = trA(j, iblk, irow, icol_in_row + m) + val
        else
            ! icol_in_row > m:  in block I+1,  local col = icol_in_row - m
            trC(j, iblk, irow, icol_in_row - m) = trC(j, iblk, irow, icol_in_row - m) + val
        end if
    end subroutine dispatch_block_entry

    !--------------------------------------------------------------------------
    ! Build (I - gamma*dt*L_y) in m=3 block form for y-sweep.
    ! Layout: trA(n1sub, n2sub/m, m, m).
    !--------------------------------------------------------------------------
    attributes(global) subroutine build_matrix_y_kernel_weno5(coef_y, x, n1sub, n2sub, dy, gdt, trA, trB, trC)
        integer, value, intent(in) :: n1sub, n2sub
        real(8), device, intent(in)  :: coef_y(n1sub, 0:n2sub, 5)
        real(8), device, intent(in)  :: x(1-nhalo:n1sub+nhalo)
        real(8), value, intent(in)   :: dy, gdt
        real(8), device, intent(out) :: trA(n1sub, n2sub/m, m, m)
        real(8), device, intent(out) :: trB(n1sub, n2sub/m, m, m)
        real(8), device, intent(out) :: trC(n1sub, n2sub/m, m, m)

        integer :: i, jblk, jrow, j_phys, j_p, j_m
        integer :: off, k_off, iv, jv
        integer :: off_base_p, off_base_m
        real(8) :: cy, coef_base
        real(8) :: Lval(-3:3)
        real(8) :: a_p(5), a_m(5)
        integer :: icol_in_row

        jblk = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        i    = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        if (jblk > n2sub/m .or. i > n1sub) return

        cy =  omega * (x(i) - xc_rot)
        if (cy >= 0.0d0) then
            off_base_p = -2;  off_base_m = -2
        else
            off_base_p = -1;  off_base_m = -1
        end if

        do jv = 1, m
          do iv = 1, m
            trA(i, jblk, iv, jv) = 0.0d0
            trB(i, jblk, iv, jv) = 0.0d0
            trC(i, jblk, iv, jv) = 0.0d0
          end do
        end do
        do iv = 1, m
          trB(i, jblk, iv, iv) = 1.0d0
        end do

        coef_base = gdt * cy / dy      ! sign: see derivation below

        do jrow = 1, m
            j_phys = (jblk - 1) * m + jrow
            j_p = j_phys
            j_m = j_phys - 1
            a_p(1)=coef_y(i,j_p,1); a_p(2)=coef_y(i,j_p,2); a_p(3)=coef_y(i,j_p,3)
            a_p(4)=coef_y(i,j_p,4); a_p(5)=coef_y(i,j_p,5)
            a_m(1)=coef_y(i,j_m,1); a_m(2)=coef_y(i,j_m,2); a_m(3)=coef_y(i,j_m,3)
            a_m(4)=coef_y(i,j_m,4); a_m(5)=coef_y(i,j_m,5)

            do off = -3, 3;  Lval(off) = 0.0d0;  end do
            do k_off = 1, 5
                off = off_base_p + (k_off - 1)
                Lval(off) = Lval(off) + a_p(k_off)
            end do
            do k_off = 1, 5
                off = -1 + off_base_m + (k_off - 1)
                Lval(off) = Lval(off) - a_m(k_off)
            end do
            ! Convert Lval (currently sum of f_hat coefficient differences) to M[i,i+off]:
            ! L_y[i, i+off] = -(c/h)*(sum) = -(cy/dy) * Lval
            ! M[i, i+off] = -gdt * L_y = gdt*cy/dy * Lval = coef_base * Lval
            do off = -3, 3
                Lval(off) = coef_base * Lval(off)
            end do

            do off = -3, 3
                if (Lval(off) == 0.0d0) cycle
                icol_in_row = jrow + off
                if (icol_in_row >= 1 .and. icol_in_row <= m) then
                    trB(i, jblk, jrow, icol_in_row) = trB(i, jblk, jrow, icol_in_row) + Lval(off)
                else if (icol_in_row < 1) then
                    trA(i, jblk, jrow, icol_in_row + m) = trA(i, jblk, jrow, icol_in_row + m) + Lval(off)
                else
                    trC(i, jblk, jrow, icol_in_row - m) = trC(i, jblk, jrow, icol_in_row - m) + Lval(off)
                end if
            end do
        end do
    end subroutine build_matrix_y_kernel_weno5

    !==========================================================================
    ! "Explicit K via matvec" support — replaces the buggy "free K" formula
    ! K_i = (U_i - rhs_i)/(gamma*dt) which is invalid under ADI splitting.
    !
    ! Mathematical setup:
    !   ADI:  Mx My U_i = rhs_i  with  Mx = I - gamma*dt*L_x,  My = I - gamma*dt*L_y
    !   After x-sweep   : Mx U_* = rhs_i  (save U_*)
    !   After y-sweep   : My U_i = U_*    (so L_y U_i = (U_i - U_*)/gamma_dt  EXACT)
    !   Need separately : L_x U_i = (U_i - Mx U_i)/gamma_dt  via matvec on Mx
    !   Then            : K_i = L_x U_i + L_y U_i  (matches Python's K[i] = A_mat @ U_i)
    !==========================================================================

    !--------------------------------------------------------------------------
    ! Copy x-block (without ghost) into the ghost-padded buffer (physical region).
    ! Ghost slots iblk=0 and iblk=nblk_x+1 are filled by exchange_xblock_halo.
    !--------------------------------------------------------------------------
    attributes(global) subroutine copy_xblock_to_halo_kernel(src, dst, n1sub, n2sub)
        integer, value, intent(in) :: n1sub, n2sub
        real(8), device, intent(in)  :: src(n2sub, n1sub/m, m)
        real(8), device, intent(out) :: dst(n2sub, 0:n1sub/m+1, m)
        integer :: j, iblk, irow

        iblk = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j    = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        if (iblk > n1sub/m .or. j > n2sub) return
        do irow = 1, m
            dst(j, iblk, irow) = src(j, iblk, irow)
        end do
    end subroutine copy_xblock_to_halo_kernel

    !--------------------------------------------------------------------------
    ! Apply L_x to U_i (in x-block layout with x-direction ghost halos):
    !   L_x U_i = (U_i - Mx U_i) / gamma_dt
    !   Mx U_i  = trA(I)*U(I-1) + trB(I)*U(I) + trC(I)*U(I+1)   (m x m block matvec)
    !
    ! Requires U_full to have valid ghost blocks at iblk=0 and iblk=nblk_x+1
    ! (call exchange_xblock_halo first).
    !--------------------------------------------------------------------------
    attributes(global) subroutine apply_Lx_xblock_kernel(trA, trB, trC, U_full, LxU, gdt, n1sub, n2sub)
        integer, value, intent(in) :: n1sub, n2sub
        real(8), value, intent(in) :: gdt
        real(8), device, intent(in)  :: trA(n2sub, n1sub/m, m, m)
        real(8), device, intent(in)  :: trB(n2sub, n1sub/m, m, m)
        real(8), device, intent(in)  :: trC(n2sub, n1sub/m, m, m)
        real(8), device, intent(in)  :: U_full(n2sub, 0:n1sub/m+1, m)
        real(8), device, intent(out) :: LxU(n2sub, n1sub/m, m)
        integer :: j, iblk, irow, jrow
        real(8) :: MxU, inv_gdt

        iblk = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j    = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        if (iblk > n1sub/m .or. j > n2sub) return

        inv_gdt = 1.0d0 / gdt
        do irow = 1, m
            MxU = 0.0d0
            do jrow = 1, m
                MxU = MxU + trA(j, iblk, irow, jrow) * U_full(j, iblk-1, jrow) &
                          + trB(j, iblk, irow, jrow) * U_full(j, iblk,   jrow) &
                          + trC(j, iblk, irow, jrow) * U_full(j, iblk+1, jrow)
            end do
            LxU(j, iblk, irow) = (U_full(j, iblk, irow) - MxU) * inv_gdt
        end do
    end subroutine apply_Lx_xblock_kernel

    !--------------------------------------------------------------------------
    ! Compute K_i = L_x U_i + L_y U_i  (faithfully reproduces Python's
    !   K[i] = A_mat @ U_i  under ADI factorization).
    !
    !   L_y U_i = (U_i - U_*) / gamma_dt   (free, since My U_i = U_*)
    !   L_x U_i provided externally as LxU (from apply_Lx_xblock_kernel)
    !--------------------------------------------------------------------------
    attributes(global) subroutine compute_K_correct_kernel(LxU_xb, U_full, U_star, gdt, i_stage, K_xb, n1sub, n2sub)
        integer, value, intent(in) :: n1sub, n2sub, i_stage
        real(8), value, intent(in) :: gdt
        real(8), device, intent(in)    :: LxU_xb(n2sub, n1sub/m, m)
        real(8), device, intent(in)    :: U_full(n2sub, 0:n1sub/m+1, m)
        real(8), device, intent(in)    :: U_star(n2sub, n1sub/m, m)
        real(8), device, intent(inout) :: K_xb(n2sub, n1sub/m, m, s_stages)
        integer :: j, iblk, irow
        real(8) :: inv_gdt, LyU

        iblk = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j    = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        if (iblk > n1sub/m .or. j > n2sub) return
        inv_gdt = 1.0d0 / gdt
        do irow = 1, m
            LyU = (U_full(j, iblk, irow) - U_star(j, iblk, irow)) * inv_gdt
            K_xb(j, iblk, irow, i_stage) = LxU_xb(j, iblk, irow) + LyU
        end do
    end subroutine compute_K_correct_kernel

end module kernels


program weno_btdma_cuda
    use cudafor
    use params
    use kernels
    use mpi
    use mod_btdma_gpu_v2
    use mpiutil
    implicit none

    integer :: nprocs, myrank, ierr
    integer :: n1sub, n2sub
    integer :: globindx_xa, globindx_xb
    integer :: globindx_ya, globindx_yb
    integer :: i, j, istep, nstep, istat, i_stage
    integer :: argc

    integer :: dev
    type(cudadeviceprop) :: prop

    type(BTDMA_PLAN_gpu_v2) :: plan_x, plan_y

    real(8) :: t_pack_a=0.d0, t_pack_b=0.d0
    real(8) :: t_weno_a=0.d0, t_weno_b=0.d0
    real(8) :: t_matrix_a=0.d0, t_matrix_b=0.d0
    real(8) :: t_stage_rhs_a=0.d0, t_stage_rhs_b=0.d0
    real(8) :: t_trans_a=0.d0, t_trans_b=0.d0
    real(8) :: t_solve_a=0.d0, t_solve_b=0.d0
    real(8) :: t_compK_a=0.d0, t_compK_b=0.d0
    real(8) :: t_comm_a=0.d0, t_comm_b=0.d0

    real(8) :: dx, dy, dt, time, t_final, CFL_target, CFL_effective, gamma_dt
    real(8) :: L1_global, Linf_global, umax_global, umin_global
    real(8) :: t_total_measured
    character(len=16) :: case_id
    character(len=512) :: output_dir, vtk_path, metrics_path
    real(8), allocatable :: x(:), y(:)
    real(8), allocatable :: xsub(:), ysub(:)
    real(8), allocatable :: U_host(:,:)

    real(8), device, allocatable :: xsub_d(:), ysub_d(:)
    real(8), device, allocatable :: U_d(:,:)
    real(8), device, allocatable :: U_n_xb_d(:,:,:)
    real(8), device, allocatable :: rhs_xb_d(:,:,:)
    real(8), device, allocatable :: K_xb_d(:,:,:,:)
    real(8), device, allocatable :: dU_x_d(:,:,:)
    real(8), device, allocatable :: dU_y_d(:,:,:)
    ! Original M matrices (built once per step, used by apply_Lx matvec)
    real(8), device, allocatable :: trA_x_d(:,:,:,:), trB_x_d(:,:,:,:), trC_x_d(:,:,:,:)
    real(8), device, allocatable :: trA_y_d(:,:,:,:), trB_y_d(:,:,:,:), trC_y_d(:,:,:,:)
    ! Scratch copies for BTDMA solve (PaScaL_BTDMA destroys A, C in-place via Gauss elim)
    real(8), device, allocatable :: trA_x_solve_d(:,:,:,:), trB_x_solve_d(:,:,:,:), trC_x_solve_d(:,:,:,:)
    real(8), device, allocatable :: trA_y_solve_d(:,:,:,:), trB_y_solve_d(:,:,:,:), trC_y_solve_d(:,:,:,:)
    real(8), device, allocatable :: coef_x_d(:,:,:), coef_y_d(:,:,:)
    ! --- Explicit-K (Python parity) buffers ---
    real(8), device, allocatable :: U_star_xb_d(:,:,:)     ! save U after x-sweep, per stage
    real(8), device, allocatable :: U_xb_halo_d(:,:,:)     ! U_i with x-direction ghost block (for matvec)
    real(8), device, allocatable :: LxU_xb_d(:,:,:)        ! L_x U_i result

    real(8)          :: Ac(s_stages, s_stages)
    ! real(8), device  :: Ac_d(s_stages, s_stages)

    type(dim3) :: bl2d, gr2d
    type(dim3) :: bl_x, gr_x, bl_y, gr_y
    type(dim3) :: bl_if_x, gr_if_x, bl_if_y, gr_if_y

    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)

    argc = command_argument_count()
    case_id = ''
    output_dir = '.'
    if (argc >= 1) call get_command_argument(1, case_id)
    if (argc >= 2) call get_command_argument(2, output_dir)

    select case (trim(case_id))
    case ('cfl1p0')
        CFL_target = 1.0d0
        nstep = 1621
    case ('cfl1p5')
        CFL_target = 1.5d0
        nstep = 1080
    case ('cfl2p0')
        CFL_target = 2.0d0
        nstep = 810
    case default
        if (myrank == 0) then
            write(*,'(A)') 'Usage: weno2d_gpu <cfl1p0|cfl1p5|cfl2p0> [output-directory]'
        end if
        call MPI_Abort(MPI_COMM_WORLD, 2, ierr)
    end select

    if (n1 /= 516 .or. n2 /= 516 .or. np1 /= 2 .or. np2 /= 2 .or. m /= 3) then
        if (myrank == 0) then
            write(*,'(A)') 'ERROR: the public manuscript driver requires N=516x516, NP=2x2, and m=3.'
        end if
        call MPI_Abort(MPI_COMM_WORLD, 2, ierr)
    end if
    if (nprocs /= np1*np2) then
        if (myrank == 0) then
            write(*,'(A,I0,A,I0,A)') 'ERROR: launched with ', nprocs, &
                ' MPI ranks; this case requires ', np1*np2, '.'
        end if
        call MPI_Abort(MPI_COMM_WORLD, 2, ierr)
    end if

    block
        integer :: local_comm, local_rank, ngpu, dev_id
        call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, local_comm, ierr)
        call MPI_Comm_rank(local_comm, local_rank, ierr)
        ierr = cudaGetDeviceCount(ngpu)
        if (ngpu <= 0) then
            call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        end if
        dev_id = mod(local_rank, ngpu)
        ierr = cudaSetDevice(dev_id)
        ierr = cudaGetDevice(dev)
        ierr = cudaGetDeviceProperties(prop, dev)
        call MPI_Comm_free(local_comm, ierr)
        if (myrank == 0) write(*,'(A,I0,A)') ' Detected ', ngpu, ' GPU(s) per node'
    end block

    np_dim(0:1) = (/np1, np2/)
    period(0) = .true.; period(1) = .true.
    call mpi_topology_make()

    n1sub = mpiutil_para(1, n1, comm_1d_x%myrank, comm_1d_x%nprocs, globindx_xa, globindx_xb)
    n2sub = mpiutil_para(1, n2, comm_1d_y%myrank, comm_1d_y%nprocs, globindx_ya, globindx_yb)

    if (mod(n1sub, m) /= 0 .or. mod(n2sub, m) /= 0) then
        if (myrank == 0) then
        end if
        call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end if

    allocate(x(1:n1), y(1:n2))
    allocate(xsub(1-nhalo:n1sub+nhalo), ysub(1-nhalo:n2sub+nhalo))
    allocate(xsub_d(1-nhalo:n1sub+nhalo), ysub_d(1-nhalo:n2sub+nhalo))
    allocate(U_host(1-nhalo:n1sub+nhalo, 1-nhalo:n2sub+nhalo))

    allocate(U_d     (1-nhalo:n1sub+nhalo, 1-nhalo:n2sub+nhalo))
    allocate(U_n_xb_d(n2sub, n1sub/m, m))
    allocate(rhs_xb_d(n2sub, n1sub/m, m))
    allocate(K_xb_d  (n2sub, n1sub/m, m, s_stages))
    allocate(dU_x_d  (n2sub, n1sub/m, m))
    allocate(dU_y_d  (n1sub, n2sub/m, m))
    allocate(trA_x_d(n2sub, n1sub/m, m, m), trB_x_d(n2sub, n1sub/m, m, m), trC_x_d(n2sub, n1sub/m, m, m))
    allocate(trA_y_d(n1sub, n2sub/m, m, m), trB_y_d(n1sub, n2sub/m, m, m), trC_y_d(n1sub, n2sub/m, m, m))
    allocate(trA_x_solve_d(n2sub, n1sub/m, m, m), trB_x_solve_d(n2sub, n1sub/m, m, m), trC_x_solve_d(n2sub, n1sub/m, m, m))
    allocate(trA_y_solve_d(n1sub, n2sub/m, m, m), trB_y_solve_d(n1sub, n2sub/m, m, m), trC_y_solve_d(n1sub, n2sub/m, m, m))
    allocate(coef_x_d(n2sub, 0:n1sub, 5))
    allocate(coef_y_d(n1sub, 0:n2sub, 5))
    ! Explicit-K buffers for SDIRK4 + ADI
    allocate(U_star_xb_d(n2sub, n1sub/m, m))
    allocate(U_xb_halo_d(n2sub, 0:n1sub/m+1, m))
    allocate(LxU_xb_d(n2sub, n1sub/m, m))

    K_xb_d = 0.0d0

    Ac = 0.0d0
    Ac(1,1) = gamma_rk
    Ac(2,1) = 1.0d0/2.0d0;       Ac(2,2) = gamma_rk
    Ac(3,1) = 17.0d0/50.0d0;     Ac(3,2) = -1.0d0/25.0d0;      Ac(3,3) = gamma_rk
    Ac(4,1) = 371.0d0/1360.0d0;  Ac(4,2) = -137.0d0/2720.0d0;  Ac(4,3) = 15.0d0/544.0d0;   Ac(4,4) = gamma_rk
    Ac(5,1) = 25.0d0/24.0d0
    Ac(5,2) = -49.0d0/48.0d0
    Ac(5,3) = 125.0d0/16.0d0
    Ac(5,4) = -85.0d0/12.0d0
    Ac(5,5) = gamma_rk
    Ac_d = Ac

    dx = Lx / dble(n1); dy = Ly / dble(n2)
    do i = 1, n1;  x(i) = (dble(i) - 0.5d0) * dx;  end do
    do j = 1, n2;  y(j) = (dble(j) - 0.5d0) * dy;  end do

    do i = 1-nhalo, n1sub+nhalo
        if (globindx_xa + i - 1 >= 1 .and. globindx_xa + i - 1 <= n1) then
            xsub(i) = x(globindx_xa + i - 1)
        else if (globindx_xa + i - 1 < 1) then
            xsub(i) = x(n1 + globindx_xa + i - 1) - Lx
        else
            xsub(i) = x(globindx_xa + i - 1 - n1) + Lx
        end if
    end do
    do j = 1-nhalo, n2sub+nhalo
        if (globindx_ya + j - 1 >= 1 .and. globindx_ya + j - 1 <= n2) then
            ysub(j) = y(globindx_ya + j - 1)
        else if (globindx_ya + j - 1 < 1) then
            ysub(j) = y(n2 + globindx_ya + j - 1) - Ly
        else
            ysub(j) = y(globindx_ya + j - 1 - n2) + Ly
        end if
    end do
    xsub_d = xsub; ysub_d = ysub

    t_final = T_period
    ! The case selector above fixes the manuscript step count.  The final step
    ! is not shortened independently; a uniform dt reaches exactly one period.
    dt = t_final / dble(nstep)
    CFL_effective = dt * (omega * 0.5d0) / min(dx, dy)
    gamma_dt = gamma_rk * dt

    bl2d  = dim3(16, 16, 1);  gr2d  = dim3((n1sub+15)/16, (n2sub+15)/16, 1)
    bl_x  = dim3(16, 16, 1);  gr_x  = dim3(((n1sub/m)+15)/16, (n2sub+15)/16, 1)
    bl_y  = dim3(16, 16, 1);  gr_y  = dim3(((n2sub/m)+15)/16, (n1sub+15)/16, 1)
    bl_if_x = dim3(16, 16, 1); gr_if_x = dim3((n1sub+16+15)/16, (n2sub+15)/16, 1)
    bl_if_y = dim3(16, 16, 1); gr_if_y = dim3((n2sub+16+15)/16, (n1sub+15)/16, 1)

    call init_zalesak_kernel<<<gr2d, bl2d>>>(U_d, xsub_d, ysub_d, n1sub, n2sub)
    istat = cudaDeviceSynchronize()
    call exchange_halo_2d_gpu(U_d, n1sub, n2sub)

    call btdma_makeplan_gpu_v2(plan_x, m, n2sub, n1sub/m, comm_1d_x%mpi_comm)
    call btdma_makeplan_gpu_v2(plan_y, m, n1sub, n2sub/m, comm_1d_y%mpi_comm)

    if (myrank == 0) then
        write(*,'(A)')  " ========== 2D Zalesak / WENO5 (frozen) + SDIRK4 + m=3 BTDMA =========="
        write(*,'(A,A)')       " case            = ", trim(case_id)
        write(*,'(A,I8,I8)')  " N1, N2         = ", n1, n2
        write(*,'(A,I8,I8)')  " NP1, NP2       = ", np1, np2
        write(*,'(A,I8)')     " block size m   = ", m
        write(*,'(A,I8)')     " halo nhalo     = ", nhalo
        write(*,'(A,I8)')     " nblk (x, per rank) = ", n1sub/m
        write(*,'(A,I8)')     " SDIRK4 stages  = ", s_stages
        write(*,'(A,F12.6)')  " gamma (SDIRK)  = ", gamma_rk
        write(*,'(A,F12.5)')  " CFL_nominal    = ", CFL_target
        write(*,'(A,F12.8)')  " CFL_effective  = ", CFL_effective
        write(*,'(A,E12.4)')  " dt             = ", dt
        write(*,'(A,E12.4)')  " gamma * dt     = ", gamma_dt
        write(*,'(A,I8)')     " nstep          = ", nstep
    end if

    time = 0.0d0
    do istep = 1, nstep
        time = time + dt

        ! Pack U^n
        if (istep > 1) t_pack_a = MPI_WTIME()
        call pack_scalar_to_xblock_kernel<<<gr2d, bl2d>>>(U_d, U_n_xb_d, n1sub, n2sub)
        istat = cudaDeviceSynchronize()
        if (istep > 1) t_pack_b = t_pack_b + (MPI_WTIME() - t_pack_a)

        ! Compute frozen WENO5 stencil coefs at every interface (once per step)
        if (istep > 1) t_weno_a = MPI_WTIME()
        call compute_weno_coeffs_x_kernel<<<gr_if_x, bl_if_x>>>(U_d, ysub_d, coef_x_d, n1sub, n2sub)
        call compute_weno_coeffs_y_kernel<<<gr_if_y, bl_if_y>>>(U_d, xsub_d, coef_y_d, n1sub, n2sub)
        istat = cudaDeviceSynchronize()
        if (istep > 1) t_weno_b = t_weno_b + (MPI_WTIME() - t_weno_a)

        ! Build SDIRK4 matrices (I - gamma*dt*L_x), (I - gamma*dt*L_y) in m=3 blocks
        if (istep > 1) t_matrix_a = MPI_WTIME()
        call build_matrix_x_kernel_weno5<<<gr_x, bl_x>>>(coef_x_d, ysub_d, n1sub, n2sub, dx, gamma_dt, trA_x_d, trB_x_d, trC_x_d)
        call build_matrix_y_kernel_weno5<<<gr_y, bl_y>>>(coef_y_d, xsub_d, n1sub, n2sub, dy, gamma_dt, trA_y_d, trB_y_d, trC_y_d)
        istat = cudaDeviceSynchronize()
        if (istep > 1) t_matrix_b = t_matrix_b + (MPI_WTIME() - t_matrix_a)

        ! SDIRK4 stage loop
        do i_stage = 1, s_stages
            if (istep > 1) t_stage_rhs_a = MPI_WTIME()
            call stage_rhs_xblock_kernel<<<gr_x, bl_x>>>(U_n_xb_d, K_xb_d, Ac_d, i_stage, dt, rhs_xb_d, n1sub, n2sub)
            istat = cudaDeviceSynchronize()
            if (istep > 1) t_stage_rhs_b = t_stage_rhs_b + (MPI_WTIME() - t_stage_rhs_a)

            call copy_xblock_kernel<<<gr_x, bl_x>>>(rhs_xb_d, dU_x_d, n1sub, n2sub)
            istat = cudaDeviceSynchronize()

            ! ---- X-sweep: Mx U_* = rhs_i ----
            ! NOTE: PaScaL_BTDMA destroys A and C in-place during Gauss elimination.
            ! Copy original M_x into scratch buffers so the originals survive for apply_Lx.
            trA_x_solve_d = trA_x_d
            trB_x_solve_d = trB_x_d
            trC_x_solve_d = trC_x_d
            if (istep > 1) t_solve_a = MPI_WTIME()
            call btdma_many_cycl_mpi_gpu_v2(trA_x_solve_d, trB_x_solve_d, trC_x_solve_d, dU_x_d, m, n2sub, n1sub/m, plan_x)
            istat = cudaDeviceSynchronize()
            if (istep > 1) t_solve_b = t_solve_b + (MPI_WTIME() - t_solve_a)

            ! ---- Save U_* (for L_y U_i = (U_i - U_*)/gamma_dt later) ----
            call copy_xblock_kernel<<<gr_x, bl_x>>>(dU_x_d, U_star_xb_d, n1sub, n2sub)
            istat = cudaDeviceSynchronize()

            if (istep > 1) t_trans_a = MPI_WTIME()
            call transpose_xy_d_kernel<<<gr2d, bl2d>>>(dU_x_d, dU_y_d, n1sub, n2sub)
            istat = cudaDeviceSynchronize()
            if (istep > 1) t_trans_b = t_trans_b + (MPI_WTIME() - t_trans_a)

            ! ---- Y-sweep: My U_i = U_* ----
            trA_y_solve_d = trA_y_d
            trB_y_solve_d = trB_y_d
            trC_y_solve_d = trC_y_d
            if (istep > 1) t_solve_a = MPI_WTIME()
            call btdma_many_cycl_mpi_gpu_v2(trA_y_solve_d, trB_y_solve_d, trC_y_solve_d, dU_y_d, m, n1sub, n2sub/m, plan_y)
            istat = cudaDeviceSynchronize()
            if (istep > 1) t_solve_b = t_solve_b + (MPI_WTIME() - t_solve_a)

            if (istep > 1) t_trans_a = MPI_WTIME()
            call transpose_yx_d_kernel<<<gr2d, bl2d>>>(dU_y_d, dU_x_d, n1sub, n2sub)
            istat = cudaDeviceSynchronize()
            if (istep > 1) t_trans_b = t_trans_b + (MPI_WTIME() - t_trans_a)
            ! dU_x_d now holds U_i (final stage solution)

            ! ---- K_i = L_x U_i + L_y U_i  (matches Python's K[i] = A_mat @ U_i) ----
            !   Step 1: copy U_i into halo-padded buffer
            !   Step 2: exchange ghost blocks across x-direction MPI ranks
            !   Step 3: matvec  L_x U_i = (U_i - Mx U_i)/gamma_dt
            !   Step 4: K_i = L_x U_i + (U_i - U_*)/gamma_dt
            if (istep > 1) t_compK_a = MPI_WTIME()
            call copy_xblock_to_halo_kernel<<<gr_x, bl_x>>>(dU_x_d, U_xb_halo_d, n1sub, n2sub)
            istat = cudaDeviceSynchronize()
            call exchange_xblock_halo(U_xb_halo_d, n1sub, n2sub)
            call apply_Lx_xblock_kernel<<<gr_x, bl_x>>>(trA_x_d, trB_x_d, trC_x_d, &
                                                       U_xb_halo_d, LxU_xb_d, gamma_dt, n1sub, n2sub)
            istat = cudaDeviceSynchronize()
            call compute_K_correct_kernel<<<gr_x, bl_x>>>(LxU_xb_d, U_xb_halo_d, U_star_xb_d, &
                                                         gamma_dt, i_stage, K_xb_d, n1sub, n2sub)
            istat = cudaDeviceSynchronize()
            if (istep > 1) t_compK_b = t_compK_b + (MPI_WTIME() - t_compK_a)
        end do

        ! U^{n+1} = U_s  (last stage); unpack x-block -> scalar
        if (istep > 1) t_pack_a = MPI_WTIME()
        call unpack_xblock_to_scalar_kernel<<<gr2d, bl2d>>>(dU_x_d, U_d, n1sub, n2sub)
        istat = cudaDeviceSynchronize()
        if (istep > 1) t_pack_b = t_pack_b + (MPI_WTIME() - t_pack_a)

        ! Halo exchange for next step's WENO weight computation
        if (istep > 1) t_comm_a = MPI_WTIME()
        call exchange_halo_2d_gpu(U_d, n1sub, n2sub)
        if (istep > 1) t_comm_b = t_comm_b + (MPI_WTIME() - t_comm_a)
    end do

    ! Error vs initial (after 1 rotation = IC)
    block
        real(8) :: L1_loc, Linf_loc
        real(8) :: umax_loc, umin_loc
        real(8), device, allocatable :: U_init_d(:,:)
        real(8), allocatable :: U_num_host(:,:), U_init_host(:,:)

        allocate(U_init_d(1-nhalo:n1sub+nhalo, 1-nhalo:n2sub+nhalo))
        allocate(U_num_host(1-nhalo:n1sub+nhalo, 1-nhalo:n2sub+nhalo))
        allocate(U_init_host(1-nhalo:n1sub+nhalo, 1-nhalo:n2sub+nhalo))
        call init_zalesak_kernel<<<gr2d, bl2d>>>(U_init_d, xsub_d, ysub_d, n1sub, n2sub)
        istat = cudaDeviceSynchronize()

        U_num_host  = U_d
        U_init_host = U_init_d

        L1_loc = 0.0d0; Linf_loc = 0.0d0
        umax_loc = -1.0d99; umin_loc = 1.0d99
        do j = 1, n2sub
            do i = 1, n1sub
                L1_loc   = L1_loc + abs(U_num_host(i,j) - U_init_host(i,j))
                Linf_loc = max(Linf_loc, abs(U_num_host(i,j) - U_init_host(i,j)))
                umax_loc = max(umax_loc, U_num_host(i,j))
                umin_loc = min(umin_loc, U_num_host(i,j))
            end do
        end do
        L1_loc = L1_loc * dx * dy

        call MPI_Allreduce(L1_loc,   L1_global,   1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
        call MPI_Allreduce(Linf_loc, Linf_global, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr)
        call MPI_Allreduce(umax_loc, umax_global, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr)
        call MPI_Allreduce(umin_loc, umin_global, 1, MPI_DOUBLE_PRECISION, MPI_MIN, MPI_COMM_WORLD, ierr)

        deallocate(U_num_host, U_init_host, U_init_d)

        if (myrank == 0) then
            write(*,'(A)') " ================== Error / extrema =================="
            write(*,'(A,E20.12)') " L1 error     = ", L1_global
            write(*,'(A,E20.12)') " Linf error   = ", Linf_global
            write(*,'(A,F12.5)')  " max(u)       = ", umax_global
            write(*,'(A,F12.5)')  " min(u)       = ", umin_global
            write(*,'(A,F12.5)')  " final time   = ", time
        end if
    end block

    t_total_measured = t_pack_b+t_weno_b+t_matrix_b+t_stage_rhs_b+ &
                       t_trans_b+t_solve_b+t_compK_b+t_comm_b
    if (myrank == 0) then
        write(*,'(A)') " ============ Rank-0 phase times (s, excl. step 1) =============="
        write(*,'(9(A15,A1))') "t_total","|","t_pack","|","t_weno","|","t_matrix","|",&
                                "t_stage_rhs","|","t_trans","|","t_solve","|","t_compK","|","t_comm","|"
        write(*,'(9(E15.8,A1))') &
            t_total_measured, "|", &
            t_pack_b,"|",t_weno_b,"|",t_matrix_b,"|",t_stage_rhs_b,"|",t_trans_b,"|",t_solve_b,"|",t_compK_b,"|",t_comm_b,"|"
    end if

    !--------------------------------------------------------------------------
    ! VTK output for ParaView (single gathered file at rank 0)
    !   output_weno5_<case>_516x516_final.vtk contains 3 fields:
    !     u_final    : solution after one rotation
    !     u_initial  : Zalesak IC (for comparison)
    !     u_error    : u_final - u_initial
    !   DATASET: STRUCTURED_POINTS (regular 2D grid, cell-centred)
    !--------------------------------------------------------------------------
    block
        integer :: funit, ii, jj, r
        integer :: coords(0:1), ierr2
        real(8) :: xg, yg, rr
        logical :: in_disk, in_slot
        real(8), allocatable :: U_local_flat(:)
        real(8), allocatable :: U_global_flat(:)
        real(8), allocatable :: U_global(:,:), U_init_global(:,:)

        ! 1. Copy device -> host
        U_host = U_d

        ! 2. Pack local physical cells into contiguous flat buffer
        allocate(U_local_flat(n1sub*n2sub))
        do jj = 1, n2sub
            do ii = 1, n1sub
                U_local_flat((jj-1)*n1sub + ii) = U_host(ii, jj)
            end do
        end do

        ! 3. Gather to rank 0
        if (myrank == 0) then
            allocate(U_global_flat(nprocs * n1sub * n2sub))
        else
            allocate(U_global_flat(1))
        end if
        call MPI_Gather(U_local_flat,  n1sub*n2sub, MPI_DOUBLE_PRECISION, &
                        U_global_flat, n1sub*n2sub, MPI_DOUBLE_PRECISION, &
                        0, MPI_COMM_WORLD, ierr)

        ! 4. Rank 0 reassembles global 2D array using Cartesian coords
        if (myrank == 0) then
            allocate(U_global(n1, n2))
            do r = 0, nprocs - 1
                call MPI_Cart_coords(mpi_world_cart, r, 2, coords, ierr2)
                do jj = 1, n2sub
                    do ii = 1, n1sub
                        U_global(coords(0)*n1sub + ii, coords(1)*n2sub + jj) = &
                            U_global_flat(r*n1sub*n2sub + (jj-1)*n1sub + ii)
                    end do
                end do
            end do

            ! 5. Rank 0 regenerates Zalesak IC on the global grid (for comparison fields)
            allocate(U_init_global(n1, n2))
            do jj = 1, n2
                do ii = 1, n1
                    xg = (dble(ii) - 0.5d0) * dx
                    yg = (dble(jj) - 0.5d0) * dy
                    rr = sqrt((xg - zal_x0)**2 + (yg - zal_y0)**2)
                    in_disk = (rr < zal_R)
                    in_slot = (abs(xg - zal_x0) <= zal_slot_hw) .and. (yg <= zal_slot_top)
                    if (in_disk .and. .not. in_slot) then
                        U_init_global(ii, jj) = 1.0d0
                    else
                        U_init_global(ii, jj) = 0.0d0
                    end if
                end do
            end do

            ! 6. Write legacy VTK (ASCII, STRUCTURED_POINTS) for ParaView
            write(vtk_path, '(A,"/output_weno5_",A,"_",I0,"x",I0,"_final.vtk")') &
                trim(output_dir), trim(case_id), n1, n2
            open(newunit=funit, file=trim(vtk_path), status='replace')

            write(funit,'(A)') '# vtk DataFile Version 3.0'
            write(funit,'(A,I0,A,I0,A,F7.4,A,F5.2)') &
                '2D Zalesak WENO5+SDIRK4 m=3 N=', n1, 'x', n2, ' t=', time, ' CFL=', CFL_target
            write(funit,'(A)') 'ASCII'
            write(funit,'(A)') 'DATASET STRUCTURED_POINTS'
            write(funit,'(A,3(I0,1X))') 'DIMENSIONS ', n1, n2, 1
            write(funit,'(A,3(E14.6,1X))') 'ORIGIN ', 0.5d0*dx, 0.5d0*dy, 0.0d0
            write(funit,'(A,3(E14.6,1X))') 'SPACING ', dx, dy, 1.0d0
            write(funit,'(A,I0)') 'POINT_DATA ', n1*n2

            ! Field 1: u_final
            write(funit,'(A)') 'SCALARS u_final double 1'
            write(funit,'(A)') 'LOOKUP_TABLE default'
            do jj = 1, n2
                do ii = 1, n1
                    write(funit,'(E16.8)') U_global(ii, jj)
                end do
            end do

            ! Field 2: u_initial
            write(funit,'(A)') 'SCALARS u_initial double 1'
            write(funit,'(A)') 'LOOKUP_TABLE default'
            do jj = 1, n2
                do ii = 1, n1
                    write(funit,'(E16.8)') U_init_global(ii, jj)
                end do
            end do

            ! Field 3: u_error (u_final - u_initial)
            write(funit,'(A)') 'SCALARS u_error double 1'
            write(funit,'(A)') 'LOOKUP_TABLE default'
            do jj = 1, n2
                do ii = 1, n1
                    write(funit,'(E16.8)') U_global(ii, jj) - U_init_global(ii, jj)
                end do
            end do

            close(funit)
            write(*,'(A,A)') ' VTK output written: ', trim(vtk_path)

            deallocate(U_global, U_init_global)
        end if

        deallocate(U_local_flat, U_global_flat)
    end block

    if (myrank == 0) then
        block
            integer :: funit
            write(metrics_path, '(A,"/metrics_",A,".csv")') trim(output_dir), trim(case_id)
            open(newunit=funit, file=trim(metrics_path), status='replace')
            write(funit,'(A)') 'case_id,cfl_nominal,cfl_effective,nstep,dt,nx,ny,npx,npy,gpus,' // &
                'l1_error,linf_error,u_min,u_max,rank0_phase_sum_seconds,' // &
                'rank0_solve_seconds,vtk_file'
            write(funit,'(A,",",F4.1,",",ES24.16E3,",",I0,",",ES24.16E3,",",' // &
                        'I0,",",I0,",",I0,",",I0,",",I0,",",ES24.16E3,",",' // &
                        'ES24.16E3,",",ES24.16E3,",",ES24.16E3,",",ES24.16E3,",",' // &
                        'ES24.16E3,",",A)') &
                trim(case_id), CFL_target, CFL_effective, nstep, dt, n1, n2, np1, np2, nprocs, &
                L1_global, Linf_global, umin_global, umax_global, t_total_measured, t_solve_b, trim(vtk_path)
            close(funit)
            write(*,'(A,A)') ' Metrics written:    ', trim(metrics_path)
        end block
    end if

    call btdma_cleanplan_gpu_v2(plan_x)
    call btdma_cleanplan_gpu_v2(plan_y)

    deallocate(U_d, U_n_xb_d, rhs_xb_d, K_xb_d, dU_x_d, dU_y_d)
    deallocate(trA_x_d, trB_x_d, trC_x_d, trA_y_d, trB_y_d, trC_y_d)
    deallocate(coef_x_d, coef_y_d)
    deallocate(U_star_xb_d, U_xb_halo_d, LxU_xb_d)
    deallocate(x, y, xsub, ysub, xsub_d, ysub_d, U_host)

    call mpi_topology_clean()
    call MPI_FINALIZE(ierr)
end program weno_btdma_cuda
