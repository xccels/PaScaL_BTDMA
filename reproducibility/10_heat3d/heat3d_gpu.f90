!===============================================================================
! 3D Heat Equation Solver - Douglas-Gunn ADI (4th order)
! Multi-GPU implementation using PaScaL_BTDMA
!
! u_t = alpha * (u_xx + u_yy + u_zz),  (x,y,z) in [0,L]^3,  Dirichlet BC
! Spatial: O(h^4),  Temporal: O(dt^2),  Unconditionally stable
! Exact solution: u = sin(pi*x/L)*sin(pi*y/L)*sin(pi*z/L)*exp(-3*alpha*(pi/L)^2*t)
!
! Douglas-Gunn 3-sweep ADI:
!   Step 1 (x): (I - A_x) u*       = (I + A_x + 2A_y + 2A_z) u^n
!   Step 2 (y): (I - A_y) u**      = u*  - A_y * u^n
!   Step 3 (z): (I - A_z) u^{n+1}  = u** - A_z * u^n
!===============================================================================

module params
    use mpi
    use cudafor
    implicit none

    integer, parameter :: nx = NX_VAL
    integer, parameter :: ny = NY_VAL
    integer, parameter :: nz = NZ_VAL
    integer, parameter :: np1 = NP1
    integer, parameter :: np2 = NP2
    integer, parameter :: np3 = NP3
    integer, parameter :: mb = 2
    integer, parameter :: nrun = NRUN
    integer, parameter :: nref = NREF_VAL

    real(8), parameter :: pi = 3.141592653589793d0
    real(8), parameter :: Lx = 1.0d0
    real(8), parameter :: Ly = 1.0d0
    real(8), parameter :: Lz = 1.0d0
    real(8), parameter :: alpha_diff = 1.0d0

    integer, parameter :: nhalo = 2

    integer, public :: mpi_world_cart
    integer, public :: np_dim(0:2)
    logical, public :: period(0:2)

    type, public :: cart_comm_1d
        integer :: myrank, nprocs
        integer :: west_rank, east_rank
        integer :: mpi_comm
    end type cart_comm_1d

    type(cart_comm_1d), public :: comm_1d_x, comm_1d_y, comm_1d_z

contains

    subroutine mpi_topology_make()
        implicit none
        logical :: remain(0:2)
        integer :: ierr

        call MPI_Cart_create(MPI_COMM_WORLD, 3, np_dim, period, .false., mpi_world_cart, ierr)

        ! x-subcommunicator
        remain = (/.true., .false., .false./)
        call MPI_Cart_sub(mpi_world_cart, remain, comm_1d_x%mpi_comm, ierr)
        call MPI_Comm_rank(comm_1d_x%mpi_comm, comm_1d_x%myrank, ierr)
        call MPI_Comm_size(comm_1d_x%mpi_comm, comm_1d_x%nprocs, ierr)
        call MPI_Cart_shift(comm_1d_x%mpi_comm, 0, 1, comm_1d_x%west_rank, comm_1d_x%east_rank, ierr)

        ! y-subcommunicator
        remain = (/.false., .true., .false./)
        call MPI_Cart_sub(mpi_world_cart, remain, comm_1d_y%mpi_comm, ierr)
        call MPI_Comm_rank(comm_1d_y%mpi_comm, comm_1d_y%myrank, ierr)
        call MPI_Comm_size(comm_1d_y%mpi_comm, comm_1d_y%nprocs, ierr)
        call MPI_Cart_shift(comm_1d_y%mpi_comm, 0, 1, comm_1d_y%west_rank, comm_1d_y%east_rank, ierr)

        ! z-subcommunicator
        remain = (/.false., .false., .true./)
        call MPI_Cart_sub(mpi_world_cart, remain, comm_1d_z%mpi_comm, ierr)
        call MPI_Comm_rank(comm_1d_z%mpi_comm, comm_1d_z%myrank, ierr)
        call MPI_Comm_size(comm_1d_z%mpi_comm, comm_1d_z%nprocs, ierr)
        call MPI_Cart_shift(comm_1d_z%mpi_comm, 0, 1, comm_1d_z%west_rank, comm_1d_z%east_rank, ierr)
    end subroutine mpi_topology_make

    subroutine mpi_topology_clean()
        implicit none
        integer :: ierr
        call MPI_Comm_free(mpi_world_cart, ierr)
    end subroutine mpi_topology_clean

    !---------------------------------------------------------------------------
    ! Halo exchange in x-direction (3D)
    !---------------------------------------------------------------------------
    subroutine exchange_halo_x(u_d, n1sub, n2sub, n3sub)
        use mpi
        use cudafor
        implicit none
        integer, intent(in) :: n1sub, n2sub, n3sub
        real(8), device, intent(inout) :: u_d(1-nhalo:n1sub+nhalo, 1-nhalo:n2sub+nhalo, 1-nhalo:n3sub+nhalo)

        integer :: ierr, reqs(4), stats(MPI_STATUS_SIZE, 4)
        integer :: nsend, ny_ext, nz_ext
        integer :: ih, jj, kk
        real(8), device, allocatable :: sbuf_l(:), sbuf_r(:), rbuf_l(:), rbuf_r(:)

        ny_ext = n2sub + 2*nhalo
        nz_ext = n3sub + 2*nhalo
        nsend = nhalo * ny_ext * nz_ext

        allocate(sbuf_l(nsend), sbuf_r(nsend), rbuf_l(nsend), rbuf_r(nsend))

        !$cuf kernel do(3) <<<*,*>>>
        do kk = 1-nhalo, n3sub+nhalo
            do jj = 1-nhalo, n2sub+nhalo
                do ih = 1, nhalo
                    sbuf_l(((kk-(1-nhalo))*ny_ext + (jj-(1-nhalo)))*nhalo + ih) = u_d(ih, jj, kk)
                    sbuf_r(((kk-(1-nhalo))*ny_ext + (jj-(1-nhalo)))*nhalo + ih) = u_d(n1sub-nhalo+ih, jj, kk)
                end do
            end do
        end do
        ierr = cudaDeviceSynchronize()

        call MPI_Isend(sbuf_r, nsend, MPI_DOUBLE_PRECISION, comm_1d_x%east_rank, 0, comm_1d_x%mpi_comm, reqs(1), ierr)
        call MPI_Irecv(rbuf_l, nsend, MPI_DOUBLE_PRECISION, comm_1d_x%west_rank, 0, comm_1d_x%mpi_comm, reqs(2), ierr)
        call MPI_Isend(sbuf_l, nsend, MPI_DOUBLE_PRECISION, comm_1d_x%west_rank, 1, comm_1d_x%mpi_comm, reqs(3), ierr)
        call MPI_Irecv(rbuf_r, nsend, MPI_DOUBLE_PRECISION, comm_1d_x%east_rank, 1, comm_1d_x%mpi_comm, reqs(4), ierr)
        call MPI_Waitall(4, reqs, stats, ierr)

        if (comm_1d_x%west_rank /= MPI_PROC_NULL) then
            !$cuf kernel do(3) <<<*,*>>>
            do kk = 1-nhalo, n3sub+nhalo
                do jj = 1-nhalo, n2sub+nhalo
                    do ih = 1, nhalo
                        u_d(1-nhalo+ih-1, jj, kk) = rbuf_l(((kk-(1-nhalo))*ny_ext + (jj-(1-nhalo)))*nhalo + ih)
                    end do
                end do
            end do
        end if

        if (comm_1d_x%east_rank /= MPI_PROC_NULL) then
            !$cuf kernel do(3) <<<*,*>>>
            do kk = 1-nhalo, n3sub+nhalo
                do jj = 1-nhalo, n2sub+nhalo
                    do ih = 1, nhalo
                        u_d(n1sub+ih, jj, kk) = rbuf_r(((kk-(1-nhalo))*ny_ext + (jj-(1-nhalo)))*nhalo + ih)
                    end do
                end do
            end do
        end if

        ierr = cudaDeviceSynchronize()
        deallocate(sbuf_l, sbuf_r, rbuf_l, rbuf_r)
    end subroutine exchange_halo_x

    !---------------------------------------------------------------------------
    ! Halo exchange in y-direction (3D)
    !---------------------------------------------------------------------------
    subroutine exchange_halo_y(u_d, n1sub, n2sub, n3sub)
        use mpi
        use cudafor
        implicit none
        integer, intent(in) :: n1sub, n2sub, n3sub
        real(8), device, intent(inout) :: u_d(1-nhalo:n1sub+nhalo, 1-nhalo:n2sub+nhalo, 1-nhalo:n3sub+nhalo)

        integer :: ierr, reqs(4), stats(MPI_STATUS_SIZE, 4)
        integer :: nsend, nx_ext, nz_ext
        integer :: jh, ii, kk
        real(8), device, allocatable :: sbuf_l(:), sbuf_r(:), rbuf_l(:), rbuf_r(:)

        nx_ext = n1sub + 2*nhalo
        nz_ext = n3sub + 2*nhalo
        nsend = nhalo * nx_ext * nz_ext

        allocate(sbuf_l(nsend), sbuf_r(nsend), rbuf_l(nsend), rbuf_r(nsend))

        !$cuf kernel do(3) <<<*,*>>>
        do kk = 1-nhalo, n3sub+nhalo
            do jh = 1, nhalo
                do ii = 1-nhalo, n1sub+nhalo
                    sbuf_l(((kk-(1-nhalo))*nhalo + (jh-1))*nx_ext + (ii-(1-nhalo)) + 1) = u_d(ii, jh, kk)
                    sbuf_r(((kk-(1-nhalo))*nhalo + (jh-1))*nx_ext + (ii-(1-nhalo)) + 1) = u_d(ii, n2sub-nhalo+jh, kk)
                end do
            end do
        end do
        ierr = cudaDeviceSynchronize()

        call MPI_Isend(sbuf_r, nsend, MPI_DOUBLE_PRECISION, comm_1d_y%east_rank, 0, comm_1d_y%mpi_comm, reqs(1), ierr)
        call MPI_Irecv(rbuf_l, nsend, MPI_DOUBLE_PRECISION, comm_1d_y%west_rank, 0, comm_1d_y%mpi_comm, reqs(2), ierr)
        call MPI_Isend(sbuf_l, nsend, MPI_DOUBLE_PRECISION, comm_1d_y%west_rank, 1, comm_1d_y%mpi_comm, reqs(3), ierr)
        call MPI_Irecv(rbuf_r, nsend, MPI_DOUBLE_PRECISION, comm_1d_y%east_rank, 1, comm_1d_y%mpi_comm, reqs(4), ierr)
        call MPI_Waitall(4, reqs, stats, ierr)

        if (comm_1d_y%west_rank /= MPI_PROC_NULL) then
            !$cuf kernel do(3) <<<*,*>>>
            do kk = 1-nhalo, n3sub+nhalo
                do jh = 1, nhalo
                    do ii = 1-nhalo, n1sub+nhalo
                        u_d(ii, 1-nhalo+jh-1, kk) = rbuf_l(((kk-(1-nhalo))*nhalo + (jh-1))*nx_ext + (ii-(1-nhalo)) + 1)
                    end do
                end do
            end do
        end if

        if (comm_1d_y%east_rank /= MPI_PROC_NULL) then
            !$cuf kernel do(3) <<<*,*>>>
            do kk = 1-nhalo, n3sub+nhalo
                do jh = 1, nhalo
                    do ii = 1-nhalo, n1sub+nhalo
                        u_d(ii, n2sub+jh, kk) = rbuf_r(((kk-(1-nhalo))*nhalo + (jh-1))*nx_ext + (ii-(1-nhalo)) + 1)
                    end do
                end do
            end do
        end if

        ierr = cudaDeviceSynchronize()
        deallocate(sbuf_l, sbuf_r, rbuf_l, rbuf_r)
    end subroutine exchange_halo_y

    !---------------------------------------------------------------------------
    ! Halo exchange in z-direction (3D)
    !---------------------------------------------------------------------------
    subroutine exchange_halo_z(u_d, n1sub, n2sub, n3sub)
        use mpi
        use cudafor
        implicit none
        integer, intent(in) :: n1sub, n2sub, n3sub
        real(8), device, intent(inout) :: u_d(1-nhalo:n1sub+nhalo, 1-nhalo:n2sub+nhalo, 1-nhalo:n3sub+nhalo)

        integer :: ierr, reqs(4), stats(MPI_STATUS_SIZE, 4)
        integer :: nsend, nx_ext, ny_ext
        integer :: kh, jj, ii
        real(8), device, allocatable :: sbuf_l(:), sbuf_r(:), rbuf_l(:), rbuf_r(:)

        nx_ext = n1sub + 2*nhalo
        ny_ext = n2sub + 2*nhalo
        nsend = nhalo * nx_ext * ny_ext

        allocate(sbuf_l(nsend), sbuf_r(nsend), rbuf_l(nsend), rbuf_r(nsend))

        !$cuf kernel do(3) <<<*,*>>>
        do kh = 1, nhalo
            do jj = 1-nhalo, n2sub+nhalo
                do ii = 1-nhalo, n1sub+nhalo
                    sbuf_l(((kh-1)*ny_ext + (jj-(1-nhalo)))*nx_ext + (ii-(1-nhalo)) + 1) = u_d(ii, jj, kh)
                    sbuf_r(((kh-1)*ny_ext + (jj-(1-nhalo)))*nx_ext + (ii-(1-nhalo)) + 1) = u_d(ii, jj, n3sub-nhalo+kh)
                end do
            end do
        end do
        ierr = cudaDeviceSynchronize()

        call MPI_Isend(sbuf_r, nsend, MPI_DOUBLE_PRECISION, comm_1d_z%east_rank, 0, comm_1d_z%mpi_comm, reqs(1), ierr)
        call MPI_Irecv(rbuf_l, nsend, MPI_DOUBLE_PRECISION, comm_1d_z%west_rank, 0, comm_1d_z%mpi_comm, reqs(2), ierr)
        call MPI_Isend(sbuf_l, nsend, MPI_DOUBLE_PRECISION, comm_1d_z%west_rank, 1, comm_1d_z%mpi_comm, reqs(3), ierr)
        call MPI_Irecv(rbuf_r, nsend, MPI_DOUBLE_PRECISION, comm_1d_z%east_rank, 1, comm_1d_z%mpi_comm, reqs(4), ierr)
        call MPI_Waitall(4, reqs, stats, ierr)

        if (comm_1d_z%west_rank /= MPI_PROC_NULL) then
            !$cuf kernel do(3) <<<*,*>>>
            do kh = 1, nhalo
                do jj = 1-nhalo, n2sub+nhalo
                    do ii = 1-nhalo, n1sub+nhalo
                        u_d(ii, jj, 1-nhalo+kh-1) = rbuf_l(((kh-1)*ny_ext + (jj-(1-nhalo)))*nx_ext + (ii-(1-nhalo)) + 1)
                    end do
                end do
            end do
        end if

        if (comm_1d_z%east_rank /= MPI_PROC_NULL) then
            !$cuf kernel do(3) <<<*,*>>>
            do kh = 1, nhalo
                do jj = 1-nhalo, n2sub+nhalo
                    do ii = 1-nhalo, n1sub+nhalo
                        u_d(ii, jj, n3sub+kh) = rbuf_r(((kh-1)*ny_ext + (jj-(1-nhalo)))*nx_ext + (ii-(1-nhalo)) + 1)
                    end do
                end do
            end do
        end if

        ierr = cudaDeviceSynchronize()
        deallocate(sbuf_l, sbuf_r, rbuf_l, rbuf_r)
    end subroutine exchange_halo_z

end module params

!===============================================================================
module kernels
    use cudafor
    use params
    implicit none
contains

    ! Initialize with exact solution
    attributes(global) subroutine init_solution_kernel(u, xc, yc, zc, n1sub, n2sub, n3sub, time)
        implicit none
        integer, value, intent(in) :: n1sub, n2sub, n3sub
        real(8), device, intent(out) :: u(1-nhalo:n1sub+nhalo, 1-nhalo:n2sub+nhalo, 1-nhalo:n3sub+nhalo)
        real(8), device, intent(in) :: xc(1-nhalo:n1sub+nhalo), yc(1-nhalo:n2sub+nhalo), zc(1-nhalo:n3sub+nhalo)
        real(8), value, intent(in) :: time

        integer :: i, j, k
        real(8) :: decay

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x + (1 - nhalo) - 1
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y + (1 - nhalo) - 1
        k = (blockIdx%z - 1) * blockDim%z + threadIdx%z + (1 - nhalo) - 1

        if (i < 1-nhalo .or. i > n1sub+nhalo) return
        if (j < 1-nhalo .or. j > n2sub+nhalo) return
        if (k < 1-nhalo .or. k > n3sub+nhalo) return

        decay = exp(-3.0d0 * alpha_diff * (pi/Lx)**2 * time)
        u(i,j,k) = sin(pi * xc(i) / Lx) * sin(pi * yc(j) / Ly) * sin(pi * zc(k) / Lz) * decay

    end subroutine init_solution_kernel

    ! 4th order 1D stencil: -sigma*u(p-2) + 16*sigma*u(p-1) + (1-30*sigma)*u(p) + 16*sigma*u(p+1) - sigma*u(p+2)
    ! This applies (I + A_dir) to u at a single point along direction dir.

    !---------------------------------------------------------------------------
    ! X-sweep RHS:  b = (I + A_x + 2*A_y + 2*A_z) u^n
    ! sys = (k-1)*n2sub + j,  i = 2*(iblk-1) + m
    !---------------------------------------------------------------------------
    attributes(global) subroutine compute_rhs_xsweep_kernel( &
            u_old, n1sub, n2sub, n3sub, D_rhs, sigma, nblk_x)
        implicit none
        integer, value, intent(in) :: n1sub, n2sub, n3sub, nblk_x
        real(8), device, intent(in) :: u_old(1-nhalo:n1sub+nhalo, 1-nhalo:n2sub+nhalo, 1-nhalo:n3sub+nhalo)
        real(8), device, intent(out) :: D_rhs(1:n2sub*n3sub, 1:nblk_x, 1:mb)
        real(8), value, intent(in) :: sigma
        integer :: sys, iblk, kk, i, j, k
        real(8) :: val_x, val_y, val_z, u0

        sys  = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        iblk = (blockIdx%y - 1) * blockDim%y + threadIdx%y

        if (sys < 1 .or. sys > n2sub*n3sub) return
        if (iblk < 1 .or. iblk > nblk_x) return

        k = (sys - 1) / n2sub + 1
        j = sys - (k - 1) * n2sub

        do kk = 1, mb
            i = 2 * (iblk - 1) + kk
            u0 = u_old(i, j, k)

            ! A_x stencil
            val_x = -sigma*u_old(i-2,j,k) + 16.0d0*sigma*u_old(i-1,j,k) &
                  - 30.0d0*sigma*u0 &
                  + 16.0d0*sigma*u_old(i+1,j,k) - sigma*u_old(i+2,j,k)

            ! A_y stencil (coefficient 2)
            val_y = -sigma*u_old(i,j-2,k) + 16.0d0*sigma*u_old(i,j-1,k) &
                  - 30.0d0*sigma*u0 &
                  + 16.0d0*sigma*u_old(i,j+1,k) - sigma*u_old(i,j+2,k)

            ! A_z stencil (coefficient 2)
            val_z = -sigma*u_old(i,j,k-2) + 16.0d0*sigma*u_old(i,j,k-1) &
                  - 30.0d0*sigma*u0 &
                  + 16.0d0*sigma*u_old(i,j,k+1) - sigma*u_old(i,j,k+2)

            ! (I + A_x + 2*A_y + 2*A_z) u^n
            D_rhs(sys, iblk, kk) = u0 + val_x + 2.0d0*val_y + 2.0d0*val_z
        end do

    end subroutine compute_rhs_xsweep_kernel

    !---------------------------------------------------------------------------
    ! Y-sweep RHS:  d = u* - A_y * u^n
    ! sys = (k-1)*n1sub + i,  j = 2*(jblk-1) + m
    !---------------------------------------------------------------------------
    attributes(global) subroutine compute_rhs_ysweep_kernel( &
            u, u_old, n1sub, n2sub, n3sub, D_rhs, sigma, nblk_y)
        implicit none
        integer, value, intent(in) :: n1sub, n2sub, n3sub, nblk_y
        real(8), device, intent(in) :: u(1-nhalo:n1sub+nhalo, 1-nhalo:n2sub+nhalo, 1-nhalo:n3sub+nhalo)
        real(8), device, intent(in) :: u_old(1-nhalo:n1sub+nhalo, 1-nhalo:n2sub+nhalo, 1-nhalo:n3sub+nhalo)
        real(8), device, intent(out) :: D_rhs(1:n1sub*n3sub, 1:nblk_y, 1:mb)
        real(8), value, intent(in) :: sigma
        integer :: sys, jblk, kk, i, j, k
        real(8) :: val_y

        sys  = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        jblk = (blockIdx%y - 1) * blockDim%y + threadIdx%y

        if (sys < 1 .or. sys > n1sub*n3sub) return
        if (jblk < 1 .or. jblk > nblk_y) return

        k = (sys - 1) / n1sub + 1
        i = sys - (k - 1) * n1sub

        do kk = 1, mb
            j = 2 * (jblk - 1) + kk

            ! A_y applied to u_old
            val_y = -sigma*u_old(i,j-2,k) + 16.0d0*sigma*u_old(i,j-1,k) &
                  - 30.0d0*sigma*u_old(i,j,k) &
                  + 16.0d0*sigma*u_old(i,j+1,k) - sigma*u_old(i,j+2,k)

            D_rhs(sys, jblk, kk) = u(i, j, k) - val_y
        end do

    end subroutine compute_rhs_ysweep_kernel

    !---------------------------------------------------------------------------
    ! Z-sweep RHS:  d = u** - A_z * u^n
    ! sys = (j-1)*n1sub + i,  k = 2*(kblk-1) + m
    !---------------------------------------------------------------------------
    attributes(global) subroutine compute_rhs_zsweep_kernel( &
            u, u_old, n1sub, n2sub, n3sub, D_rhs, sigma, nblk_z)
        implicit none
        integer, value, intent(in) :: n1sub, n2sub, n3sub, nblk_z
        real(8), device, intent(in) :: u(1-nhalo:n1sub+nhalo, 1-nhalo:n2sub+nhalo, 1-nhalo:n3sub+nhalo)
        real(8), device, intent(in) :: u_old(1-nhalo:n1sub+nhalo, 1-nhalo:n2sub+nhalo, 1-nhalo:n3sub+nhalo)
        real(8), device, intent(out) :: D_rhs(1:n1sub*n2sub, 1:nblk_z, 1:mb)
        real(8), value, intent(in) :: sigma
        integer :: sys, kblk, kk, i, j, k
        real(8) :: val_z

        sys  = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        kblk = (blockIdx%y - 1) * blockDim%y + threadIdx%y

        if (sys < 1 .or. sys > n1sub*n2sub) return
        if (kblk < 1 .or. kblk > nblk_z) return

        j = (sys - 1) / n1sub + 1
        i = sys - (j - 1) * n1sub

        do kk = 1, mb
            k = 2 * (kblk - 1) + kk

            ! A_z applied to u_old
            val_z = -sigma*u_old(i,j,k-2) + 16.0d0*sigma*u_old(i,j,k-1) &
                  - 30.0d0*sigma*u_old(i,j,k) &
                  + 16.0d0*sigma*u_old(i,j,k+1) - sigma*u_old(i,j,k+2)

            D_rhs(sys, kblk, kk) = u(i, j, k) - val_z
        end do

    end subroutine compute_rhs_zsweep_kernel

    !---------------------------------------------------------------------------
    ! Build block tridiagonal matrix (same structure for all directions)
    !---------------------------------------------------------------------------
    attributes(global) subroutine build_matrix_kernel( &
            A_mat, B_mat, C_mat, nsys, nblk, sigma, is_first, is_last)
        implicit none
        integer, value, intent(in) :: nsys, nblk
        real(8), device, intent(out) :: A_mat(1:nsys, 1:nblk, 1:mb, 1:mb)
        real(8), device, intent(out) :: B_mat(1:nsys, 1:nblk, 1:mb, 1:mb)
        real(8), device, intent(out) :: C_mat(1:nsys, 1:nblk, 1:mb, 1:mb)
        real(8), value, intent(in) :: sigma
        integer, value, intent(in) :: is_first, is_last

        integer :: s, ib

        s  = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        ib = (blockIdx%y - 1) * blockDim%y + threadIdx%y

        if (s < 1 .or. s > nsys) return
        if (ib < 1 .or. ib > nblk) return

        ! Standard interior block
        A_mat(s, ib, 1, 1) =  sigma
        A_mat(s, ib, 1, 2) = -16.0d0 * sigma
        A_mat(s, ib, 2, 1) =  0.0d0
        A_mat(s, ib, 2, 2) =  sigma

        B_mat(s, ib, 1, 1) =  1.0d0 + 30.0d0 * sigma
        B_mat(s, ib, 1, 2) = -16.0d0 * sigma
        B_mat(s, ib, 2, 1) = -16.0d0 * sigma
        B_mat(s, ib, 2, 2) =  1.0d0 + 30.0d0 * sigma

        C_mat(s, ib, 1, 1) =  sigma
        C_mat(s, ib, 1, 2) =  0.0d0
        C_mat(s, ib, 2, 1) = -16.0d0 * sigma
        C_mat(s, ib, 2, 2) =  sigma

        ! First block globally
        if (is_first == 1 .and. ib == 1) then
            A_mat(s, 1, 1, 1) = 0.0d0
            A_mat(s, 1, 1, 2) = 0.0d0
            A_mat(s, 1, 2, 1) = 0.0d0
            A_mat(s, 1, 2, 2) = 0.0d0

            B_mat(s, 1, 1, 1) =  1.0d0 + 20.0d0 * sigma
            B_mat(s, 1, 1, 2) = -6.0d0 * sigma
            B_mat(s, 1, 2, 1) = -16.0d0 * sigma
            B_mat(s, 1, 2, 2) =  1.0d0 + 30.0d0 * sigma

            C_mat(s, 1, 1, 1) = -4.0d0 * sigma
            C_mat(s, 1, 1, 2) =  sigma
            C_mat(s, 1, 2, 1) = -16.0d0 * sigma
            C_mat(s, 1, 2, 2) =  sigma
        end if

        ! Last block globally
        if (is_last == 1 .and. ib == nblk) then
            A_mat(s, nblk, 1, 1) =  sigma
            A_mat(s, nblk, 1, 2) = -16.0d0 * sigma
            A_mat(s, nblk, 2, 1) =  sigma
            A_mat(s, nblk, 2, 2) = -4.0d0 * sigma

            B_mat(s, nblk, 1, 1) =  1.0d0 + 30.0d0 * sigma
            B_mat(s, nblk, 1, 2) = -16.0d0 * sigma
            B_mat(s, nblk, 2, 1) = -6.0d0 * sigma
            B_mat(s, nblk, 2, 2) =  1.0d0 + 20.0d0 * sigma

            C_mat(s, nblk, 1, 1) = 0.0d0
            C_mat(s, nblk, 1, 2) = 0.0d0
            C_mat(s, nblk, 2, 1) = 0.0d0
            C_mat(s, nblk, 2, 2) = 0.0d0
        end if

    end subroutine build_matrix_kernel

    !---------------------------------------------------------------------------
    ! Unpack kernels
    !---------------------------------------------------------------------------
    attributes(global) subroutine unpack_xsweep_kernel(D_sol, u, n1sub, n2sub, n3sub, nblk_x)
        implicit none
        integer, value, intent(in) :: n1sub, n2sub, n3sub, nblk_x
        real(8), device, intent(in) :: D_sol(1:n2sub*n3sub, 1:nblk_x, 1:mb)
        real(8), device, intent(inout) :: u(1-nhalo:n1sub+nhalo, 1-nhalo:n2sub+nhalo, 1-nhalo:n3sub+nhalo)
        integer :: sys, iblk, kk, i, j, k

        sys  = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        iblk = (blockIdx%y - 1) * blockDim%y + threadIdx%y

        if (sys < 1 .or. sys > n2sub*n3sub) return
        if (iblk < 1 .or. iblk > nblk_x) return

        k = (sys - 1) / n2sub + 1
        j = sys - (k - 1) * n2sub

        do kk = 1, mb
            i = 2 * (iblk - 1) + kk
            u(i, j, k) = D_sol(sys, iblk, kk)
        end do
    end subroutine unpack_xsweep_kernel

    attributes(global) subroutine unpack_ysweep_kernel(D_sol, u, n1sub, n2sub, n3sub, nblk_y)
        implicit none
        integer, value, intent(in) :: n1sub, n2sub, n3sub, nblk_y
        real(8), device, intent(in) :: D_sol(1:n1sub*n3sub, 1:nblk_y, 1:mb)
        real(8), device, intent(inout) :: u(1-nhalo:n1sub+nhalo, 1-nhalo:n2sub+nhalo, 1-nhalo:n3sub+nhalo)
        integer :: sys, jblk, kk, i, j, k

        sys  = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        jblk = (blockIdx%y - 1) * blockDim%y + threadIdx%y

        if (sys < 1 .or. sys > n1sub*n3sub) return
        if (jblk < 1 .or. jblk > nblk_y) return

        k = (sys - 1) / n1sub + 1
        i = sys - (k - 1) * n1sub

        do kk = 1, mb
            j = 2 * (jblk - 1) + kk
            u(i, j, k) = D_sol(sys, jblk, kk)
        end do
    end subroutine unpack_ysweep_kernel

    attributes(global) subroutine unpack_zsweep_kernel(D_sol, u, n1sub, n2sub, n3sub, nblk_z)
        implicit none
        integer, value, intent(in) :: n1sub, n2sub, n3sub, nblk_z
        real(8), device, intent(in) :: D_sol(1:n1sub*n2sub, 1:nblk_z, 1:mb)
        real(8), device, intent(inout) :: u(1-nhalo:n1sub+nhalo, 1-nhalo:n2sub+nhalo, 1-nhalo:n3sub+nhalo)
        integer :: sys, kblk, kk, i, j, k

        sys  = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        kblk = (blockIdx%y - 1) * blockDim%y + threadIdx%y

        if (sys < 1 .or. sys > n1sub*n2sub) return
        if (kblk < 1 .or. kblk > nblk_z) return

        j = (sys - 1) / n1sub + 1
        i = sys - (j - 1) * n1sub

        do kk = 1, mb
            k = 2 * (kblk - 1) + kk
            u(i, j, k) = D_sol(sys, kblk, kk)
        end do
    end subroutine unpack_zsweep_kernel

    !---------------------------------------------------------------------------
    ! Apply Dirichlet BC to halo regions at domain boundaries
    !---------------------------------------------------------------------------
    attributes(global) subroutine apply_bc_kernel(u, xc, yc, zc, n1sub, n2sub, n3sub, &
            is_first_x, is_last_x, is_first_y, is_last_y, is_first_z, is_last_z, time)
        implicit none
        integer, value, intent(in) :: n1sub, n2sub, n3sub
        real(8), device, intent(inout) :: u(1-nhalo:n1sub+nhalo, 1-nhalo:n2sub+nhalo, 1-nhalo:n3sub+nhalo)
        real(8), device, intent(in) :: xc(1-nhalo:n1sub+nhalo), yc(1-nhalo:n2sub+nhalo), zc(1-nhalo:n3sub+nhalo)
        integer, value, intent(in) :: is_first_x, is_last_x, is_first_y, is_last_y, is_first_z, is_last_z
        real(8), value, intent(in) :: time

        integer :: idx, jdx, i, j, k
        real(8) :: decay

        idx = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        jdx = (blockIdx%y - 1) * blockDim%y + threadIdx%y

        decay = exp(-3.0d0 * alpha_diff * (pi/Lx)**2 * time)

        ! x-boundaries
        if (is_first_x == 1) then
            j = idx - nhalo;  k = jdx - nhalo
            if (j >= 1-nhalo .and. j <= n2sub+nhalo .and. &
                k >= 1-nhalo .and. k <= n3sub+nhalo) then
                do i = 1-nhalo, 0
                    u(i, j, k) = sin(pi*xc(i)/Lx) * sin(pi*yc(j)/Ly) * sin(pi*zc(k)/Lz) * decay
                end do
            end if
        end if
        if (is_last_x == 1) then
            j = idx - nhalo;  k = jdx - nhalo
            if (j >= 1-nhalo .and. j <= n2sub+nhalo .and. &
                k >= 1-nhalo .and. k <= n3sub+nhalo) then
                do i = n1sub+1, n1sub+nhalo
                    u(i, j, k) = sin(pi*xc(i)/Lx) * sin(pi*yc(j)/Ly) * sin(pi*zc(k)/Lz) * decay
                end do
            end if
        end if

        ! y-boundaries
        if (is_first_y == 1) then
            i = idx - nhalo;  k = jdx - nhalo
            if (i >= 1-nhalo .and. i <= n1sub+nhalo .and. &
                k >= 1-nhalo .and. k <= n3sub+nhalo) then
                do j = 1-nhalo, 0
                    u(i, j, k) = sin(pi*xc(i)/Lx) * sin(pi*yc(j)/Ly) * sin(pi*zc(k)/Lz) * decay
                end do
            end if
        end if
        if (is_last_y == 1) then
            i = idx - nhalo;  k = jdx - nhalo
            if (i >= 1-nhalo .and. i <= n1sub+nhalo .and. &
                k >= 1-nhalo .and. k <= n3sub+nhalo) then
                do j = n2sub+1, n2sub+nhalo
                    u(i, j, k) = sin(pi*xc(i)/Lx) * sin(pi*yc(j)/Ly) * sin(pi*zc(k)/Lz) * decay
                end do
            end if
        end if

        ! z-boundaries
        if (is_first_z == 1) then
            i = idx - nhalo;  j = jdx - nhalo
            if (i >= 1-nhalo .and. i <= n1sub+nhalo .and. &
                j >= 1-nhalo .and. j <= n2sub+nhalo) then
                do k = 1-nhalo, 0
                    u(i, j, k) = sin(pi*xc(i)/Lx) * sin(pi*yc(j)/Ly) * sin(pi*zc(k)/Lz) * decay
                end do
            end if
        end if
        if (is_last_z == 1) then
            i = idx - nhalo;  j = jdx - nhalo
            if (i >= 1-nhalo .and. i <= n1sub+nhalo .and. &
                j >= 1-nhalo .and. j <= n2sub+nhalo) then
                do k = n3sub+1, n3sub+nhalo
                    u(i, j, k) = sin(pi*xc(i)/Lx) * sin(pi*yc(j)/Ly) * sin(pi*zc(k)/Lz) * decay
                end do
            end if
        end if

    end subroutine apply_bc_kernel

    !---------------------------------------------------------------------------
    ! Compute L2 error vs exact solution
    !---------------------------------------------------------------------------
    attributes(global) subroutine compute_error_kernel(u, xc, yc, zc, n1sub, n2sub, n3sub, time, err_local)
        implicit none
        integer, value, intent(in) :: n1sub, n2sub, n3sub
        real(8), device, intent(in) :: u(1-nhalo:n1sub+nhalo, 1-nhalo:n2sub+nhalo, 1-nhalo:n3sub+nhalo)
        real(8), device, intent(in) :: xc(1-nhalo:n1sub+nhalo), yc(1-nhalo:n2sub+nhalo), zc(1-nhalo:n3sub+nhalo)
        real(8), value, intent(in) :: time
        real(8), device, intent(inout) :: err_local(1)

        integer :: i, j, k
        real(8) :: decay, u_exact, diff, tmp

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = (blockIdx%z - 1) * blockDim%z + threadIdx%z

        if (i < 1 .or. i > n1sub) return
        if (j < 1 .or. j > n2sub) return
        if (k < 1 .or. k > n3sub) return

        decay = exp(-3.0d0 * alpha_diff * (pi/Lx)**2 * time)
        u_exact = sin(pi * xc(i) / Lx) * sin(pi * yc(j) / Ly) * sin(pi * zc(k) / Lz) * decay
        diff = u(i, j, k) - u_exact

        tmp = atomicadd(err_local(1), diff * diff)

    end subroutine compute_error_kernel

end module kernels

!===============================================================================
program heat3d_btdma_cuda
    use cudafor
    use params
    use kernels
    use mpi
    use mod_btdma_gpu_v2
    use mpiutil
    implicit none

    integer :: nprocs, myrank, ierr, istat
    integer :: node_comm, local_rank
    integer :: n1sub, n2sub, n3sub, nblk_x, nblk_y, nblk_z
    integer :: globindx_xa, globindx_xb
    integer :: globindx_ya, globindx_yb
    integer :: globindx_za, globindx_zb
    integer :: i, j, k, dev, ngpu, gpurank
    integer :: is_first_x, is_last_x, is_first_y, is_last_y, is_first_z, is_last_z
    type(cudadeviceprop) :: prop

    integer :: nsys_x, nsys_y, nsys_z
    type(BTDMA_PLAN_gpu_v2) :: plan_x, plan_y, plan_z

    real(8) :: hx, hy, hz, h, dt, sigma, time_current
    real(8) :: err_global, err_local_h
    integer :: istep

    real(8), allocatable :: xsub(:), ysub(:), zsub(:)

    real(8), device, allocatable :: u_d(:,:,:), u_old_d(:,:,:)
    real(8), device, allocatable :: xsub_d(:), ysub_d(:), zsub_d(:)
    real(8), device, allocatable :: Ax_d(:,:,:,:), Bx_d(:,:,:,:), Cx_d(:,:,:,:), Dx_d(:,:,:)
    real(8), device, allocatable :: Ay_d(:,:,:,:), By_d(:,:,:,:), Cy_d(:,:,:,:), Dy_d(:,:,:)
    real(8), device, allocatable :: Az_d(:,:,:,:), Bz_d(:,:,:,:), Cz_d(:,:,:,:), Dz_d(:,:,:)
    real(8), device, allocatable :: err_d(:)

    type(dim3) :: blockSize, gridSize, gridSize_init, gridSize_bc, gridSize_err
    real(8) :: t_total_a, t_total_b
    !--- per-component timing (Euler style, Table 2) -----------------------
    real(8) :: t_copy_a=0.d0,   t_copy_b=0.d0
    real(8) :: t_rhs_a=0.d0,    t_rhs_b=0.d0
    real(8) :: t_matrix_a=0.d0, t_matrix_b=0.d0
    real(8) :: t_solve_a=0.d0,  t_solve_b=0.d0
    real(8) :: t_trans_a=0.d0,  t_trans_b=0.d0      ! unpack kernels (layout re-arrangement)
    real(8) :: t_bc_a=0.d0,     t_bc_b=0.d0
    real(8) :: t_comm_a=0.d0,   t_comm_b=0.d0
    real(8) :: t_total_comp
    real(8) :: timing_local(9), timing_max(9)
    integer :: maxdim

    !--- MPI init ---
    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)

    if (nprocs /= np1*np2*np3) then
        if (myrank == 0) then
            write(*,'(A,I0,A,I0)') 'ERROR: MPI ranks=', nprocs, &
                ' but NP1*NP2*NP3=', np1*np2*np3
        end if
        call MPI_Abort(MPI_COMM_WORLD, 2, ierr)
    end if

    if (mod(nx, np1) /= 0 .or. mod(ny, np2) /= 0 .or. mod(nz, np3) /= 0) then
        if (myrank == 0) write(*,'(A)') 'ERROR: each grid dimension must divide its Cartesian process dimension.'
        call MPI_Abort(MPI_COMM_WORLD, 3, ierr)
    end if

    if (mod(nx/np1, 2) /= 0 .or. mod(ny/np2, 2) /= 0 .or. mod(nz/np3, 2) /= 0) then
        if (myrank == 0) write(*,'(A)') 'ERROR: each local line length must be even for m=2 regrouping.'
        call MPI_Abort(MPI_COMM_WORLD, 4, ierr)
    end if

    ! GPU assignment uses the node-local MPI rank.  With one GPU exposed per
    ! task by Slurm, ngpu is one and every task selects device zero.
    call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, node_comm, ierr)
    call MPI_Comm_rank(node_comm, local_rank, ierr)
    ierr = cudaGetDeviceCount(ngpu)
    if (ngpu <= 0) then
        write(*,'(A,I0,A)') 'ERROR: rank ', myrank, ' sees no CUDA device.'
        call MPI_Abort(MPI_COMM_WORLD, 5, ierr)
    end if
    gpurank = mod(local_rank, ngpu)
    ierr = cudaSetDevice(gpurank)
    ierr = cudaDeviceSynchronize()
    ierr = cudaGetDevice(dev)
    ierr = cudaGetDeviceProperties(prop, dev)
    call MPI_Comm_free(node_comm, ierr)

    ! Topology
    np_dim(0:2) = (/np1, np2, np3/)
    period(0) = .false.;  period(1) = .false.;  period(2) = .false.
    call mpi_topology_make()

    ! Domain decomposition
    n1sub = mpiutil_para(1, nx, comm_1d_x%myrank, comm_1d_x%nprocs, globindx_xa, globindx_xb)
    n2sub = mpiutil_para(1, ny, comm_1d_y%myrank, comm_1d_y%nprocs, globindx_ya, globindx_yb)
    n3sub = mpiutil_para(1, nz, comm_1d_z%myrank, comm_1d_z%nprocs, globindx_za, globindx_zb)
    nblk_x = n1sub / 2
    nblk_y = n2sub / 2
    nblk_z = n3sub / 2

    is_first_x = merge(1, 0, comm_1d_x%myrank == 0)
    is_last_x  = merge(1, 0, comm_1d_x%myrank == comm_1d_x%nprocs - 1)
    is_first_y = merge(1, 0, comm_1d_y%myrank == 0)
    is_last_y  = merge(1, 0, comm_1d_y%myrank == comm_1d_y%nprocs - 1)
    is_first_z = merge(1, 0, comm_1d_z%myrank == 0)
    is_last_z  = merge(1, 0, comm_1d_z%myrank == comm_1d_z%nprocs - 1)

    nsys_x = n2sub * n3sub
    nsys_y = n1sub * n3sub
    nsys_z = n1sub * n2sub

    ! Grid spacing
    hx = Lx / dble(nx + 1)
    hy = Ly / dble(ny + 1)
    hz = Lz / dble(nz + 1)
    h = hx
    dt = 0.5d0 / alpha_diff / dble(nref + 1) / dble(nref + 1)
    sigma = alpha_diff * dt / (24.0d0 * h * h)

    if (myrank == 0) then
        write(*,'(A)')       '================================================='
        write(*,'(A)')       ' 3D Heat Eq - Douglas-Gunn ADI O(h^4) - Multi-GPU'
        write(*,'(A)')       '================================================='
        write(*,'(A,I6,A,I6,A,I6)') ' Grid (internal): ', nx, ' x ', ny, ' x ', nz
        write(*,'(A,I4,A,I4,A,I4)') ' MPI topology:    ', np1, ' x ', np2, ' x ', np3
        write(*,'(A,E12.5)')    ' h  = ', h
        write(*,'(A,E12.5)')    ' dt = ', dt
        write(*,'(A,E12.5)')    ' sigma = ', sigma
        write(*,'(A,I8)')       ' N_ref = ', nref
        write(*,'(A,I8)')       ' nsteps = ', nrun
    end if

    !--- Allocate coordinates ---
    allocate(xsub(1-nhalo:n1sub+nhalo), ysub(1-nhalo:n2sub+nhalo), zsub(1-nhalo:n3sub+nhalo))
    do i = 1-nhalo, n1sub+nhalo;  xsub(i) = dble(globindx_xa + i - 1) * hx;  end do
    do j = 1-nhalo, n2sub+nhalo;  ysub(j) = dble(globindx_ya + j - 1) * hy;  end do
    do k = 1-nhalo, n3sub+nhalo;  zsub(k) = dble(globindx_za + k - 1) * hz;  end do

    allocate(xsub_d(1-nhalo:n1sub+nhalo), ysub_d(1-nhalo:n2sub+nhalo), zsub_d(1-nhalo:n3sub+nhalo))
    xsub_d = xsub;  ysub_d = ysub;  zsub_d = zsub

    !--- Allocate solution ---
    allocate(u_d(1-nhalo:n1sub+nhalo, 1-nhalo:n2sub+nhalo, 1-nhalo:n3sub+nhalo))
    allocate(u_old_d(1-nhalo:n1sub+nhalo, 1-nhalo:n2sub+nhalo, 1-nhalo:n3sub+nhalo))

    !--- Allocate BTDMA arrays ---
    allocate(Ax_d(nsys_x, nblk_x, mb, mb), Bx_d(nsys_x, nblk_x, mb, mb))
    allocate(Cx_d(nsys_x, nblk_x, mb, mb), Dx_d(nsys_x, nblk_x, mb))

    allocate(Ay_d(nsys_y, nblk_y, mb, mb), By_d(nsys_y, nblk_y, mb, mb))
    allocate(Cy_d(nsys_y, nblk_y, mb, mb), Dy_d(nsys_y, nblk_y, mb))

    allocate(Az_d(nsys_z, nblk_z, mb, mb), Bz_d(nsys_z, nblk_z, mb, mb))
    allocate(Cz_d(nsys_z, nblk_z, mb, mb), Dz_d(nsys_z, nblk_z, mb))

    allocate(err_d(1))

    !--- BTDMA plans ---
    call btdma_makeplan_gpu_v2(plan_x, mb, nsys_x, nblk_x, comm_1d_x%mpi_comm)
    call btdma_makeplan_gpu_v2(plan_y, mb, nsys_y, nblk_y, comm_1d_y%mpi_comm)
    call btdma_makeplan_gpu_v2(plan_z, mb, nsys_z, nblk_z, comm_1d_z%mpi_comm)

    !--- Initialize ---
    time_current = 0.0d0

    gridSize_init  = dim3((n1sub+2*nhalo+7)/8, (n2sub+2*nhalo+7)/8, (n3sub+2*nhalo+3)/4)
    blockSize = dim3(8, 8, 4)
    call init_solution_kernel<<<gridSize_init, blockSize>>>( &
        u_d, xsub_d, ysub_d, zsub_d, n1sub, n2sub, n3sub, 0.0d0)
    istat = cudaDeviceSynchronize()

    maxdim = max(n1sub+2*nhalo, n2sub+2*nhalo, n3sub+2*nhalo)
    gridSize_bc = dim3((maxdim+15)/16, (maxdim+15)/16, 1)
    call apply_bc_kernel<<<gridSize_bc, dim3(16,16,1)>>>( &
        u_d, xsub_d, ysub_d, zsub_d, n1sub, n2sub, n3sub, &
        is_first_x, is_last_x, is_first_y, is_last_y, is_first_z, is_last_z, 0.0d0)
    istat = cudaDeviceSynchronize()
    call exchange_halo_x(u_d, n1sub, n2sub, n3sub)
    call exchange_halo_y(u_d, n1sub, n2sub, n3sub)
    call exchange_halo_z(u_d, n1sub, n2sub, n3sub)

    !--- Time stepping ---
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    t_total_a = MPI_WTIME()

    do istep = 1, nrun
        time_current = time_current + dt

        ! Save u^n
        if (istep > 1) t_copy_a = MPI_WTIME()
        u_old_d = u_d
        istat = cudaDeviceSynchronize()
        if (istep > 1) t_copy_b = t_copy_b + (MPI_WTIME() - t_copy_a)

        ! ============ Step 1: x-sweep ============
        blockSize = dim3(64, 1, 1)
        gridSize  = dim3((nsys_x+63)/64, nblk_x, 1)

        if (istep > 1) t_rhs_a = MPI_WTIME()
        call compute_rhs_xsweep_kernel<<<gridSize, blockSize>>>( &
            u_old_d, n1sub, n2sub, n3sub, Dx_d, sigma, nblk_x)
        istat = cudaDeviceSynchronize()
        if (istep > 1) t_rhs_b = t_rhs_b + (MPI_WTIME() - t_rhs_a)

        if (istep > 1) t_matrix_a = MPI_WTIME()
        call build_matrix_kernel<<<gridSize, blockSize>>>( &
            Ax_d, Bx_d, Cx_d, nsys_x, nblk_x, sigma, is_first_x, is_last_x)
        istat = cudaDeviceSynchronize()
        if (istep > 1) t_matrix_b = t_matrix_b + (MPI_WTIME() - t_matrix_a)

        if (istep > 1) t_solve_a = MPI_WTIME()
        call btdma_many_mpi_gpu_v2(Ax_d, Bx_d, Cx_d, Dx_d, mb, nsys_x, nblk_x, plan_x)
        istat = cudaDeviceSynchronize()
        if (istep > 1) t_solve_b = t_solve_b + (MPI_WTIME() - t_solve_a)

        if (istep > 1) t_trans_a = MPI_WTIME()
        call unpack_xsweep_kernel<<<gridSize, blockSize>>>(Dx_d, u_d, n1sub, n2sub, n3sub, nblk_x)
        istat = cudaDeviceSynchronize()
        if (istep > 1) t_trans_b = t_trans_b + (MPI_WTIME() - t_trans_a)

        if (istep > 1) t_bc_a = MPI_WTIME()
        call apply_bc_kernel<<<gridSize_bc, dim3(16,16,1)>>>( &
            u_d, xsub_d, ysub_d, zsub_d, n1sub, n2sub, n3sub, &
            is_first_x, is_last_x, is_first_y, is_last_y, is_first_z, is_last_z, time_current)
        istat = cudaDeviceSynchronize()
        if (istep > 1) t_bc_b = t_bc_b + (MPI_WTIME() - t_bc_a)

        if (istep > 1) t_comm_a = MPI_WTIME()
        call exchange_halo_x(u_d, n1sub, n2sub, n3sub)
        call exchange_halo_y(u_d, n1sub, n2sub, n3sub)
        call exchange_halo_z(u_d, n1sub, n2sub, n3sub)
        if (istep > 1) t_comm_b = t_comm_b + (MPI_WTIME() - t_comm_a)

        ! ============ Step 2: y-sweep ============
        gridSize  = dim3((nsys_y+63)/64, nblk_y, 1)

        if (istep > 1) t_rhs_a = MPI_WTIME()
        call compute_rhs_ysweep_kernel<<<gridSize, blockSize>>>( &
            u_d, u_old_d, n1sub, n2sub, n3sub, Dy_d, sigma, nblk_y)
        istat = cudaDeviceSynchronize()
        if (istep > 1) t_rhs_b = t_rhs_b + (MPI_WTIME() - t_rhs_a)

        if (istep > 1) t_matrix_a = MPI_WTIME()
        call build_matrix_kernel<<<gridSize, blockSize>>>( &
            Ay_d, By_d, Cy_d, nsys_y, nblk_y, sigma, is_first_y, is_last_y)
        istat = cudaDeviceSynchronize()
        if (istep > 1) t_matrix_b = t_matrix_b + (MPI_WTIME() - t_matrix_a)

        if (istep > 1) t_solve_a = MPI_WTIME()
        call btdma_many_mpi_gpu_v2(Ay_d, By_d, Cy_d, Dy_d, mb, nsys_y, nblk_y, plan_y)
        istat = cudaDeviceSynchronize()
        if (istep > 1) t_solve_b = t_solve_b + (MPI_WTIME() - t_solve_a)

        if (istep > 1) t_trans_a = MPI_WTIME()
        call unpack_ysweep_kernel<<<gridSize, blockSize>>>(Dy_d, u_d, n1sub, n2sub, n3sub, nblk_y)
        istat = cudaDeviceSynchronize()
        if (istep > 1) t_trans_b = t_trans_b + (MPI_WTIME() - t_trans_a)

        if (istep > 1) t_bc_a = MPI_WTIME()
        call apply_bc_kernel<<<gridSize_bc, dim3(16,16,1)>>>( &
            u_d, xsub_d, ysub_d, zsub_d, n1sub, n2sub, n3sub, &
            is_first_x, is_last_x, is_first_y, is_last_y, is_first_z, is_last_z, time_current)
        istat = cudaDeviceSynchronize()
        if (istep > 1) t_bc_b = t_bc_b + (MPI_WTIME() - t_bc_a)

        if (istep > 1) t_comm_a = MPI_WTIME()
        call exchange_halo_x(u_d, n1sub, n2sub, n3sub)
        call exchange_halo_y(u_d, n1sub, n2sub, n3sub)
        call exchange_halo_z(u_d, n1sub, n2sub, n3sub)
        if (istep > 1) t_comm_b = t_comm_b + (MPI_WTIME() - t_comm_a)

        ! ============ Step 3: z-sweep ============
        gridSize  = dim3((nsys_z+63)/64, nblk_z, 1)

        if (istep > 1) t_rhs_a = MPI_WTIME()
        call compute_rhs_zsweep_kernel<<<gridSize, blockSize>>>( &
            u_d, u_old_d, n1sub, n2sub, n3sub, Dz_d, sigma, nblk_z)
        istat = cudaDeviceSynchronize()
        if (istep > 1) t_rhs_b = t_rhs_b + (MPI_WTIME() - t_rhs_a)

        if (istep > 1) t_matrix_a = MPI_WTIME()
        call build_matrix_kernel<<<gridSize, blockSize>>>( &
            Az_d, Bz_d, Cz_d, nsys_z, nblk_z, sigma, is_first_z, is_last_z)
        istat = cudaDeviceSynchronize()
        if (istep > 1) t_matrix_b = t_matrix_b + (MPI_WTIME() - t_matrix_a)

        if (istep > 1) t_solve_a = MPI_WTIME()
        call btdma_many_mpi_gpu_v2(Az_d, Bz_d, Cz_d, Dz_d, mb, nsys_z, nblk_z, plan_z)
        istat = cudaDeviceSynchronize()
        if (istep > 1) t_solve_b = t_solve_b + (MPI_WTIME() - t_solve_a)

        if (istep > 1) t_trans_a = MPI_WTIME()
        call unpack_zsweep_kernel<<<gridSize, blockSize>>>(Dz_d, u_d, n1sub, n2sub, n3sub, nblk_z)
        istat = cudaDeviceSynchronize()
        if (istep > 1) t_trans_b = t_trans_b + (MPI_WTIME() - t_trans_a)

        if (istep > 1) t_bc_a = MPI_WTIME()
        call apply_bc_kernel<<<gridSize_bc, dim3(16,16,1)>>>( &
            u_d, xsub_d, ysub_d, zsub_d, n1sub, n2sub, n3sub, &
            is_first_x, is_last_x, is_first_y, is_last_y, is_first_z, is_last_z, time_current)
        istat = cudaDeviceSynchronize()
        if (istep > 1) t_bc_b = t_bc_b + (MPI_WTIME() - t_bc_a)

        if (istep > 1) t_comm_a = MPI_WTIME()
        call exchange_halo_x(u_d, n1sub, n2sub, n3sub)
        call exchange_halo_y(u_d, n1sub, n2sub, n3sub)
        call exchange_halo_z(u_d, n1sub, n2sub, n3sub)
        if (istep > 1) t_comm_b = t_comm_b + (MPI_WTIME() - t_comm_a)

        if (mod(istep, max(nrun/10, 1)) == 0 .and. myrank == 0) then
            write(*,'(A,I6,A,E12.5)') '  Step ', istep, ',  time = ', time_current
        end if
    end do

    t_total_b = MPI_WTIME() - t_total_a
    t_total_comp = t_copy_b + t_rhs_b + t_matrix_b + t_solve_b + t_trans_b + t_bc_b + t_comm_b

    timing_local = (/ t_total_b, t_total_comp, t_copy_b, t_rhs_b, t_matrix_b, &
                      t_solve_b, t_trans_b, t_bc_b, t_comm_b /)
    call MPI_Reduce(timing_local, timing_max, size(timing_local), MPI_DOUBLE_PRECISION, &
                    MPI_MAX, 0, MPI_COMM_WORLD, ierr)

    !--- Compute error ---
    err_d(1) = 0.0d0
    gridSize_err  = dim3((n1sub+7)/8, (n2sub+7)/8, (n3sub+3)/4)
    call compute_error_kernel<<<gridSize_err, dim3(8,8,4)>>>( &
        u_d, xsub_d, ysub_d, zsub_d, n1sub, n2sub, n3sub, time_current, err_d)
    istat = cudaDeviceSynchronize()

    err_local_h = err_d(1)
    call MPI_Allreduce(err_local_h, err_global, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
    err_global = sqrt(err_global / dble(nx * ny * nz))

    if (myrank == 0) then
        write(*,'(A)')           '================================================='
        write(*,'(A,E15.8)')     ' L2 error     = ', err_global
        write(*,'(A,F10.4,A)')   ' Total time   = ', timing_max(1), ' s'
        write(*,'(A,F10.4,A)')   ' BTDMA solve  = ', timing_max(6), ' s'
        write(*,'(A)')           '================================================='

        !--- Per-component wall time (excl. step 1 warmup) -----------------
        write(*,'(A)')           '===== Wall time (s, excl. step 1) ============='
        write(*,'(1A12,3I8)') "nx -ny -nz ", nx, ny, nz
        write(*,'(1A12,3I8)') "np1-np2-np3", np_dim(0), np_dim(1), np_dim(2)
        write(*,'(A,I8)')     "nrun         = ", nrun
        write(*,'(7(A16,A1))')  "t_total_comp","|","t_copy","|","t_rhs","|","t_matrix", &
                               "|","t_solve","|","t_trans","|","t_bc","|","t_comm","|"
        write(*,'(7(E16.8,A1))') timing_max(2),"|", timing_max(3),"|", timing_max(4),"|", timing_max(5), &
                                 "|", timing_max(6),"|", timing_max(7),"|", timing_max(8),"|", timing_max(9),"|"

        !--- Table 2 (memo3 §5.3 format): A_build / RHS_build / Solve / Transpose / etc
        write(*,'(A)')           '===== Table 2 row (s, excl. step 1) =========='
        write(*,'(6(A16,A1))')  "Total","|","A_build","|","RHS_build","|", &
                                 "Solve","|","Transpose","|","etc","|"
        write(*,'(6(E16.8,A1))') timing_max(2),                    "|", &
                                  timing_max(5),                    "|", &
                                 (timing_max(4)+timing_max(3)),     "|", &
                                  timing_max(6),                    "|", &
                                  timing_max(7),                    "|", &
                                 (timing_max(8)+timing_max(9)),     "|"
        write(*,'(A,F10.2,A)') ' BTDMA solve share = ', 1.0d2*timing_max(6)/timing_max(2), ' %'

        write(*,'(A)') 'HEAT3D_RESULT_BEGIN'
        write(*,'(A,I0)') 'nx=', nx
        write(*,'(A,I0)') 'ny=', ny
        write(*,'(A,I0)') 'nz=', nz
        write(*,'(A,I0)') 'np1=', np1
        write(*,'(A,I0)') 'np2=', np2
        write(*,'(A,I0)') 'np3=', np3
        write(*,'(A,I0)') 'mpi_ranks=', nprocs
        write(*,'(A,I0)') 'nsteps=', nrun
        write(*,'(A,I0)') 'nref=', nref
        write(*,'(A,ES24.16)') 'dt=', dt
        write(*,'(A,ES24.16)') 'final_time=', time_current
        write(*,'(A,ES24.16)') 'l2_error=', err_global
        write(*,'(A,ES24.16)') 'total_loop_s=', timing_max(1)
        write(*,'(A,ES24.16)') 'measured_components_s=', timing_max(2)
        write(*,'(A,ES24.16)') 'copy_s=', timing_max(3)
        write(*,'(A,ES24.16)') 'rhs_s=', timing_max(4)
        write(*,'(A,ES24.16)') 'matrix_s=', timing_max(5)
        write(*,'(A,ES24.16)') 'btdma_s=', timing_max(6)
        write(*,'(A,ES24.16)') 'transpose_s=', timing_max(7)
        write(*,'(A,ES24.16)') 'bc_s=', timing_max(8)
        write(*,'(A,ES24.16)') 'halo_s=', timing_max(9)
        write(*,'(A)') 'HEAT3D_RESULT_END'
    end if

    !--- Cleanup ---
    call btdma_cleanplan_gpu_v2(plan_x)
    call btdma_cleanplan_gpu_v2(plan_y)
    call btdma_cleanplan_gpu_v2(plan_z)

    deallocate(u_d, u_old_d, xsub_d, ysub_d, zsub_d)
    deallocate(Ax_d, Bx_d, Cx_d, Dx_d)
    deallocate(Ay_d, By_d, Cy_d, Dy_d)
    deallocate(Az_d, Bz_d, Cz_d, Dz_d)
    deallocate(err_d)
    deallocate(xsub, ysub, zsub)

    call mpi_topology_clean()
    call MPI_FINALIZE(ierr)

end program heat3d_btdma_cuda
