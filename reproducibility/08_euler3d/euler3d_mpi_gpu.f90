module params
    use mpi
    use cudafor
    implicit none
    integer, parameter :: n1 = N1
    integer, parameter :: n2 = N2
    integer, parameter :: n3 = N3
    integer, parameter :: np1 = NP1
    integer, parameter :: np2 = NP2
    integer, parameter :: np3 = NP3
    integer, parameter :: m = 5
    integer, parameter :: nstep_override = NSTEPS
    
    real(8), parameter :: gamma = 1.4d0
    real(8), parameter :: epsilon_diff = 1.0d-8
    real(8), parameter :: pi = 3.141592653589793d0
    
    real(8), parameter :: Lx = 32.0d0
    real(8), parameter :: Ly = 32.0d0
    real(8), parameter :: Lz = 32.0d0
    
    real(8), parameter :: beta_vortex = 5.0d0
    real(8), parameter :: x0_vortex = Lx/2.0d0
    real(8), parameter :: y0_vortex = Ly/2.0d0
    real(8), parameter :: z0_vortex = Lz/2.0d0
    real(8), parameter :: u_inf = 1.0d0
    real(8), parameter :: v_inf = 0.0d0
    real(8), parameter :: w_inf = 0.0d0

    integer, public :: mpi_world_cart       !< Communicator for cartesian topology
    integer, public :: np_dim(0:2)          !< Number of MPI processes in 3D topology
    logical, public :: period(0:2)          !< Periodicity in each direction

    !> @brief   Type variable for the information of 1D communicator
    type, public :: cart_comm_1d
        integer :: myrank                   !< Rank ID in current communicator
        integer :: nprocs                   !< Number of processes in current communicator
        integer :: west_rank                !< Previous rank ID in current communicator
        integer :: east_rank                !< Next rank ID in current communicator
        integer :: mpi_comm                 !< Current communicator

        integer :: subsize(0:2), halosize(0:2)
        integer :: pack_disp_l2r(0:2), unpack_disp_l2r(0:2)
        integer :: pack_disp_r2l(0:2), unpack_disp_r2l(0:2)
    end type cart_comm_1d

    type(cart_comm_1d), public :: comm_1d_x     !< Subcommunicator information in x-direction
    type(cart_comm_1d), public :: comm_1d_y     !< Subcommunicator information in y-direction
    type(cart_comm_1d), public :: comm_1d_z     !< Subcommunicator information in z-direction

    integer, public :: buf_size
    real(8), device, allocatable, public :: buf_send(:), buf_recv(:)
    integer, device, public :: subs(0:2),halos(0:2),disp(0:2)

contains
    subroutine mpi_topology_clean()

        implicit none
        integer :: ierr

        call MPI_Comm_free(mpi_world_cart, ierr)

    end subroutine mpi_topology_clean

    subroutine mpi_topology_make()
        implicit none
        logical :: remain(0:2)
        integer :: ierr

        ! Create the cartesian topology.
        call MPI_Cart_create( MPI_COMM_WORLD    &!  input  | integer      | Input communicator (handle).
                            , 3                 &!  input  | integer      | Number of dimensions of Cartesian grid (integer).
                            , np_dim            &!  input  | processes per Cartesian direction
                            , period            &!  input  | periodicity per Cartesian direction
                            , .false.           &!  input  | preserve MPI rank order
                            , mpi_world_cart    &! *output | integer      | Communicator with new Cartesian topology (handle).
                            , ierr              &!  output | integer      | Fortran only: Error status
                            )

        ! Create subcommunicators and assign two neighboring processes in the x-direction.
        remain(0) = .true.
        remain(1) = .false.
        remain(2) = .false.
        call MPI_Cart_sub( mpi_world_cart, remain, comm_1d_x%mpi_comm, ierr)
        call MPI_Comm_rank(comm_1d_x%mpi_comm, comm_1d_x%myrank, ierr)
        call MPI_Comm_size(comm_1d_x%mpi_comm, comm_1d_x%nprocs, ierr)
        call MPI_Cart_shift(comm_1d_x%mpi_comm, 0, 1, comm_1d_x%west_rank, comm_1d_x%east_rank, ierr)

        ! Create subcommunicators and assign two neighboring processes in the y-direction
        remain(0) = .false.
        remain(1) = .true.
        remain(2) = .false.
        call MPI_Cart_sub( mpi_world_cart, remain, comm_1d_y%mpi_comm, ierr)
        call MPI_Comm_rank(comm_1d_y%mpi_comm, comm_1d_y%myrank, ierr)
        call MPI_Comm_size(comm_1d_y%mpi_comm, comm_1d_y%nprocs, ierr)
        call MPI_Cart_shift(comm_1d_y%mpi_comm, 0, 1, comm_1d_y%west_rank, comm_1d_y%east_rank, ierr)

        ! Create subcommunicators and assign two neighboring processes in the z-direction
        remain(0) = .false.
        remain(1) = .false.
        remain(2) = .true.
        call MPI_Cart_sub( mpi_world_cart, remain, comm_1d_z%mpi_comm, ierr)
        call MPI_Comm_rank(comm_1d_z%mpi_comm, comm_1d_z%myrank, ierr)
        call MPI_Comm_size(comm_1d_z%mpi_comm, comm_1d_z%nprocs, ierr)
        call MPI_Cart_shift(comm_1d_z%mpi_comm, 0, 1, comm_1d_z%west_rank, comm_1d_z%east_rank, ierr)

    end subroutine mpi_topology_make

    subroutine mpi_halocomm_plan(n1sub, n2sub, n3sub, comm_1d_x, comm_1d_y, comm_1d_z)
        use mpi
        implicit none
        integer, intent(in) :: n1sub, n2sub, n3sub
        type(cart_comm_1d) :: comm_1d_x, comm_1d_y, comm_1d_z
        
        integer :: ierr,n12,n23,n13

        n12 = (n1sub+2)*(n2sub+2)*5
        n23 = (n2sub+2)*(n3sub+2)*5
        n13 = (n1sub+2)*(n3sub+2)*5

        buf_size = maxval((/n12, n23, n13/))
        
        allocate(buf_send(buf_size), buf_recv(buf_size))

        comm_1d_x%subsize (0:2) = (/n1sub+2,n2sub+2,n3sub+2/)
        comm_1d_x%halosize(0:2) = (/      1,n2sub+2,n3sub+2/)
        comm_1d_x%pack_disp_l2r  (0:2) = (/n1sub  ,      0,      0/)
        comm_1d_x%unpack_disp_l2r(0:2) = (/      0,      0,      0/)
        comm_1d_x%pack_disp_r2l  (0:2) = (/      1,      0,      0/)
        comm_1d_x%unpack_disp_r2l(0:2) = (/n1sub+1,      0,      0/)

        comm_1d_y%subsize (0:2) = (/n1sub+2,n2sub+2,n3sub+2/)
        comm_1d_y%halosize(0:2) = (/n1sub+2,      1,n3sub+2/)
        comm_1d_y%pack_disp_l2r  (0:2) = (/      0,n2sub  ,      0/)
        comm_1d_y%unpack_disp_l2r(0:2) = (/      0,      0,      0/)
        comm_1d_y%pack_disp_r2l  (0:2) = (/      0,      1,      0/)
        comm_1d_y%unpack_disp_r2l(0:2) = (/      0,n2sub+1,      0/)

        comm_1d_z%subsize (0:2) = (/n1sub+2,n2sub+2,n3sub+2/)
        comm_1d_z%halosize(0:2) = (/n1sub+2,n2sub+2,      1/)
        comm_1d_z%pack_disp_l2r  (0:2) = (/      0,      0,n3sub  /)
        comm_1d_z%unpack_disp_l2r(0:2) = (/      0,      0,      0/)
        comm_1d_z%pack_disp_r2l  (0:2) = (/      0,      0,      1/)
        comm_1d_z%unpack_disp_r2l(0:2) = (/      0,      0,n3sub+1/)

    end subroutine mpi_halocomm_plan
    subroutine mpi_halocomm_clean()
        use mpi
        implicit none
        
        deallocate(buf_send, buf_recv)
    end subroutine mpi_halocomm_clean

    subroutine mpi_halocomm_exchange_halo_3d_gpu(Value, n1sub, n2sub, n3sub, m, comm_1d_x, comm_1d_y, comm_1d_z)
        use mpi
        use cudafor
        implicit none
        integer, intent(in) :: n1sub, n2sub, n3sub, m
        real(8), device, intent(inout) :: Value(0:n1sub+1,0:n2sub+1,0:n3sub+1,m)
        type(cart_comm_1d), intent(in) :: comm_1d_x, comm_1d_y, comm_1d_z

        integer :: ierr, reqs(6), stats(MPI_STATUS_SIZE,6)
        integer :: nsend, nrecv
        integer :: i

        type(dim3) :: blockSize3D, gridSize3D
        integer :: subs0,subs1,subs2
        integer :: halos0,halos1,halos2
        integer :: disp0,disp1,disp2

        ! Exchange in x-direction

        !--x direction
        nsend = comm_1d_x%halosize(0)*comm_1d_x%halosize(1)*comm_1d_x%halosize(2)*m
        nrecv = comm_1d_x%halosize(0)*comm_1d_x%halosize(1)*comm_1d_x%halosize(2)*m
        subs (0:2)= comm_1d_x%subsize (0:2)
        halos(0:2)= comm_1d_x%halosize(0:2)
        blockSize3D = dim3(8, 8, 8)
        gridSize3D  = dim3((comm_1d_x%halosize(0)+7)/8, (comm_1d_x%halosize(1)+7)/8, (comm_1d_x%halosize(2)+7)/8)

        disp(0:2)= comm_1d_x%pack_disp_l2r(0:2)
        call mpi_halocomm_pack_3d_gpu<<<gridSize3D, blockSize3D>>>(Value, m, buf_size, subs, halos, disp, buf_send)
        ierr = cudaDeviceSynchronize()
        call MPI_Isend(buf_send(1), nsend, MPI_DOUBLE_PRECISION, comm_1d_x%east_rank, 0, comm_1d_x%mpi_comm, reqs(1), ierr)
        call MPI_Irecv(buf_recv(1), nrecv, MPI_DOUBLE_PRECISION, comm_1d_x%west_rank, 0, comm_1d_x%mpi_comm, reqs(2), ierr)
        call MPI_Waitall(2, reqs(1:2), stats(:,1:2), ierr)
        disp(0:2)= comm_1d_x%unpack_disp_l2r(0:2)
        call mpi_halocomm_unpack_3d_gpu<<<gridSize3D, blockSize3D>>>(Value, m, buf_size, subs, halos, disp, buf_recv)
        ierr = cudaDeviceSynchronize()
        
        disp(0:2)= comm_1d_x%pack_disp_r2l(0:2)
        call mpi_halocomm_pack_3d_gpu<<<gridSize3D, blockSize3D>>>(Value, m, buf_size, subs, halos, disp, buf_send)
        ierr = cudaDeviceSynchronize()
        call MPI_Isend(buf_send(1), nsend, MPI_DOUBLE_PRECISION, comm_1d_x%west_rank, 0, comm_1d_x%mpi_comm, reqs(1), ierr)
        call MPI_Irecv(buf_recv(1), nrecv, MPI_DOUBLE_PRECISION, comm_1d_x%east_rank, 0, comm_1d_x%mpi_comm, reqs(2), ierr)
        call MPI_Waitall(2, reqs(1:2), stats(:,1:2), ierr)
        disp(0:2)= comm_1d_x%unpack_disp_r2l(0:2)
        call mpi_halocomm_unpack_3d_gpu<<<gridSize3D, blockSize3D>>>(Value, m, buf_size, subs, halos, disp, buf_recv)
        ierr = cudaDeviceSynchronize()

        !--y direction
        nsend = comm_1d_y%halosize(0)*comm_1d_y%halosize(1)*comm_1d_y%halosize(2)*m
        nrecv = comm_1d_y%halosize(0)*comm_1d_y%halosize(1)*comm_1d_y%halosize(2)*m
        subs (0:2)= comm_1d_y%subsize (0:2)
        halos(0:2)= comm_1d_y%halosize(0:2)
        blockSize3D = dim3(8, 8, 8)
        gridSize3D  = dim3((comm_1d_y%halosize(0)+7)/8, (comm_1d_y%halosize(1)+7)/8, (comm_1d_y%halosize(2)+7)/8)
        disp(0:2)= comm_1d_y%pack_disp_l2r(0:2)
        call mpi_halocomm_pack_3d_gpu<<<gridSize3D, blockSize3D>>>(Value, m, buf_size, subs, halos, disp, buf_send)
        ierr = cudaDeviceSynchronize()
        call MPI_Isend(buf_send(1), nsend, MPI_DOUBLE_PRECISION, comm_1d_y%east_rank, 0, comm_1d_y%mpi_comm, reqs(1), ierr)
        call MPI_Irecv(buf_recv(1), nrecv, MPI_DOUBLE_PRECISION, comm_1d_y%west_rank, 0, comm_1d_y%mpi_comm, reqs(2), ierr)
        call MPI_Waitall(2, reqs(1:2), stats(:,1:2), ierr)
        disp(0:2)= comm_1d_y%unpack_disp_l2r(0:2)
        call mpi_halocomm_unpack_3d_gpu<<<gridSize3D, blockSize3D>>>(Value, m, buf_size, subs, halos, disp, buf_recv)
        ierr = cudaDeviceSynchronize()
        
        disp(0:2)= comm_1d_y%pack_disp_r2l(0:2)
        call mpi_halocomm_pack_3d_gpu<<<gridSize3D, blockSize3D>>>(Value, m, buf_size, subs, halos, disp, buf_send)
        ierr = cudaDeviceSynchronize()
        call MPI_Isend(buf_send(1), nsend, MPI_DOUBLE_PRECISION, comm_1d_y%west_rank, 0, comm_1d_y%mpi_comm, reqs(1), ierr)
        call MPI_Irecv(buf_recv(1), nrecv, MPI_DOUBLE_PRECISION, comm_1d_y%east_rank, 0, comm_1d_y%mpi_comm, reqs(2), ierr)
        call MPI_Waitall(2, reqs(1:2), stats(:,1:2), ierr)
        disp(0:2)= comm_1d_y%unpack_disp_r2l(0:2)
        call mpi_halocomm_unpack_3d_gpu<<<gridSize3D, blockSize3D>>>(Value, m, buf_size, subs, halos, disp, buf_recv)
        ierr = cudaDeviceSynchronize()

        !--z direction
        nsend = comm_1d_z%halosize(0)*comm_1d_z%halosize(1)*comm_1d_z%halosize(2)*m
        nrecv = comm_1d_z%halosize(0)*comm_1d_z%halosize(1)*comm_1d_z%halosize(2)*m
        subs (0:2)= comm_1d_z%subsize (0:2)
        halos(0:2)= comm_1d_z%halosize(0:2)
        blockSize3D = dim3(8, 8, 8)
        gridSize3D  = dim3((comm_1d_z%halosize(0)+7)/8, (comm_1d_z%halosize(1)+7)/8, (comm_1d_z%halosize(2)+7)/8)
        disp(0:2)= comm_1d_z%pack_disp_l2r(0:2)
        call mpi_halocomm_pack_3d_gpu<<<gridSize3D, blockSize3D>>>(Value, m, buf_size, subs, halos, disp, buf_send)
        ierr = cudaDeviceSynchronize()
        call MPI_Isend(buf_send(1), nsend, MPI_DOUBLE_PRECISION, comm_1d_z%east_rank, 0, comm_1d_z%mpi_comm, reqs(1), ierr)
        call MPI_Irecv(buf_recv(1), nrecv, MPI_DOUBLE_PRECISION, comm_1d_z%west_rank, 0, comm_1d_z%mpi_comm, reqs(2), ierr)
        call MPI_Waitall(2, reqs(1:2), stats(:,1:2), ierr)
        disp(0:2)= comm_1d_z%unpack_disp_l2r(0:2)
        call mpi_halocomm_unpack_3d_gpu<<<gridSize3D, blockSize3D>>>(Value, m, buf_size, subs, halos, disp, buf_recv)
        ierr = cudaDeviceSynchronize()
        
        disp(0:2)= comm_1d_z%pack_disp_r2l(0:2)
        call mpi_halocomm_pack_3d_gpu<<<gridSize3D, blockSize3D>>>(Value, m, buf_size, subs, halos, disp, buf_send)
        ierr = cudaDeviceSynchronize()
        call MPI_Isend(buf_send(1), nsend, MPI_DOUBLE_PRECISION, comm_1d_z%west_rank, 0, comm_1d_z%mpi_comm, reqs(1), ierr)
        call MPI_Irecv(buf_recv(1), nrecv, MPI_DOUBLE_PRECISION, comm_1d_z%east_rank, 0, comm_1d_z%mpi_comm, reqs(2), ierr)
        call MPI_Waitall(2, reqs(1:2), stats(:,1:2), ierr)
        disp(0:2)= comm_1d_z%unpack_disp_r2l(0:2)
        call mpi_halocomm_unpack_3d_gpu<<<gridSize3D, blockSize3D>>>(Value, m, buf_size, subs, halos, disp, buf_recv)
        ierr = cudaDeviceSynchronize()
    end subroutine mpi_halocomm_exchange_halo_3d_gpu

    attributes(global) subroutine mpi_halocomm_pack_3d_gpu(Value, m, buf_size, subs, halos, disp, buf_send)
        use cudafor
        implicit none
        integer, value :: m, buf_size
        integer, device :: subs(0:2), halos(0:2), disp(0:2)
        real(8), device :: Value(0:subs(0)-1,0:subs(1)-1,0:subs(2)-1,1:m)
        real(8), device :: buf_send(0:buf_size-1)

        integer :: i,j,k,mm
        integer :: idx_send, idx_value
        integer :: i_halo, j_halo, k_halo

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x-1
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y-1
        k = (blockIdx%z - 1) * blockDim%z + threadIdx%z-1

        if (i > halos(0)-1 .or. j > halos(1)-1 .or. k > halos(2)-1) return

        do mm = 1, m
            i_halo = i + disp(0)
            j_halo = j + disp(1)
            k_halo = k + disp(2)

            idx_send = (i) + (j)*halos(0) + (k)*halos(0)*halos(1) + (mm-1)*halos(0)*halos(1)*halos(2)
            buf_send(idx_send) = Value(i_halo,j_halo,k_halo,mm)
        end do

    end subroutine mpi_halocomm_pack_3d_gpu

    attributes(global) subroutine mpi_halocomm_unpack_3d_gpu(Value, m, buf_size, subs, halos, disp, buf_recv)
        use cudafor
        implicit none
        integer, value :: m, buf_size
        integer, device :: subs(0:2), halos(0:2), disp(0:2)
        real(8), device :: Value(0:subs(0)-1,0:subs(1)-1,0:subs(2)-1,1:m)
        real(8), device :: buf_recv(0:buf_size-1)

        integer :: i,j,k,mm
        integer :: idx_send, idx_value
        integer :: i_halo, j_halo, k_halo

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x-1
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y-1
        k = (blockIdx%z - 1) * blockDim%z + threadIdx%z-1

        if (i > halos(0)-1 .or. j > halos(1)-1 .or. k > halos(2)-1) return

        do mm = 1, m
            i_halo = i + disp(0)
            j_halo = j + disp(1)
            k_halo = k + disp(2)

            idx_send = (i) + (j)*halos(0) + (k)*halos(0)*halos(1) + (mm-1)*halos(0)*halos(1)*halos(2)

            Value(i_halo,j_halo,k_halo,mm) = buf_recv(idx_send)
        end do

    end subroutine mpi_halocomm_unpack_3d_gpu

    
end module params

module kernels
    use cudafor
    use params
    implicit none
    
contains

    !======================================================================
    ! Benchmark 1: Isentropic Vortex Convection
    ! 2D isentropic vortex on uniform freestream, exact solution =
    ! initial condition translated by (u_inf*t, v_inf*t).
    ! All variables (rho, u, v, p) have non-trivial spatial distribution.
    !======================================================================
    attributes(global) subroutine initialize_vortex_kernel0(U, x, y, z, n1sub, n2sub, n3sub, m, time)
        integer, value, intent(in) :: n1sub, n2sub, n3sub, m
        real(8), device, intent(out) :: U(0:n1sub+1, 0:n2sub+1, 0:n3sub+1, m)
        real(8), device, intent(in) :: x(0:n1sub+1), y(0:n2sub+1), z(0:n3sub+1)
        real(8), value, intent(in) :: time

        ! Vortex parameters
        real(8), parameter :: beta_v = 5.0d0      ! vortex strength
        real(8), parameter :: R_c = 2.0d0         ! vortex core radius for scaling

        real(8) :: xc, yc
        real(8) :: dx_min, dy_min
        real(8) :: r2, exp_factor
        real(8) :: dT, rho, p, E
        real(8) :: du, dv
        real(8) :: uu, vv, ww
        integer :: i, j, k

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = (blockIdx%z - 1) * blockDim%z + threadIdx%z

        if (i > n1sub .or. j > n2sub .or. k > n3sub) return

        ! Vortex center moves with free-stream velocity (x-y plane only)
        xc = modulo(x0_vortex + u_inf * time, Lx)
        yc = modulo(y0_vortex + v_inf * time, Ly)

        ! Minimum distance considering periodic BC (x-y plane only)
        dx_min = x(i) - xc
        if (abs(dx_min + Lx) < abs(dx_min)) dx_min = dx_min + Lx
        if (abs(dx_min - Lx) < abs(dx_min)) dx_min = dx_min - Lx

        dy_min = y(j) - yc
        if (abs(dy_min + Ly) < abs(dy_min)) dy_min = dy_min + Ly
        if (abs(dy_min - Ly) < abs(dy_min)) dy_min = dy_min - Ly

        ! Normalized radial distance (2D, x-y plane)
        r2 = (dx_min / R_c)**2 + (dy_min / R_c)**2

        !----------------------------------------------------------------------
        ! Temperature perturbation (isentropic relation)
        ! dT = -(gamma-1)*beta^2/(8*gamma*pi^2) * exp(1-r^2)
        !----------------------------------------------------------------------
        dT = -(gamma - 1.0d0) * beta_v**2 / (8.0d0 * gamma * pi**2) * exp(1.0d0 - r2)

        !----------------------------------------------------------------------
        ! Velocity perturbation (2D rotation in x-y plane, uniform in z)
        ! (du, dv) = beta/(2*pi) * exp((1-r^2)/2) * (-dy/R_c, +dx/R_c)
        !----------------------------------------------------------------------
        exp_factor = exp((1.0d0 - r2) / 2.0d0)

        du = -beta_v / (2.0d0 * pi) * (dy_min / R_c) * exp_factor
        dv =  beta_v / (2.0d0 * pi) * (dx_min / R_c) * exp_factor

        ! Primitive variables
        rho = (1.0d0 + dT)**(1.0d0 / (gamma - 1.0d0))
        uu = u_inf + du
        vv = v_inf + dv
        ww = w_inf              ! no perturbation in z
        p = rho**gamma

        ! Total energy per unit mass
        E = p / ((gamma - 1.0d0) * rho) + 0.5d0 * (uu**2 + vv**2 + ww**2)

        ! Conservative variables
        U(i, j, k, 1) = rho
        U(i, j, k, 2) = rho * uu
        U(i, j, k, 3) = rho * vv
        U(i, j, k, 4) = rho * ww
        U(i, j, k, 5) = rho * E

    end subroutine initialize_vortex_kernel0

    !======================================================================
    ! Benchmark 1: Isentropic Vortex Convection with axis = (1,1,1)/sqrt(3)
    ! Same physics as kernel0 but vortex axis rotated to (1,1,1) direction.
    ! The vortex structure lives in the plane perpendicular to (1,1,1).
    !======================================================================
    attributes(global) subroutine initialize_vortex_kernel1(U, x, y, z, n1sub, n2sub, n3sub, m, time)
        integer, value, intent(in) :: n1sub, n2sub, n3sub, m
        real(8), device, intent(out) :: U(0:n1sub+1, 0:n2sub+1, 0:n3sub+1, m)
        real(8), device, intent(in) :: x(0:n1sub+1), y(0:n2sub+1), z(0:n3sub+1)
        real(8), value, intent(in) :: time

        ! Vortex parameters
        real(8), parameter :: beta_v = 5.0d0      ! vortex strength
        real(8), parameter :: rho_inf = 1.0d0     ! free-stream density
        real(8), parameter :: p_inf = 1.0d0       ! free-stream pressure
        real(8), parameter :: T_inf = 1.0d0       ! free-stream temperature (p = rho*T)

        ! Axis direction: (1,1,1)/sqrt(3)
        real(8), parameter :: inv_sqrt3 = 1.0d0 / sqrt(3.0d0)
        real(8), parameter :: ax = inv_sqrt3
        real(8), parameter :: ay = inv_sqrt3
        real(8), parameter :: az = inv_sqrt3

        real(8) :: xc, yc, zc
        real(8) :: dx_min, dy_min, dz_min
        real(8) :: d_dot_n, d_perp_x, d_perp_y, d_perp_z
        real(8) :: r2, r, exp_factor
        real(8) :: dT, T_local, rho, p, E
        real(8) :: tang_x, tang_y, tang_z, u_pol
        real(8) :: uu, vv, ww
        integer :: i, j, k

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = (blockIdx%z - 1) * blockDim%z + threadIdx%z

        if (i > n1sub .or. j > n2sub .or. k > n3sub) return

        ! Vortex center moves with free-stream velocity
        xc = modulo(x0_vortex + u_inf * time, Lx)
        yc = modulo(y0_vortex + v_inf * time, Ly)
        zc = modulo(z0_vortex + w_inf * time, Lz)

        ! Minimum distance considering periodic BC
        dx_min = x(i) - xc
        if (abs(dx_min + Lx) < abs(dx_min)) dx_min = dx_min + Lx
        if (abs(dx_min - Lx) < abs(dx_min)) dx_min = dx_min - Lx

        dy_min = y(j) - yc
        if (abs(dy_min + Ly) < abs(dy_min)) dy_min = dy_min + Ly
        if (abs(dy_min - Ly) < abs(dy_min)) dy_min = dy_min - Ly

        dz_min = z(k) - zc
        if (abs(dz_min + Lz) < abs(dz_min)) dz_min = dz_min + Lz
        if (abs(dz_min - Lz) < abs(dz_min)) dz_min = dz_min - Lz

        !----------------------------------------------------------------------
        ! Decompose displacement into axial and perpendicular components
        ! d_axial = (d . n_hat) * n_hat
        ! d_perp  = d - d_axial
        !----------------------------------------------------------------------
        d_dot_n = dx_min * ax + dy_min * ay + dz_min * az

        d_perp_x = dx_min - d_dot_n * ax
        d_perp_y = dy_min - d_dot_n * ay
        d_perp_z = dz_min - d_dot_n * az

        r2 = d_perp_x**2 + d_perp_y**2 + d_perp_z**2

        ! Exponential factor
        exp_factor = exp((1.0d0 - r2) / 2.0d0)

        !----------------------------------------------------------------------
        ! Velocity perturbation direction: tangent = n_hat x d_perp
        ! This gives the tangential (swirl) direction in the plane perp to axis.
        !----------------------------------------------------------------------
        tang_x = ay * d_perp_z - az * d_perp_y
        tang_y = az * d_perp_x - ax * d_perp_z
        tang_z = ax * d_perp_y - ay * d_perp_x

        ! Poloidal velocity magnitude: beta/(2*pi) * exp((1-r^2)/2)
        u_pol = beta_v / (2.0d0 * pi) * exp_factor

        ! Total velocity = free-stream + perturbation
        uu = u_inf + u_pol * tang_x
        vv = v_inf + u_pol * tang_y
        ww = w_inf + u_pol * tang_z

        !----------------------------------------------------------------------
        ! Temperature perturbation (isentropic relation)
        ! T = T_inf - (gamma-1)*beta^2/(8*gamma*pi^2) * exp(1-r^2)
        !----------------------------------------------------------------------
        dT = -(gamma - 1.0d0) * beta_v**2 / (8.0d0 * gamma * pi**2) * exp(1.0d0 - r2)
        T_local = T_inf + dT

        !----------------------------------------------------------------------
        ! Density and pressure from isentropic relations
        ! rho = T^(1/(gamma-1))  [since rho_inf = T_inf = 1]
        ! p = rho^gamma = rho * T
        !----------------------------------------------------------------------
        rho = T_local**(1.0d0 / (gamma - 1.0d0))
        p = rho**gamma

        !----------------------------------------------------------------------
        ! Total energy per unit mass
        ! E = p/((gamma-1)*rho) + 0.5*(u^2 + v^2 + w^2)
        !----------------------------------------------------------------------
        E = p / ((gamma - 1.0d0) * rho) + 0.5d0 * (uu**2 + vv**2 + ww**2)

        ! Conservative variables
        U(i, j, k, 1) = rho
        U(i, j, k, 2) = rho * uu
        U(i, j, k, 3) = rho * vv
        U(i, j, k, 4) = rho * ww
        U(i, j, k, 5) = rho * E

    end subroutine initialize_vortex_kernel1

    attributes(global) subroutine initialize_vortex_kernel2(U, x, y, z, n1sub,n2sub,n3sub, m, time)
        integer, value, intent(in) :: n1sub, n2sub, n3sub, m
        real(8), device, intent(out) :: U(0:n1sub+1, 0:n2sub+1, 0:n3sub+1, m)
        real(8), device, intent(in) :: x(0:n1sub+1), y(0:n2sub+1), z(0:n3sub+1)
        real(8), value, intent(in) :: time
        
        ! Parameters
        real(8), parameter :: rho0 = 1.0d0
        real(8), parameter :: p0 = 1.0d0
        real(8), parameter :: R_ring = 2.0d0     ! ring major radius
        real(8), parameter :: a_core = 0.5d0     ! vortex core radius
        real(8), parameter :: Gamma_circ = 2.0d0 ! circulation
        
        real(8) :: xc, yc, zc
        real(8) :: dx_min, dy_min, dz_min
        real(8) :: r_cyl, s
        real(8) :: u_poloidal, u_x_contrib, u_r_contrib
        real(8) :: rho, uu, vv, ww, p, E
        real(8) :: cos_phi, sin_phi  ! azimuthal angle in y-z plane
        real(8) :: nx, nr            ! unit vector in poloidal direction
        integer :: i, j, k
        
        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = (blockIdx%z - 1) * blockDim%z + threadIdx%z
        
        if (i > n1sub .or. j > n2sub .or. k > n3sub) return
        
        ! Ring center
        xc = modulo(x0_vortex + u_inf * time, Lx)
        yc = modulo(y0_vortex + v_inf * time, Ly)
        zc = modulo(z0_vortex + w_inf * time, Lz)
        
        ! Distance from ring center
        dx_min = x(i) - xc
        if (abs(dx_min + Lx) < abs(dx_min)) dx_min = dx_min + Lx
        if (abs(dx_min - Lx) < abs(dx_min)) dx_min = dx_min - Lx
        
        dy_min = y(j) - yc
        if (abs(dy_min + Ly) < abs(dy_min)) dy_min = dy_min + Ly
        if (abs(dy_min - Ly) < abs(dy_min)) dy_min = dy_min - Ly
        
        dz_min = z(k) - zc
        if (abs(dz_min + Lz) < abs(dz_min)) dz_min = dz_min + Lz
        if (abs(dz_min - Lz) < abs(dz_min)) dz_min = dz_min - Lz
        
        ! Cylindrical radius (distance from x-axis)
        r_cyl = sqrt(dy_min**2 + dz_min**2)
        
        ! Distance from vortex core center
        ! Core center is at (0, R_ring * cos(phi), R_ring * sin(phi)) 
        ! in local coordinates, where phi is azimuthal angle
        s = sqrt(dx_min**2 + (r_cyl - R_ring)**2)
        
        !----------------------------------------------------------------------
        ! Poloidal velocity magnitude (Lamb-Oseen vortex profile)
        ! u_theta = Gamma/(2*pi*s) * (1 - exp(-(s/a)^2))
        !----------------------------------------------------------------------
        if (s > 1.0d-10) then
            u_poloidal = Gamma_circ / (2.0d0 * pi * s) * (1.0d0 - exp(-(s / a_core)**2))
        else
            ! At core center: solid body rotation limit
            ! u_theta = Gamma/(2*pi*a^2) * s  as s->0
            u_poloidal = Gamma_circ / (2.0d0 * pi * a_core**2) * s
        endif
        
        !----------------------------------------------------------------------
        ! Poloidal direction (tangent to circle around vortex core)
        ! 
        ! In the (x, r-R) plane centered at core:
        !   - Poloidal velocity is perpendicular to radial direction from core
        !   - For POSITIVE circulation (counterclockwise when viewed from 
        !     outside the torus), the direction is:
        !     
        !     At r > R (outside): velocity points in -x direction
        !     At r < R (inside):  velocity points in +x direction
        !     At x > 0 (downstream): velocity points outward (increasing r)
        !     At x < 0 (upstream):   velocity points inward (decreasing r)
        !
        ! Unit vector tangent to poloidal circle (counterclockwise):
        !   e_poloidal = (-sin(theta_p), cos(theta_p)) in (x, r-R) coordinates
        !   where theta_p is measured from the outer equator (+r direction)
        !
        ! cos(theta_p) = (r - R) / s
        ! sin(theta_p) = x / s
        !----------------------------------------------------------------------
        
        if (s > 1.0d-10) then
            ! Unit vector components in (x, r) coordinates
            ! Tangent to poloidal circle, counterclockwise direction
            nx = -(r_cyl - R_ring) / s   ! x-component of tangent
            nr = dx_min / s               ! r-component of tangent
        else
            nx = 0.0d0
            nr = 0.0d0
        endif
        
        ! Velocity contributions
        u_x_contrib = u_poloidal * nx    ! x-direction velocity
        u_r_contrib = u_poloidal * nr    ! radial velocity (in y-z plane)
        
        ! Total velocity
        uu = u_inf + u_x_contrib
        
        if (r_cyl > 1.0d-10) then
            ! Project radial velocity to y and z
            cos_phi = dy_min / r_cyl
            sin_phi = dz_min / r_cyl
            vv = v_inf + u_r_contrib * cos_phi
            ww = w_inf + u_r_contrib * sin_phi
        else
            ! On the x-axis: no y,z velocity from ring
            vv = v_inf
            ww = w_inf
        endif
        
        !----------------------------------------------------------------------
        ! Pressure (with centrifugal correction in core)
        !----------------------------------------------------------------------
        if (s < 4.0d0 * a_core) then
            p = p0 - 0.5d0 * rho0 * (Gamma_circ / (2.0d0 * pi * a_core))**2 &
                * exp(-2.0d0 * (s / a_core)**2)
        else
            p = p0
        endif
        if (p < 0.1d0 * p0) p = 0.1d0 * p0
        
        ! Density
        rho = rho0
        
        ! Total energy
        E = p / ((gamma - 1.0d0) * rho) + 0.5d0 * (uu**2 + vv**2 + ww**2)
        
        ! Conservative variables
        U(i, j, k, 1) = rho
        U(i, j, k, 2) = rho * uu
        U(i, j, k, 3) = rho * vv
        U(i, j, k, 4) = rho * ww
        U(i, j, k, 5) = rho * E
        
    end subroutine initialize_vortex_kernel2

    attributes(global) subroutine compute_fluxes_kernel(U, n1sub, n2sub, n3sub, m, F, G, H)
        integer, value, intent(in) :: n1sub, n2sub, n3sub, m
        real(8), device, intent(in) :: U(0:n1sub+1, 0:n2sub+1, 0:n3sub+1, m)
        real(8), device, intent(out) :: F(0:n1sub+1, 0:n2sub+1, 0:n3sub+1, m)
        real(8), device, intent(out) :: G(0:n1sub+1, 0:n2sub+1, 0:n3sub+1, m)
        real(8), device, intent(out) :: H(0:n1sub+1, 0:n2sub+1, 0:n3sub+1, m)

        real(8) :: rho, rhou, rhov, rhow, rhoE, uu, vv, ww, p
        integer :: i, j, k
        
        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = (blockIdx%z - 1) * blockDim%z + threadIdx%z
        
        if (i > n1sub .or. j > n2sub .or. k > n3sub) return
        
        rho = U(i,j,k,1); rhou = U(i,j,k,2); rhov = U(i,j,k,3); rhow = U(i,j,k,4); rhoE = U(i,j,k,5)
        uu = rhou/rho; vv = rhov/rho; ww = rhow/rho
        p = (gamma - 1.0d0) * (rhoE - 0.5d0 * rho * (uu**2 + vv**2 + ww**2))
        
        F(i,j,k,1) = rhou; F(i,j,k,2) = rhou*uu + p; F(i,j,k,3) = rhou*vv
        F(i,j,k,4) = rhou*ww; F(i,j,k,5) = (rhoE + p)*uu
        
        G(i,j,k,1) = rhov; G(i,j,k,2) = rhov*uu; G(i,j,k,3) = rhov*vv + p
        G(i,j,k,4) = rhov*ww; G(i,j,k,5) = (rhoE + p)*vv
        
        H(i,j,k,1) = rhow; H(i,j,k,2) = rhow*uu; H(i,j,k,3) = rhow*vv
        H(i,j,k,4) = rhow*ww + p; H(i,j,k,5) = (rhoE + p)*ww
    end subroutine compute_fluxes_kernel

    attributes(global) subroutine compute_RHS_kernel(U, n1sub, n2sub, n3sub, m, F, G, H, RHS, dx, dy, dz, dt, eps)
        integer, value, intent(in) :: n1sub, n2sub, n3sub, m
        real(8), device, intent(in) :: U(0:n1sub+1,0:n2sub+1,0:n3sub+1,m)
        real(8), device, intent(in) :: F(0:n1sub+1,0:n2sub+1,0:n3sub+1,m)
        real(8), device, intent(in) :: G(0:n1sub+1,0:n2sub+1,0:n3sub+1,m)
        real(8), device, intent(in) :: H(0:n1sub+1,0:n2sub+1,0:n3sub+1,m)
        real(8), device, intent(out) :: RHS(n1sub, n2sub, n3sub, m)
        real(8), value, intent(in) :: dx, dy, dz, dt, eps
        
        real(8) :: dFdx, dGdy, dHdz, Lap_U
        integer :: i, j, k, iv, im, ip, jm, jp, km, kp
        
        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = (blockIdx%z - 1) * blockDim%z + threadIdx%z
        
        if (i > n1sub .or. j > n2sub .or. k > n3sub) return
        
        im = i - 1
        ip = i + 1
        jm = j - 1
        jp = j + 1
        km = k - 1
        kp = k + 1
        
        do iv = 1, m
            dFdx = (F(ip,j,k,iv) - F(im,j,k,iv)) / (2.0d0 * dx)
            dGdy = (G(i,jp,k,iv) - G(i,jm,k,iv)) / (2.0d0 * dy)
            dHdz = (H(i,j,kp,iv) - H(i,j,km,iv)) / (2.0d0 * dz)
            Lap_U = (U(ip,j,k,iv) - 2.0d0*U(i,j,k,iv) + U(im,j,k,iv)) / dx**2 &
                  + (U(i,jp,k,iv) - 2.0d0*U(i,j,k,iv) + U(i,jm,k,iv)) / dy**2 &
                  + (U(i,j,kp,iv) - 2.0d0*U(i,j,k,iv) + U(i,j,km,iv)) / dz**2
            RHS(i,j,k,iv) = -dt * (dFdx + dGdy + dHdz - eps * Lap_U)
        end do
    end subroutine compute_RHS_kernel

    attributes(global) subroutine copy_array_kernel(src, dst, n1sub, n2sub, n3sub, m)
        integer, value, intent(in) :: n1sub, n2sub, n3sub, m
        real(8), device, intent(in) :: src(n1sub, n2sub, n3sub, m)
        real(8), device, intent(out) :: dst(n1sub, n2sub, n3sub, m)
        integer :: i, j, k, iv
        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = (blockIdx%z - 1) * blockDim%z + threadIdx%z
        if (i > n1sub .or. j > n2sub .or. k > n3sub) return
        do iv = 1, m
            dst(i, j, k, iv) = src(i, j, k, iv)
        end do
    end subroutine copy_array_kernel

    ! --------------------------------------------------------------------
    ! Device helpers: construct flux Jacobian (A, B, or C) from primitive
    ! variables (rho, u, v, w, E) at a single grid point.
    ! Used by build_matrix_{x,y,z}_kernel to evaluate A_{i-1}, A_{i+1}
    ! (and analogously B_{j±1}, C_{k±1}) — i.e. "Method A" in the paper
    ! conservative discretization of ∂(A ΔU)/∂x ≈ (A_{i+1}ΔU_{i+1} -
    ! A_{i-1}ΔU_{i-1})/(2 Δx).
    ! --------------------------------------------------------------------
    attributes(device) subroutine compute_Amat(rho, uu, vv, ww, E, A_mat)
        real(8), intent(in) :: rho, uu, vv, ww, E
        real(8), intent(out) :: A_mat(5,5)
        real(8) :: q2, H_enth, c1, c2
        q2 = uu**2 + vv**2 + ww**2
        H_enth = E + (gamma - 1.0d0) * (E - 0.5d0 * q2)
        c1 = gamma - 1.0d0; c2 = 0.5d0 * c1
        A_mat = 0.0d0
        A_mat(1,2) = 1.0d0
        A_mat(2,1) = c2*q2 - uu**2; A_mat(2,2) = (3.0d0-gamma)*uu
        A_mat(2,3) = -c1*vv; A_mat(2,4) = -c1*ww; A_mat(2,5) = c1
        A_mat(3,1) = -uu*vv; A_mat(3,2) = vv; A_mat(3,3) = uu
        A_mat(4,1) = -uu*ww; A_mat(4,2) = ww; A_mat(4,4) = uu
        A_mat(5,1) = (c2*q2 - H_enth)*uu; A_mat(5,2) = H_enth - c1*uu**2
        A_mat(5,3) = -c1*uu*vv; A_mat(5,4) = -c1*uu*ww; A_mat(5,5) = gamma*uu
    end subroutine compute_Amat

    attributes(device) subroutine compute_Bmat(rho, uu, vv, ww, E, B_mat)
        real(8), intent(in) :: rho, uu, vv, ww, E
        real(8), intent(out) :: B_mat(5,5)
        real(8) :: q2, H_enth, c1, c2
        q2 = uu**2 + vv**2 + ww**2
        H_enth = E + (gamma - 1.0d0) * (E - 0.5d0 * q2)
        c1 = gamma - 1.0d0; c2 = 0.5d0 * c1
        B_mat = 0.0d0
        B_mat(1,3) = 1.0d0
        B_mat(2,1) = -uu*vv; B_mat(2,2) = vv; B_mat(2,3) = uu
        B_mat(3,1) = c2*q2 - vv**2; B_mat(3,2) = -c1*uu
        B_mat(3,3) = (3.0d0-gamma)*vv; B_mat(3,4) = -c1*ww; B_mat(3,5) = c1
        B_mat(4,1) = -vv*ww; B_mat(4,3) = ww; B_mat(4,4) = vv
        B_mat(5,1) = (c2*q2 - H_enth)*vv; B_mat(5,2) = -c1*uu*vv
        B_mat(5,3) = H_enth - c1*vv**2; B_mat(5,4) = -c1*vv*ww; B_mat(5,5) = gamma*vv
    end subroutine compute_Bmat

    attributes(device) subroutine compute_Cmat(rho, uu, vv, ww, E, C_mat)
        real(8), intent(in) :: rho, uu, vv, ww, E
        real(8), intent(out) :: C_mat(5,5)
        real(8) :: q2, H_enth, c1, c2
        q2 = uu**2 + vv**2 + ww**2
        H_enth = E + (gamma - 1.0d0) * (E - 0.5d0 * q2)
        c1 = gamma - 1.0d0; c2 = 0.5d0 * c1
        C_mat = 0.0d0
        C_mat(1,4) = 1.0d0
        C_mat(2,1) = -uu*ww; C_mat(2,2) = ww; C_mat(2,4) = uu
        C_mat(3,1) = -vv*ww; C_mat(3,3) = ww; C_mat(3,4) = vv
        C_mat(4,1) = c2*q2 - ww**2; C_mat(4,2) = -c1*uu; C_mat(4,3) = -c1*vv
        C_mat(4,4) = (3.0d0-gamma)*ww; C_mat(4,5) = c1
        C_mat(5,1) = (c2*q2 - H_enth)*ww; C_mat(5,2) = -c1*uu*ww
        C_mat(5,3) = -c1*vv*ww; C_mat(5,4) = H_enth - c1*ww**2; C_mat(5,5) = gamma*ww
    end subroutine compute_Cmat

    attributes(global) subroutine build_matrix_z_kernel(U, n1sub, n2sub, n3sub, m, trA, trB, trC, dz, dt, eps)
        integer, value, intent(in) :: n1sub, n2sub, n3sub, m
        real(8), device, intent(in) :: U(0:n1sub+1, 0:n2sub+1, 0:n3sub+1, m)
        real(8), device, intent(out) :: trA(n1sub,n2sub,n3sub,m,m), trB(n1sub,n2sub,n3sub,m,m), trC(n1sub,n2sub,n3sub,m,m)
        real(8), value, intent(in) :: dz, dt, eps

        real(8) :: C_left(5,5), C_right(5,5)
        real(8) :: rho, uu, vv, ww, E, coef_conv, coef_diff
        integer :: i, j, k, iv, jv

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = (blockIdx%z - 1) * blockDim%z + threadIdx%z
        if (i > n1sub .or. j > n2sub .or. k > n3sub) return

        coef_conv = dt / (4.0d0 * dz)
        coef_diff = eps * dt / (2.0d0 * dz**2)

        ! C_{k-1} from U(i,j,k-1,*)
        rho = U(i,j,k-1,1); uu = U(i,j,k-1,2)/rho; vv = U(i,j,k-1,3)/rho
        ww  = U(i,j,k-1,4)/rho; E  = U(i,j,k-1,5)/rho
        call compute_Cmat(rho, uu, vv, ww, E, C_left)

        ! C_{k+1} from U(i,j,k+1,*)
        rho = U(i,j,k+1,1); uu = U(i,j,k+1,2)/rho; vv = U(i,j,k+1,3)/rho
        ww  = U(i,j,k+1,4)/rho; E  = U(i,j,k+1,5)/rho
        call compute_Cmat(rho, uu, vv, ww, E, C_right)

        do jv = 1, m
            do iv = 1, m
                if (iv == jv) then
                    trA(i,j,k,iv,jv) = -coef_conv*C_left (iv,jv) - coef_diff
                    trB(i,j,k,iv,jv) = 1.0d0 + 2.0d0*coef_diff
                    trC(i,j,k,iv,jv) =  coef_conv*C_right(iv,jv) - coef_diff
                else
                    trA(i,j,k,iv,jv) = -coef_conv*C_left (iv,jv)
                    trB(i,j,k,iv,jv) = 0.0d0
                    trC(i,j,k,iv,jv) =  coef_conv*C_right(iv,jv)
                end if
            end do
        end do
    end subroutine build_matrix_z_kernel

    attributes(global) subroutine build_matrix_y_kernel(U, n1sub, n2sub, n3sub, m, trA, trB, trC, dy, dt, eps)
        integer, value, intent(in) :: n1sub, n2sub, n3sub, m
        real(8), device, intent(in) :: U(0:n1sub+1, 0:n2sub+1, 0:n3sub+1, m)
        real(8), device, intent(out) :: trA(n1sub,n2sub,n3sub,m,m), trB(n1sub,n2sub,n3sub,m,m), trC(n1sub,n2sub,n3sub,m,m)
        real(8), value, intent(in) :: dy, dt, eps

        real(8) :: B_left(5,5), B_right(5,5)
        real(8) :: rho, uu, vv, ww, E, coef_conv, coef_diff
        integer :: i, j, k, iv, jv

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = (blockIdx%z - 1) * blockDim%z + threadIdx%z
        if (i > n1sub .or. j > n2sub .or. k > n3sub) return

        coef_conv = dt / (4.0d0 * dy)
        coef_diff = eps * dt / (2.0d0 * dy**2)

        ! B_{j-1} from U(i,j-1,k,*)
        rho = U(i,j-1,k,1); uu = U(i,j-1,k,2)/rho; vv = U(i,j-1,k,3)/rho
        ww  = U(i,j-1,k,4)/rho; E  = U(i,j-1,k,5)/rho
        call compute_Bmat(rho, uu, vv, ww, E, B_left)

        ! B_{j+1} from U(i,j+1,k,*)
        rho = U(i,j+1,k,1); uu = U(i,j+1,k,2)/rho; vv = U(i,j+1,k,3)/rho
        ww  = U(i,j+1,k,4)/rho; E  = U(i,j+1,k,5)/rho
        call compute_Bmat(rho, uu, vv, ww, E, B_right)

        do jv = 1, m
            do iv = 1, m
                if (iv == jv) then
                    trA(i,j,k,iv,jv) = -coef_conv*B_left (iv,jv) - coef_diff
                    trB(i,j,k,iv,jv) = 1.0d0 + 2.0d0*coef_diff
                    trC(i,j,k,iv,jv) =  coef_conv*B_right(iv,jv) - coef_diff
                else
                    trA(i,j,k,iv,jv) = -coef_conv*B_left (iv,jv)
                    trB(i,j,k,iv,jv) = 0.0d0
                    trC(i,j,k,iv,jv) =  coef_conv*B_right(iv,jv)
                end if
            end do
        end do
    end subroutine build_matrix_y_kernel

    attributes(global) subroutine build_matrix_x_kernel(U, n1sub, n2sub, n3sub, m, trA, trB, trC, dx, dt, eps)
        integer, value, intent(in) :: n1sub, n2sub, n3sub, m
        real(8), device, intent(in) :: U(0:n1sub+1, 0:n2sub+1, 0:n3sub+1, m)
        real(8), device, intent(out) :: trA(n1sub,n2sub,n3sub,m,m), trB(n1sub,n2sub,n3sub,m,m), trC(n1sub,n2sub,n3sub,m,m)
        real(8), value, intent(in) :: dx, dt, eps

        real(8) :: A_left(5,5), A_right(5,5)
        real(8) :: rho, uu, vv, ww, E, coef_conv, coef_diff
        integer :: i, j, k, iv, jv

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = (blockIdx%z - 1) * blockDim%z + threadIdx%z
        if (i > n1sub .or. j > n2sub .or. k > n3sub) return

        coef_conv = dt / (4.0d0 * dx)
        coef_diff = eps * dt / (2.0d0 * dx**2)

        ! A_{i-1} from U(i-1,j,k,*)
        rho = U(i-1,j,k,1); uu = U(i-1,j,k,2)/rho; vv = U(i-1,j,k,3)/rho
        ww  = U(i-1,j,k,4)/rho; E  = U(i-1,j,k,5)/rho
        call compute_Amat(rho, uu, vv, ww, E, A_left)

        ! A_{i+1} from U(i+1,j,k,*)
        rho = U(i+1,j,k,1); uu = U(i+1,j,k,2)/rho; vv = U(i+1,j,k,3)/rho
        ww  = U(i+1,j,k,4)/rho; E  = U(i+1,j,k,5)/rho
        call compute_Amat(rho, uu, vv, ww, E, A_right)

        do jv = 1, m
            do iv = 1, m
                if (iv == jv) then
                    trA(i,j,k,iv,jv) = -coef_conv*A_left (iv,jv) - coef_diff
                    trB(i,j,k,iv,jv) = 1.0d0 + 2.0d0*coef_diff
                    trC(i,j,k,iv,jv) =  coef_conv*A_right(iv,jv) - coef_diff
                else
                    trA(i,j,k,iv,jv) = -coef_conv*A_left (iv,jv)
                    trB(i,j,k,iv,jv) = 0.0d0
                    trC(i,j,k,iv,jv) =  coef_conv*A_right(iv,jv)
                end if
            end do
        end do
    end subroutine build_matrix_x_kernel

    attributes(global) subroutine transpose_jk_kernel(U_in, U_out, n1sub, n2sub, n3sub, m)
        integer, value, intent(in) :: n1sub, n2sub, n3sub, m
        real(8), device, intent(in) :: U_in(n1sub, n2sub, n3sub, m)
        real(8), device, intent(out) :: U_out(n1sub, n3sub, n2sub, m)
        integer :: i, j, k, iv
        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = (blockIdx%z - 1) * blockDim%z + threadIdx%z
        if (i > n1sub .or. j > n2sub .or. k > n3sub) return
        do iv = 1, m
            U_out(i, k, j, iv) = U_in(i, j, k, iv)
        end do
    end subroutine transpose_jk_kernel

    attributes(global) subroutine transpose_jk_5d_kernel(A_in, A_out, n1sub, n2sub, n3sub, m)
        integer, value, intent(in) :: n1sub, n2sub, n3sub, m
        real(8), device, intent(in) :: A_in(n1sub, n2sub, n3sub, m, m)
        real(8), device, intent(out) :: A_out(n1sub, n3sub, n2sub, m, m)
        integer :: i, j, k, iv, jv
        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = (blockIdx%z - 1) * blockDim%z + threadIdx%z
        if (i > n1sub .or. j > n2sub .or. k > n3sub) return
        do jv = 1, m
            do iv = 1, m
                A_out(i, k, j, iv, jv) = A_in(i, j, k, iv, jv)
            end do
        end do
    end subroutine transpose_jk_5d_kernel

    attributes(global) subroutine transpose_kj_kernel(U_in, U_out, n1sub, n2sub, n3sub, m)
        integer, value, intent(in) :: n1sub, n2sub, n3sub, m
        real(8), device, intent(in) :: U_in(n1sub, n3sub, n2sub, m)
        real(8), device, intent(out) :: U_out(n1sub, n2sub, n3sub, m)
        integer :: i, j, k, iv
        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = (blockIdx%z - 1) * blockDim%z + threadIdx%z
        if (i > n1sub .or. j > n2sub .or. k > n3sub) return
        do iv = 1, m
            U_out(i, j, k, iv) = U_in(i, k, j, iv)  ! (i,k,j) → (i,j,k)
        end do
    end subroutine transpose_kj_kernel

    attributes(global) subroutine transpose_ik_kernel(U_in, U_out, n1sub, n2sub, n3sub, m)
        integer, value, intent(in) :: n1sub, n2sub, n3sub, m
        real(8), device, intent(in) :: U_in(n1sub, n2sub, n3sub, m)
        real(8), device, intent(out) :: U_out(n2sub, n3sub, n1sub, m)
        integer :: i, j, k, iv
        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = (blockIdx%z - 1) * blockDim%z + threadIdx%z
        if (i > n1sub .or. j > n2sub .or. k > n3sub) return
        do iv = 1, m
            U_out(j, k, i, iv) = U_in(i, j, k, iv)
        end do
    end subroutine transpose_ik_kernel

    attributes(global) subroutine transpose_ik_5d_kernel(A_in, A_out, n1sub, n2sub, n3sub, m)
        integer, value, intent(in) :: n1sub, n2sub, n3sub, m
        real(8), device, intent(in) :: A_in(n1sub, n2sub, n3sub, m, m)
        real(8), device, intent(out) :: A_out(n2sub, n3sub, n1sub, m, m)
        integer :: i, j, k, iv, jv
        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = (blockIdx%z - 1) * blockDim%z + threadIdx%z
        if (i > n1sub .or. j > n2sub .or. k > n3sub) return
        do jv = 1, m
            do iv = 1, m
                A_out(j, k, i, iv, jv) = A_in(i, j, k, iv, jv)
            end do
        end do
    end subroutine transpose_ik_5d_kernel

    attributes(global) subroutine transpose_ki_kernel(U_in, U_out, n1sub, n2sub, n3sub, m)
        integer, value, intent(in) :: n1sub, n2sub, n3sub, m
        real(8), device, intent(in) :: U_in(n2sub, n3sub, n1sub, m)
        real(8), device, intent(out) :: U_out(n1sub, n2sub, n3sub, m)
        integer :: i, j, k, iv
        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = (blockIdx%z - 1) * blockDim%z + threadIdx%z
        if (i > n1sub .or. j > n2sub .or. k > n3sub) return
        do iv = 1, m
            U_out(i, j, k, iv) = U_in(j, k, i, iv)  ! (j,k,i) → (i,j,k)
        end do
    end subroutine transpose_ki_kernel

    attributes(global) subroutine update_solution_kernel(U, dU, n1sub, n2sub, n3sub, m)
        integer, value, intent(in) :: n1sub, n2sub, n3sub, m
        real(8), device, intent(inout) :: U(0:n1sub+1, 0:n2sub+1, 0:n3sub+1, m)
        real(8), device, intent(in) :: dU(n1sub, n2sub, n3sub, m)
        integer :: i, j, k, iv
        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = (blockIdx%z - 1) * blockDim%z + threadIdx%z
        if (i > n1sub .or. j > n2sub .or. k > n3sub) return
        do iv = 1, m
            U(i, j, k, iv) = U(i, j, k, iv) + dU(i, j, k, iv)
        end do
    end subroutine update_solution_kernel

end module kernels

program euler3d_btdma_cuda
    use cudafor
    use params
    use kernels
    use mpi
    use mod_btdma_gpu_v2
    use mpiutil
    implicit none
    integer :: nprocs, myrank, ierr
    integer :: globindx_xa, globindx_xb;
    integer :: globindx_ya, globindx_yb;
    integer :: globindx_za, globindx_zb;
    integer :: n1sub,n2sub,n3sub
    integer :: i,j,k

    integer :: dev
    type(cudadeviceprop) :: prop

    type(BTDMA_PLAN_gpu_v2) :: plan_x, plan_y, plan_z

    real(8) :: t_flux_a=0.d0,t_comm_a=0.d0,t_rhs_a=0.d0,t_copy_a=0.d0,t_matrix_a=0.d0,t_trans_a=0.d0,t_solve_a=0.d0,t_update_a=0.d0
    real(8) :: t_flux_b=0.d0,t_comm_b=0.d0,t_rhs_b=0.d0,t_copy_b=0.d0,t_matrix_b=0.d0,t_trans_b=0.d0,t_solve_b=0.d0,t_update_b=0.d0
    !------ Problem parameters ------
    real(8) :: dx, dy, dz
    real(8) :: dt, time, t_final, cfl_number
    integer :: nstep, istep, istat
    real(8), allocatable :: x(:), y(:), z(:)
    real(8), allocatable :: xsub(:), ysub(:), zsub(:)
    real(8), allocatable :: x_ext(:), y_ext(:), z_ext(:)
    real(8), allocatable :: U_host(:,:,:,:)
    
    real(8), device, allocatable :: xsub_d(:), ysub_d(:), zsub_d(:)
    real(8), device, allocatable :: U_d(:,:,:,:), dU_d(:,:,:,:), RHS_d(:,:,:,:)
    real(8), device, allocatable :: Flux_F_d(:,:,:,:), Flux_G_d(:,:,:,:), Flux_H_d(:,:,:,:)
    real(8), device, allocatable :: trA_d(:,:,:,:,:), trB_d(:,:,:,:,:), trC_d(:,:,:,:,:)
    real(8), device, allocatable :: dU_work_d(:,:,:,:)
    real(8), device, allocatable :: trA_work_d(:,:,:,:,:), trB_work_d(:,:,:,:,:), trC_work_d(:,:,:,:,:)
    ! real(8), device, allocatable :: trE_work_d(:,:,:,:,:)
    
    type(dim3) :: blockSize3D, gridSize3D
    real(8) :: timing_local(14), timing_max(14)
    !-------------------------------
    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)

    if (nprocs /= np1*np2*np3) then
        if (myrank == 0) then
            write(*,'(A,I0,A,I0)') 'ERROR: MPI ranks = ', nprocs, &
                                   ', but NP1*NP2*NP3 = ', np1*np2*np3
        end if
        call MPI_Abort(MPI_COMM_WORLD, 2, ierr)
    end if

    ! Assign GPU: use node-local rank and query GPU count at runtime so the
    ! binding works regardless of GPUs-per-node (1, 2, 4, 8, ...).
    block
        integer :: local_comm, local_rank, ngpu, dev_id
        call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, &
                                 MPI_INFO_NULL, local_comm, ierr)
        call MPI_Comm_rank(local_comm, local_rank, ierr)
        ierr = cudaGetDeviceCount(ngpu)
        if (ngpu <= 0) then
            if (myrank == 0) write(*,*) "ERROR: no CUDA devices visible"
            call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        end if
        dev_id = mod(local_rank, ngpu)
        ierr = cudaSetDevice(dev_id)
        ierr = cudaGetDevice(dev)
        ierr = cudaGetDeviceProperties(prop, dev)
        call MPI_Comm_free(local_comm, ierr)
        if (myrank == 0) write(*,'(A,I0,A)') ' Detected ', ngpu, ' GPU(s) per node'
    end block

    np_dim(0:2) =(/np1,np2,np3/)
    period(0)=.true.; period(1)=.true.; period(2)=.true.
    call mpi_topology_make()

    n1sub = mpiutil_para(1, n1, comm_1d_x%myrank, comm_1d_x%nprocs, globindx_xa, globindx_xb)
    n2sub = mpiutil_para(1, n2, comm_1d_y%myrank, comm_1d_y%nprocs, globindx_ya, globindx_yb)
    n3sub = mpiutil_para(1, n3, comm_1d_z%myrank, comm_1d_z%nprocs, globindx_za, globindx_zb)

    call mpi_halocomm_plan(n1sub,n2sub,n3sub,comm_1d_x, comm_1d_y, comm_1d_z)

    !**************************************************************************************************************
    !** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** 
    allocate(x(1:n1), y(1:n2), z(1:n3))
    allocate(x_ext(0:n1+1), y_ext(0:n2+1), z_ext(0:n3+1))
    allocate(xsub(0:n1sub+1), ysub(0:n2sub+1), zsub(0:n3sub+1))
    allocate(xsub_d(0:n1sub+1), ysub_d(0:n2sub+1), zsub_d(0:n3sub+1))
    allocate(U_host(0:n1sub+1,0:n2sub+1,0:n3sub+1,m))

    allocate(U_d(0:n1sub+1,0:n2sub+1,0:n3sub+1,m), dU_d(n1sub,n2sub,n3sub,m), RHS_d(n1sub,n2sub,n3sub,m))
    allocate(Flux_F_d(0:n1sub+1,0:n2sub+1,0:n3sub+1,m))
    allocate(Flux_G_d(0:n1sub+1,0:n2sub+1,0:n3sub+1,m))
    allocate(Flux_H_d(0:n1sub+1,0:n2sub+1,0:n3sub+1,m))
    allocate(trA_d(n1sub,n2sub,n3sub,m,m), trB_d(n1sub,n2sub,n3sub,m,m), trC_d(n1sub,n2sub,n3sub,m,m))
    allocate(dU_work_d(n1sub,n2sub,n3sub,m))
    allocate(trA_work_d(n1sub,n2sub,n3sub,m,m), trB_work_d(n1sub,n2sub,n3sub,m,m), trC_work_d(n1sub,n2sub,n3sub,m,m))

    dx = Lx / dble(n1); dy = Ly / dble(n2); dz = Lz / dble(n3)
    do i = 1, n1
        x(i) = (dble(i) - 0.5d0) * dx
    end do
    do j = 1, n2
        y(j) = (dble(j) - 0.5d0) * dy
    end do
    do k = 1, n3
        z(k) = (dble(k) - 0.5d0) * dz
    end do
    x_ext(1:n1) = x; y_ext(1:n2) = y; z_ext(1:n3) = z
    x_ext(0) = x(n1);   x_ext(n1+1) = x(1)
    y_ext(0) = y(n2);   y_ext(n2+1) = y(1)
    z_ext(0) = z(n3);   z_ext(n3+1) = z(1)
    do i = 0, n1sub+1
        xsub(i) = x_ext(globindx_xa + i - 1) 
    end do
    do j = 0, n2sub+1
        ysub(j) = y_ext(globindx_ya + j - 1) 
    end do
    do k = 0, n3sub+1
        zsub(k) = z_ext(globindx_za + k - 1) 
    end do
    xsub_d = xsub; ysub_d = ysub; zsub_d = zsub
    if(comm_1d_x%myrank == 0) xsub(0) = xsub(1)-dx
    if(comm_1d_y%myrank == 0) ysub(0) = ysub(1)-dy
    if(comm_1d_z%myrank == 0) zsub(0) = zsub(1)-dz
    if(comm_1d_x%myrank == comm_1d_x%nprocs-1) xsub(n1sub+1) = xsub(n1sub)+dx
    if(comm_1d_y%myrank == comm_1d_y%nprocs-1) ysub(n2sub+1) = ysub(n2sub)+dy
    if(comm_1d_z%myrank == comm_1d_z%nprocs-1) zsub(n3sub+1) = zsub(n3sub)+dz
    
    dt = dx
    t_final = Lx / u_inf
    nstep = nint(t_final / dt)
    if (nstep_override > 0) then
        nstep = nstep_override
        t_final = dt*dble(nstep)
    end if
    
    blockSize3D = dim3(8, 8, 8)
    gridSize3D  = dim3((n1sub+7)/8, (n2sub+7)/8, (n3sub+7)/8)
    call initialize_vortex_kernel0<<<gridSize3D, blockSize3D>>>(U_d, xsub_d, ysub_d, zsub_d, n1sub,n2sub,n3sub, m,0.0d0)
    call mpi_halocomm_exchange_halo_3d_gpu(U_d, n1sub, n2sub, n3sub, m, comm_1d_x, comm_1d_y, comm_1d_z)

    U_host = U_d
    call write_centerline_csv(U_host, xsub, n1sub, n2sub, n3sub, m, myrank, &
                              globindx_xa, globindx_ya, globindx_za, n2, n3, &
                              dx, dy, 'initial')

    call btdma_makeplan_gpu_v2(plan_x,m,n3sub*n2sub,n1sub,comm_1d_x%mpi_comm)
    call btdma_makeplan_gpu_v2(plan_y,m,n1sub*n3sub,n2sub,comm_1d_y%mpi_comm)
    call btdma_makeplan_gpu_v2(plan_z,m,n1sub*n2sub,n3sub,comm_1d_z%mpi_comm)

    time = 0.0d0

    ! --- CFL computation (before time loop) ---
    block
        real(8) :: rho_loc, uu_loc, vv_loc, ww_loc, p_loc, c_loc
        real(8) :: max_speed_loc, max_speed_global
        U_host = U_d
        max_speed_loc = 0.0d0
        do k = 1, n3sub
            do j = 1, n2sub
                do i = 1, n1sub
                    rho_loc = U_host(i,j,k,1)
                    uu_loc  = U_host(i,j,k,2) / rho_loc
                    vv_loc  = U_host(i,j,k,3) / rho_loc
                    ww_loc  = U_host(i,j,k,4) / rho_loc
                    p_loc   = (gamma - 1.0d0) * (U_host(i,j,k,5) - 0.5d0 * rho_loc * (uu_loc**2 + vv_loc**2 + ww_loc**2))
                    c_loc   = sqrt(gamma * p_loc / rho_loc)
                    max_speed_loc = max(max_speed_loc, abs(uu_loc)+abs(vv_loc)+abs(ww_loc)+c_loc)
                end do
            end do
        end do
        call MPI_Allreduce(max_speed_loc, max_speed_global, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr)
        cfl_number = max_speed_global * dt / min(dx, dy, dz)
        if (myrank == 0) then
            write(*,'(A,F12.4)') " CFL number     = ", cfl_number
            write(*,'(A,E12.4)') " dt             = ", dt
            write(*,'(A,E12.4)') " dx             = ", dx
            write(*,'(A,I8)')    " nstep          = ", nstep
            write(*,'(A)') 'RUN_CONFIG_CSV,n1,n2,n3,np1,np2,np3,m,nstep,dt,t_final,cfl,epsilon'
            write(*,'(A,8(",",I0),4(",",ES24.16E3))') 'RUN_CONFIG_CSV', &
                n1, n2, n3, np1, np2, np3, m, nstep, dt, t_final, cfl_number, epsilon_diff
        end if
    end block

    do istep = 1,nstep
        time = time + dt

        if(istep>1) t_flux_a = MPI_WTIME()
        call compute_fluxes_kernel<<<gridSize3D, blockSize3D>>>(U_d, n1sub, n2sub, n3sub, m, Flux_F_d, Flux_G_d, Flux_H_d)
        istat = cudaDeviceSynchronize()
        if(istep>1) t_flux_b = t_flux_b+ MPI_WTIME()-t_flux_a

        if(istep>1) t_comm_a = MPI_WTIME()
        call mpi_halocomm_exchange_halo_3d_gpu(Flux_F_d, n1sub, n2sub, n3sub, m, comm_1d_x, comm_1d_y, comm_1d_z)
        call mpi_halocomm_exchange_halo_3d_gpu(Flux_G_d, n1sub, n2sub, n3sub, m, comm_1d_x, comm_1d_y, comm_1d_z)
        call mpi_halocomm_exchange_halo_3d_gpu(Flux_H_d, n1sub, n2sub, n3sub, m, comm_1d_x, comm_1d_y, comm_1d_z)
        istat = cudaDeviceSynchronize()
        if(istep>1) t_comm_b = t_comm_b+ MPI_WTIME()-t_comm_a

        if(istep>1) t_rhs_a = MPI_WTIME()
        call compute_RHS_kernel<<<gridSize3D, blockSize3D>>>(U_d, n1sub, n2sub, n3sub, m, Flux_F_d, Flux_G_d, Flux_H_d, &
                                                             RHS_d, dx, dy, dz, dt, epsilon_diff)
        istat = cudaDeviceSynchronize()
        if(istep>1) t_rhs_b = t_rhs_b+ MPI_WTIME()-t_rhs_a
              
        if(istep>1) t_copy_a = MPI_WTIME()
        call copy_array_kernel<<<gridSize3D, blockSize3D>>>(RHS_d, dU_d, n1sub, n2sub, n3sub, m)
        istat = cudaDeviceSynchronize()
        if(istep>1) t_copy_b = t_copy_b+ MPI_WTIME()-t_copy_a

        ! Z-sweep
        if(istep>1) t_matrix_a = MPI_WTIME()
        call build_matrix_z_kernel<<<gridSize3D, blockSize3D>>>(U_d, n1sub, &
            n2sub, n3sub, m, trA_d, trB_d, trC_d, dz, dt, epsilon_diff)
        istat = cudaDeviceSynchronize()
        if(istep>1) t_matrix_b = t_matrix_b+ MPI_WTIME()-t_matrix_a

        if(istep>1) t_solve_a = MPI_WTIME()
        call btdma_many_cycl_mpi_gpu_v2(trA_d,trB_d,trC_d,dU_d,m,n1sub*n2sub,n3sub,plan_z)
        istat = cudaDeviceSynchronize()
        if(istep>1) t_solve_b = t_solve_b+ MPI_WTIME()-t_solve_a

        ! Y-sweep
        if(istep>1) t_matrix_a = MPI_WTIME()
        call build_matrix_y_kernel<<<gridSize3D, blockSize3D>>>(U_d, n1sub, &
            n2sub, n3sub, m, trA_d, trB_d, trC_d, dy, dt, epsilon_diff)
        istat = cudaDeviceSynchronize()
        if(istep>1) t_matrix_b = t_matrix_b+ MPI_WTIME()-t_matrix_a

        if(istep>1) t_trans_a = MPI_WTIME()
        call transpose_jk_kernel<<<gridSize3D, blockSize3D>>>(dU_d, dU_work_d, n1sub, n2sub, n3sub, m)
        call transpose_jk_5d_kernel<<<gridSize3D, blockSize3D>>>(trA_d, trA_work_d, n1sub, n2sub, n3sub, m)
        call transpose_jk_5d_kernel<<<gridSize3D, blockSize3D>>>(trB_d, trB_work_d, n1sub, n2sub, n3sub, m)
        call transpose_jk_5d_kernel<<<gridSize3D, blockSize3D>>>(trC_d, trC_work_d, n1sub, n2sub, n3sub, m)
        istat = cudaDeviceSynchronize()
        if(istep>1) t_trans_b = t_trans_b+ MPI_WTIME()-t_trans_a


        if(istep>1) t_solve_a = MPI_WTIME()
        call btdma_many_cycl_mpi_gpu_v2(trA_work_d,trB_work_d,trC_work_d,dU_work_d,m,n1sub*n3sub,n2sub,plan_y)
        istat = cudaDeviceSynchronize()
        if(istep>1) t_solve_b = t_solve_b+ MPI_WTIME()-t_solve_a

        if(istep>1) t_trans_a = MPI_WTIME()
        call transpose_kj_kernel<<<gridSize3D, blockSize3D>>>(dU_work_d, dU_d, n1sub, n2sub, n3sub, m)
        istat = cudaDeviceSynchronize()
        if(istep>1) t_trans_b = t_trans_b+ MPI_WTIME()-t_trans_a

        ! X-sweep
        if(istep>1) t_matrix_a = MPI_WTIME()
        call build_matrix_x_kernel<<<gridSize3D, blockSize3D>>>(U_d, n1sub, &
            n2sub, n3sub, m, trA_d, trB_d, trC_d, dx, dt, epsilon_diff)
        istat = cudaDeviceSynchronize()
        if(istep>1) t_matrix_b = t_matrix_b+ MPI_WTIME()-t_matrix_a

        if(istep>1) t_trans_a = MPI_WTIME()
        call transpose_ik_kernel<<<gridSize3D, blockSize3D>>>(dU_d, dU_work_d, n1sub, n2sub, n3sub, m)
        call transpose_ik_5d_kernel<<<gridSize3D, blockSize3D>>>(trA_d, trA_work_d, n1sub, n2sub, n3sub, m)
        call transpose_ik_5d_kernel<<<gridSize3D, blockSize3D>>>(trB_d, trB_work_d, n1sub, n2sub, n3sub, m)
        call transpose_ik_5d_kernel<<<gridSize3D, blockSize3D>>>(trC_d, trC_work_d, n1sub, n2sub, n3sub, m)
        istat = cudaDeviceSynchronize()
        if(istep>1) t_trans_b = t_trans_b+ MPI_WTIME()-t_trans_a

        if(istep>1) t_solve_a = MPI_WTIME()
        call btdma_many_cycl_mpi_gpu_v2(trA_work_d,trB_work_d,trC_work_d,dU_work_d,m,n2sub*n3sub,n1sub,plan_x)
        istat = cudaDeviceSynchronize()
        if(istep>1) t_solve_b = t_solve_b+ MPI_WTIME()-t_solve_a

        if(istep>1) t_trans_a = MPI_WTIME()
        call transpose_ki_kernel<<<gridSize3D, blockSize3D>>>(dU_work_d, dU_d, n1sub, n2sub, n3sub, m)
        istat = cudaDeviceSynchronize()
        if(istep>1) t_trans_b = t_trans_b+ MPI_WTIME()-t_trans_a

        if(istep>1) t_update_a = MPI_WTIME()
        call update_solution_kernel<<<gridSize3D, blockSize3D>>>(U_d, dU_d, n1sub, n2sub, n3sub, m)
        istat = cudaDeviceSynchronize()
        if(istep>1) t_update_b = t_update_b+ MPI_WTIME()-t_update_a

        if(istep>1) t_comm_a = MPI_WTIME()
        call mpi_halocomm_exchange_halo_3d_gpu(U_d, n1sub, n2sub, n3sub, m, comm_1d_x, comm_1d_y, comm_1d_z)
        istat = cudaDeviceSynchronize()
        if(istep>1) t_comm_b = t_comm_b+ MPI_WTIME()-t_comm_a

        ! Step 1 is a warm-up. Reset the cumulative solver phase timers so
        ! the application and solver-phase totals cover the same timed steps.
        if (istep == 1) then
            t1__b = 0.0d0
            t2__b = 0.0d0
            t3__b = 0.0d0
            t4__b = 0.0d0
            t5__b = 0.0d0
        end if

    enddo
    
    U_host = U_d
    call write_centerline_csv(U_host, xsub, n1sub, n2sub, n3sub, m, myrank, &
                              globindx_xa, globindx_ya, globindx_za, n2, n3, &
                              dx, dy, 'final')

    ! Solver-independent validation against the analytically translated vortex.
    block
        real(8) :: num_loc(3), den_loc(3), num_global(3), den_global(3), rel_l2(3)
        real(8) :: rho_num, rho_ref, p_num, p_ref
        real(8) :: u_num_m, u_num_p, v_num_m, v_num_p
        real(8) :: u_ref_m, u_ref_p, v_ref_m, v_ref_p
        real(8) :: omega_num, omega_ref
        real(8), device, allocatable :: U_exact_d(:,:,:,:)
        real(8), allocatable :: U_num_host(:,:,:,:)

        allocate(U_exact_d(0:n1sub+1, 0:n2sub+1, 0:n3sub+1, m))
        allocate(U_num_host(0:n1sub+1, 0:n2sub+1, 0:n3sub+1, m))
        call initialize_vortex_kernel0<<<gridSize3D, blockSize3D>>>(U_exact_d, &
            xsub_d, ysub_d, zsub_d, n1sub, n2sub, n3sub, m, time)
        istat = cudaDeviceSynchronize()
        call mpi_halocomm_exchange_halo_3d_gpu(U_exact_d, n1sub, n2sub, n3sub, &
            m, comm_1d_x, comm_1d_y, comm_1d_z)

        U_num_host = U_d
        U_host = U_exact_d

        num_loc = 0.0d0
        den_loc = 0.0d0
        do k = 1, n3sub
            do j = 1, n2sub
                do i = 1, n1sub
                    rho_num = U_num_host(i,j,k,1)
                    rho_ref = U_host(i,j,k,1)
                    p_num = (gamma - 1.0d0) * (U_num_host(i,j,k,5) - &
                        0.5d0*(U_num_host(i,j,k,2)**2 + U_num_host(i,j,k,3)**2 + &
                               U_num_host(i,j,k,4)**2)/rho_num)
                    p_ref = (gamma - 1.0d0) * (U_host(i,j,k,5) - &
                        0.5d0*(U_host(i,j,k,2)**2 + U_host(i,j,k,3)**2 + &
                               U_host(i,j,k,4)**2)/rho_ref)

                    v_num_m = U_num_host(i-1,j,k,3)/U_num_host(i-1,j,k,1)
                    v_num_p = U_num_host(i+1,j,k,3)/U_num_host(i+1,j,k,1)
                    u_num_m = U_num_host(i,j-1,k,2)/U_num_host(i,j-1,k,1)
                    u_num_p = U_num_host(i,j+1,k,2)/U_num_host(i,j+1,k,1)
                    v_ref_m = U_host(i-1,j,k,3)/U_host(i-1,j,k,1)
                    v_ref_p = U_host(i+1,j,k,3)/U_host(i+1,j,k,1)
                    u_ref_m = U_host(i,j-1,k,2)/U_host(i,j-1,k,1)
                    u_ref_p = U_host(i,j+1,k,2)/U_host(i,j+1,k,1)
                    omega_num = (v_num_p-v_num_m)/(2.0d0*dx) - &
                                (u_num_p-u_num_m)/(2.0d0*dy)
                    omega_ref = (v_ref_p-v_ref_m)/(2.0d0*dx) - &
                                (u_ref_p-u_ref_m)/(2.0d0*dy)

                    num_loc(1) = num_loc(1) + (rho_num-rho_ref)**2
                    num_loc(2) = num_loc(2) + (p_num-p_ref)**2
                    num_loc(3) = num_loc(3) + (omega_num-omega_ref)**2
                    den_loc(1) = den_loc(1) + rho_ref**2
                    den_loc(2) = den_loc(2) + p_ref**2
                    den_loc(3) = den_loc(3) + omega_ref**2
                end do
            end do
        end do
        deallocate(U_num_host)

        call MPI_Allreduce(num_loc, num_global, 3, MPI_DOUBLE_PRECISION, &
                           MPI_SUM, MPI_COMM_WORLD, ierr)
        call MPI_Allreduce(den_loc, den_global, 3, MPI_DOUBLE_PRECISION, &
                           MPI_SUM, MPI_COMM_WORLD, ierr)
        rel_l2 = sqrt(num_global/max(den_global, tiny(1.0d0)))

        if (myrank == 0) then
            write(*,'(A)') 'VALIDATION_CSV,field,relative_l2'
            write(*,'(A,",",ES24.16E3)') 'VALIDATION_CSV,rho', rel_l2(1)
            write(*,'(A,",",ES24.16E3)') 'VALIDATION_CSV,pressure', rel_l2(2)
            write(*,'(A,",",ES24.16E3)') 'VALIDATION_CSV,vorticity_z', rel_l2(3)
        end if

        deallocate(U_exact_d)

    end block

    timing_local(2:9) = (/t_flux_b, t_comm_b, t_rhs_b, t_copy_b, &
                          t_matrix_b, t_trans_b, t_solve_b, t_update_b/)
    timing_local(1) = sum(timing_local(2:9))
    timing_local(10:14) = (/t1__b, t2__b, t3__b, t4__b, t5__b/)
    call MPI_Reduce(timing_local, timing_max, 14, MPI_DOUBLE_PRECISION, MPI_MAX, &
                    0, MPI_COMM_WORLD, ierr)

    if (myrank == 0) then
        write(*,'(A)') 'TIMING_CSV,scope,seconds,timed_steps'
        write(*,'(A,",",ES24.16E3,",",I0)') 'TIMING_CSV,total', timing_max(1), max(0,nstep-1)
        write(*,'(A,",",ES24.16E3,",",I0)') 'TIMING_CSV,flux', timing_max(2), max(0,nstep-1)
        write(*,'(A,",",ES24.16E3,",",I0)') 'TIMING_CSV,halo', timing_max(3), max(0,nstep-1)
        write(*,'(A,",",ES24.16E3,",",I0)') 'TIMING_CSV,rhs', timing_max(4), max(0,nstep-1)
        write(*,'(A,",",ES24.16E3,",",I0)') 'TIMING_CSV,copy', timing_max(5), max(0,nstep-1)
        write(*,'(A,",",ES24.16E3,",",I0)') 'TIMING_CSV,matrix', timing_max(6), max(0,nstep-1)
        write(*,'(A,",",ES24.16E3,",",I0)') 'TIMING_CSV,transpose', timing_max(7), max(0,nstep-1)
        write(*,'(A,",",ES24.16E3,",",I0)') 'TIMING_CSV,solve', timing_max(8), max(0,nstep-1)
        write(*,'(A,",",ES24.16E3,",",I0)') 'TIMING_CSV,update', timing_max(9), max(0,nstep-1)
        write(*,'(A)') 'SOLVER_PHASE_CSV,phase,seconds,timed_steps'
        write(*,'(A,",",ES24.16E3,",",I0)') 'SOLVER_PHASE_CSV,local_modified', timing_max(10), max(0,nstep-1)
        write(*,'(A,",",ES24.16E3,",",I0)') 'SOLVER_PHASE_CSV,forward_exchange', timing_max(11), max(0,nstep-1)
        write(*,'(A,",",ES24.16E3,",",I0)') 'SOLVER_PHASE_CSV,reduced_solve', timing_max(12), max(0,nstep-1)
        write(*,'(A,",",ES24.16E3,",",I0)') 'SOLVER_PHASE_CSV,backward_exchange', timing_max(13), max(0,nstep-1)
        write(*,'(A,",",ES24.16E3,",",I0)') 'SOLVER_PHASE_CSV,local_update', timing_max(14), max(0,nstep-1)
    end if

    ! write(*,*) '******', ' Step = ', istep, ' rank = ', myrank
    call btdma_cleanplan_gpu_v2(plan_x) 
    call btdma_cleanplan_gpu_v2(plan_y) 
    call btdma_cleanplan_gpu_v2(plan_z) 
    !<<<<
    
    deallocate(U_d, dU_d, RHS_d, Flux_F_d, Flux_G_d, Flux_H_d)
    deallocate(trA_d, trB_d, trC_d, dU_work_d, trA_work_d, trB_work_d, trC_work_d)
    ! deallocate(trE_work_d)

    deallocate(x, y, z)
    deallocate(x_ext, y_ext, z_ext)
    deallocate(xsub, ysub, zsub)
    deallocate(xsub_d, ysub_d, zsub_d)
    deallocate(U_host)
    !** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** 
    !**************************************************************************************************************

    call mpi_halocomm_clean()
    call mpi_topology_clean()
    call MPI_FINALIZE(ierr)
end program euler3d_btdma_cuda


subroutine output_solution(U, x, y, z, nsub1, nsub2, nsub3, m, rank, istep)
    implicit none
    integer, intent(in) :: nsub1, nsub2, nsub3, m, rank, istep
    real(8), intent(in) :: U(0:nsub1+1,0:nsub2+1,0:nsub3+1,m), x(0:nsub1+1), y(0:nsub2+1), z(0:nsub3+1)
    real(8), parameter :: gamma = 1.4d0
    character(len=100) :: filename
    integer :: i, j, k
    real(8) :: rho, uu, vv, ww, rhoE, p
    
    integer :: writeunit

    write(filename, '(A,I5.5,A,I5.5,A)') 'output_',rank, '_', istep, '.vtk'
    open(newunit=writeunit, file=trim(filename), status='replace')
    
    write(writeunit, '(A)') '# vtk DataFile Version 3.0'
    write(writeunit, '(A,I5)') 'Euler3D Step ', istep
    write(writeunit, '(A)') 'ASCII'
    write(writeunit, '(A)') 'DATASET STRUCTURED_GRID'
    write(writeunit, '(A,3I6)') 'DIMENSIONS ', nsub1+2, nsub2+2, nsub3+2
    write(writeunit, '(A,I10,A)') 'POINTS ', (nsub1+2)*(nsub2+2)*(nsub3+2), ' double'
    
    do k = 0, nsub3+1
        do j = 0, nsub2+1
            do i = 0, nsub1+1
                write(writeunit, '(3E16.8)') x(i), y(j), z(k)
            end do
        end do
    end do
    
    write(writeunit, '(A,I10)') 'POINT_DATA ', (nsub1+2)*(nsub2+2)*(nsub3+2)
    
    write(writeunit, '(A)') 'SCALARS density double 1'
    write(writeunit, '(A)') 'LOOKUP_TABLE default'
    do k = 0, nsub3+1
        do j = 0, nsub2+1
            do i = 0, nsub1+1
                write(writeunit, '(E16.8)') U(i,j,k,1)
            end do
        end do
    end do
    
    write(writeunit, '(A)') 'SCALARS pressure double 1'
    write(writeunit, '(A)') 'LOOKUP_TABLE default'
    do k = 0, nsub3+1
        do j = 0, nsub2+1
            do i = 0, nsub1+1
                rho = U(i,j,k,1); uu = U(i,j,k,2)/rho; vv = U(i,j,k,3)/rho
                ww = U(i,j,k,4)/rho; rhoE = U(i,j,k,5)
                p = (gamma - 1.0d0) * (rhoE - 0.5d0 * rho * (uu**2 + vv**2 + ww**2))
                write(writeunit, '(E16.8)') p
                ! write(writeunit, '(E16.8)') U(i,j,k,2)
            end do
        end do
    end do
    
    write(writeunit, '(A)') 'VECTORS velocity double'
    do k = 0, nsub3+1
        do j = 0, nsub2+1
            do i = 0, nsub1+1
                rho = U(i,j,k,1)
                write(writeunit, '(3E16.8)') U(i,j,k,2)/rho, U(i,j,k,3)/rho, U(i,j,k,4)/rho
                ! write(writeunit, '(3E16.8)') U(i,j,k,3), U(i,j,k,4), U(i,j,k,5)
            end do
        end do
    end do
    
    close(writeunit)
    if(rank == 0) write(*,'(A,A)') ' Output: ', trim(filename)
end subroutine output_solution

subroutine output_solution_2D(U, x, y, z, nsub1, nsub2, nsub3, m, rank, istep)
    implicit none
    integer, intent(in) :: nsub1, nsub2, nsub3, m, rank, istep
    real(8), intent(in) :: U(0:nsub1+1,0:nsub2+1,0:nsub3+1,m), x(0:nsub1+1), y(0:nsub2+1), z(0:nsub3+1)
    real(8), parameter :: gamma = 1.4d0
    character(len=100) :: filename
    integer :: i, j, k
    integer :: im, ip, jm, jp
    integer :: jc, kc
    real(8) :: rho, uu, vv, ww, rhoE, p
    real(8) :: dvdx, dudy, omega_z

    integer :: writeunit

    write(filename, '(A,I5.5,A,I5.5,A)') 'output_XY_',rank, '_', istep, '.vtk'
    open(newunit=writeunit, file=trim(filename), status='replace')
    
    write(writeunit, '(A)') '# vtk DataFile Version 3.0'
    write(writeunit, '(A,I5)') 'Euler3D Step ', istep
    write(writeunit, '(A)') 'ASCII'
    write(writeunit, '(A)') 'DATASET STRUCTURED_GRID'
    write(writeunit, '(A,3I6)') 'DIMENSIONS ', nsub1+2, nsub2+2, 1
    write(writeunit, '(A,I10,A)') 'POINTS ', (nsub1+2)*(nsub2+2)*(1), ' double'
    
    do k = 1,1
        do j = 0, nsub2+1
            do i = 0, nsub1+1
                write(writeunit, '(3E16.8)') x(i), y(j), z(k)
            end do
        end do
    end do
    
    write(writeunit, '(A,I10)') 'POINT_DATA ', (nsub1+2)*(nsub2+2)*(1)
    
    write(writeunit, '(A)') 'SCALARS density double 1'
    write(writeunit, '(A)') 'LOOKUP_TABLE default'
    do k = 1,1
        do j = 0, nsub2+1
            do i = 0, nsub1+1
                write(writeunit, '(E16.8)') U(i,j,k,1)
            end do
        end do
    end do
    
    write(writeunit, '(A)') 'SCALARS pressure double 1'
    write(writeunit, '(A)') 'LOOKUP_TABLE default'
    do k = 1,1
        do j = 0, nsub2+1
            do i = 0, nsub1+1
                rho = U(i,j,k,1); uu = U(i,j,k,2)/rho; vv = U(i,j,k,3)/rho
                ww = U(i,j,k,4)/rho; rhoE = U(i,j,k,5)
                p = (gamma - 1.0d0) * (rhoE - 0.5d0 * rho * (uu**2 + vv**2 + ww**2))
                write(writeunit, '(E16.8)') p
                ! write(writeunit, '(E16.8)') U(i,j,k,2)
            end do
        end do
    end do
    
    write(writeunit, '(A)') 'VECTORS velocity double'
    do k = 1,1
        do j = 0, nsub2+1
            do i = 0, nsub1+1
                rho = U(i,j,k,1)
                write(writeunit, '(3E16.8)') U(i,j,k,2)/rho, U(i,j,k,3)/rho, U(i,j,k,4)/rho
                ! write(writeunit, '(3E16.8)') U(i,j,k,3), U(i,j,k,4), U(i,j,k,5)
            end do
        end do
    end do

    ! Vorticity z-component: omega_z = dv/dx - du/dy  (central diff; one-sided at halo edges)
    write(writeunit, '(A)') 'SCALARS vorticity_z double 1'
    write(writeunit, '(A)') 'LOOKUP_TABLE default'
    do k = 1,1
        do j = 0, nsub2+1
            if (j == 0) then
                jm = 0;       jp = 1
            else if (j == nsub2+1) then
                jm = nsub2;   jp = nsub2+1
            else
                jm = j-1;     jp = j+1
            end if
            do i = 0, nsub1+1
                if (i == 0) then
                    im = 0;       ip = 1
                else if (i == nsub1+1) then
                    im = nsub1;   ip = nsub1+1
                else
                    im = i-1;     ip = i+1
                end if
                dvdx = (U(ip,j,k,3)/U(ip,j,k,1) - U(im,j,k,3)/U(im,j,k,1)) / (x(ip) - x(im))
                dudy = (U(i,jp,k,2)/U(i,jp,k,1) - U(i,jm,k,2)/U(i,jm,k,1)) / (y(jp) - y(jm))
                omega_z = dvdx - dudy
                write(writeunit, '(E16.8)') omega_z
            end do
        end do
    end do

    close(writeunit)
    if(rank == 0) write(*,'(A,A)') ' Output: ', trim(filename)

    ! ----------------------------------------------------------------------
    ! 1D line along x at (y_center, z_center): plain-text columnar output.
    ! Nearest-cell index via nint (uniform grid) — mirrors the existing
    ! "j = n2/2 - globindx_ya + 1" pattern: valid only on the rank whose
    ! subdomain contains the target.  Columns: x  rho  p  u  v  w  omega_z
    ! ----------------------------------------------------------------------
    jc = 1
    kc = 1

    write(filename, '(A,I5.5,A,I5.5,A)') 'plot1d_x_yzcent_rank', rank, '_step', istep, '.txt'
    open(newunit=writeunit, file=trim(filename), status='replace')
    write(writeunit, '(A)') '# 1D line along x at (y_center, z_center)'
    write(writeunit, '(A,2I8)')          '# jc, kc (local)     = ', jc, kc
    write(writeunit, '(A,2E16.8)')       '# y(jc), z(kc)       = ', y(jc), z(kc)
    write(writeunit, '(A)') &
        '# x rho p u v w omega_z'
    j = jc; k = kc
    do i = 1, nsub1
        rho = U(i,j,k,1)
        uu  = U(i,j,k,2)/rho
        vv  = U(i,j,k,3)/rho
        ww  = U(i,j,k,4)/rho
        rhoE = U(i,j,k,5)
        p = (gamma - 1.0d0) * (rhoE - 0.5d0 * rho * (uu**2 + vv**2 + ww**2))
        ! vorticity_z = dv/dx - du/dy (central diff; one-sided at i=1, i=nsub1 if halo present)
        im = max(i-1, 0);       ip = min(i+1, nsub1+1)
        jm = max(j-1, 0);       jp = min(j+1, nsub2+1)
        dvdx = (U(ip,j,k,3)/U(ip,j,k,1) - U(im,j,k,3)/U(im,j,k,1)) / (x(ip) - x(im))
        dudy = (U(i,jp,k,2)/U(i,jp,k,1) - U(i,jm,k,2)/U(i,jm,k,1)) / (y(jp) - y(jm))
        omega_z = dvdx - dudy
        write(writeunit, '(7E22.12)') x(i), rho, p, uu, vv, ww, omega_z
    end do
    close(writeunit)
    if(rank == 0) write(*,'(A,A)') ' Output: ', trim(filename)
end subroutine output_solution_2D


subroutine write_centerline_csv(U, x, nsub1, nsub2, nsub3, m, rank, &
                                global_x_first, global_y_first, global_z_first, &
                                n2_global, n3_global, dx, dy, stage)
    implicit none
    integer, intent(in) :: nsub1, nsub2, nsub3, m, rank
    integer, intent(in) :: global_x_first, global_y_first, global_z_first
    integer, intent(in) :: n2_global, n3_global
    real(8), intent(in) :: U(0:nsub1+1,0:nsub2+1,0:nsub3+1,m)
    real(8), intent(in) :: x(0:nsub1+1), dx, dy
    character(len=*), intent(in) :: stage

    real(8), parameter :: gamma = 1.4d0
    character(len=128) :: filename
    integer :: i, j, k, global_i, writeunit
    real(8) :: rho, uu, vv, ww, pressure, dvdx, dudy, omega_z

    j = n2_global/2 - global_y_first + 1
    k = n3_global/2 - global_z_first + 1
    if (j < 1 .or. j > nsub2 .or. k < 1 .or. k > nsub3) return

    write(filename,'(A,A,A,I4.4,A)') 'centerline_', trim(stage), '_rank', rank, '.csv'
    open(newunit=writeunit, file=trim(filename), status='replace', action='write')
    write(writeunit,'(A)') 'global_i,x,rho,pressure,u,v,w,vorticity_z'

    do i = 1, nsub1
        global_i = global_x_first + i - 1
        rho = U(i,j,k,1)
        uu = U(i,j,k,2)/rho
        vv = U(i,j,k,3)/rho
        ww = U(i,j,k,4)/rho
        pressure = (gamma-1.0d0) * (U(i,j,k,5) - &
                   0.5d0*(U(i,j,k,2)**2 + U(i,j,k,3)**2 + U(i,j,k,4)**2)/rho)
        dvdx = (U(i+1,j,k,3)/U(i+1,j,k,1) - &
                U(i-1,j,k,3)/U(i-1,j,k,1))/(2.0d0*dx)
        dudy = (U(i,j+1,k,2)/U(i,j+1,k,1) - &
                U(i,j-1,k,2)/U(i,j-1,k,1))/(2.0d0*dy)
        omega_z = dvdx-dudy
        write(writeunit,'(I0,7(",",ES24.16E3))') global_i, x(i), rho, pressure, &
                                                        uu, vv, ww, omega_z
    end do
    close(writeunit)
end subroutine write_centerline_csv
