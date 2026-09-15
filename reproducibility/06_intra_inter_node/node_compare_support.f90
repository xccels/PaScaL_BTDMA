module node_compare_support
    use iso_fortran_env, only: real64
    use mpi
    use cudafor
    implicit none

    private
    public :: dp, block_size, n_global, n_systems
    public :: configure_run, initialize_legacy_system, initialize_v2_system
    public :: report_repeat, synchronize_gpu, check_mpi

    integer, parameter :: dp = real64
    integer, parameter :: block_size = 8
    integer, parameter :: n_global = 2048
    integer, parameter :: n_systems = 128 * 128
    integer, parameter :: required_ranks = 4

contains

    subroutine configure_run(regime, warmups, repeats, myrank, nprocs, &
            local_comm, local_rank, local_size)
        character(len=*), intent(out) :: regime
        integer, intent(out) :: warmups, repeats, myrank, nprocs
        integer, intent(out) :: local_comm, local_rank, local_size
        character(len=64) :: argument
        integer :: ierr, local_size_min, local_size_max
        integer :: device_count, device_id, cuda_status

        call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
        call check_mpi(ierr, 'MPI_Comm_rank')
        call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
        call check_mpi(ierr, 'MPI_Comm_size')

        if (command_argument_count() /= 3) then
            call abort_run('Usage: <executable> <intra-node|inter-node> <warmups> <repeats>')
        end if
        call get_command_argument(1, argument)
        regime = trim(adjustl(argument))
        call read_integer_argument(2, warmups)
        call read_integer_argument(3, repeats)

        if (trim(regime) /= 'intra-node' .and. trim(regime) /= 'inter-node') then
            call abort_run('The regime must be intra-node or inter-node.')
        end if
        if (warmups < 0) call abort_run('The number of warmups must be nonnegative.')
        if (repeats < 1) call abort_run('The number of repeats must be positive.')
        if (nprocs /= required_ranks) then
            call abort_run('This fixed comparison requires exactly four MPI processes.')
        end if
        if (mod(n_global, nprocs) /= 0 .or. mod(n_systems, nprocs) /= 0) then
            call abort_run('N and nsys must be divisible by the process count.')
        end if

        call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, &
            MPI_INFO_NULL, local_comm, ierr)
        call check_mpi(ierr, 'MPI_Comm_split_type')
        call MPI_Comm_rank(local_comm, local_rank, ierr)
        call check_mpi(ierr, 'MPI_Comm_rank(local)')
        call MPI_Comm_size(local_comm, local_size, ierr)
        call check_mpi(ierr, 'MPI_Comm_size(local)')
        call MPI_Allreduce(local_size, local_size_min, 1, MPI_INTEGER, MPI_MIN, &
            MPI_COMM_WORLD, ierr)
        call check_mpi(ierr, 'MPI_Allreduce(local_size_min)')
        call MPI_Allreduce(local_size, local_size_max, 1, MPI_INTEGER, MPI_MAX, &
            MPI_COMM_WORLD, ierr)
        call check_mpi(ierr, 'MPI_Allreduce(local_size_max)')

        if (trim(regime) == 'intra-node') then
            if (local_size_min /= 4 .or. local_size_max /= 4) then
                call abort_run('The intra-node case requires four ranks on one shared-memory node.')
            end if
        else
            if (local_size_min /= 1 .or. local_size_max /= 1) then
                call abort_run('The inter-node case requires one rank on each of four nodes.')
            end if
        end if

        cuda_status = cudaGetDeviceCount(device_count)
        call check_cuda(cuda_status, 'cudaGetDeviceCount')
        if (device_count < 1) call abort_run('No CUDA device is visible to this rank.')
        device_id = mod(local_rank, device_count)
        cuda_status = cudaSetDevice(device_id)
        call check_cuda(cuda_status, 'cudaSetDevice')
        cuda_status = cudaDeviceSynchronize()
        call check_cuda(cuda_status, 'cudaDeviceSynchronize(after device selection)')
    end subroutine configure_run

    subroutine initialize_legacy_system(a, b, c, d)
        real(dp), device, intent(out) :: a(:,:,:,:), b(:,:,:,:), c(:,:,:,:), d(:,:,:)
        integer :: cuda_status, i

        a = 0.0_dp
        b = 0.0_dp
        c = 0.0_dp
        d = 1.0_dp
        do i = 1, block_size
            a(i,i,:,:) = -0.25_dp
            b(i,i,:,:) =  2.50_dp
            c(i,i,:,:) = -0.25_dp
        end do
        cuda_status = cudaDeviceSynchronize()
        call check_cuda(cuda_status, 'cudaDeviceSynchronize(legacy initialization)')
    end subroutine initialize_legacy_system

    subroutine initialize_v2_system(a, b, c, d)
        real(dp), device, intent(out) :: a(:,:,:,:), b(:,:,:,:), c(:,:,:,:), d(:,:,:)
        integer :: cuda_status, i

        a = 0.0_dp
        b = 0.0_dp
        c = 0.0_dp
        d = 1.0_dp
        do i = 1, block_size
            a(:,:,i,i) = -0.25_dp
            b(:,:,i,i) =  2.50_dp
            c(:,:,i,i) = -0.25_dp
        end do
        cuda_status = cudaDeviceSynchronize()
        call check_cuda(cuda_status, 'cudaDeviceSynchronize(v2 initialization)')
    end subroutine initialize_v2_system

    subroutine report_repeat(method, regime, repeat_index, local_phases, &
            local_wall, local_size, comm)
        character(len=*), intent(in) :: method, regime
        integer, intent(in) :: repeat_index, local_size, comm
        real(dp), intent(in) :: local_phases(5), local_wall
        real(dp) :: representative_phases(5), maximum_wall
        real(dp) :: computation, communication, total, timing_gap
        real(dp) :: local_computation, local_communication, local_total
        real(dp) :: local_pair(2), maximum_pair(2)
        integer :: ierr, myrank, nprocs, nodes, gpus_per_node, critical_rank

        call MPI_Comm_rank(comm, myrank, ierr)
        call check_mpi(ierr, 'MPI_Comm_rank(report)')
        call MPI_Comm_size(comm, nprocs, ierr)
        call check_mpi(ierr, 'MPI_Comm_size(report)')

        if (trim(method) == 'conventional_alltoall') then
            local_computation = local_phases(3)
            local_communication = local_phases(2) + local_phases(4)
        else if (trim(method) == 'pascal_btdma') then
            local_computation = local_phases(1) + local_phases(3) + local_phases(5)
            local_communication = local_phases(2) + local_phases(4)
        else
            call abort_run('Unknown method passed to report_repeat.')
        end if
        local_total = local_computation + local_communication
        local_pair = [local_total, real(myrank, dp)]
        call MPI_Allreduce(local_pair, maximum_pair, 1, MPI_2DOUBLE_PRECISION, &
            MPI_MAXLOC, comm, ierr)
        call check_mpi(ierr, 'MPI_Allreduce(critical rank)')
        critical_rank = nint(maximum_pair(2))
        representative_phases = local_phases
        call MPI_Bcast(representative_phases, 5, MPI_DOUBLE_PRECISION, &
            critical_rank, comm, ierr)
        call check_mpi(ierr, 'MPI_Bcast(critical-rank phases)')
        call MPI_Reduce(local_wall, maximum_wall, 1, MPI_DOUBLE_PRECISION, &
            MPI_MAX, 0, comm, ierr)
        call check_mpi(ierr, 'MPI_Reduce(wall maximum)')

        if (myrank /= 0) return
        if (trim(regime) == 'intra-node') then
            nodes = 1
            gpus_per_node = 4
        else
            nodes = 4
            gpus_per_node = 1
        end if

        if (trim(method) == 'conventional_alltoall') then
            computation = representative_phases(3)
            communication = representative_phases(2) + representative_phases(4)
        else
            computation = representative_phases(1) + representative_phases(3) &
                + representative_phases(5)
            communication = representative_phases(2) + representative_phases(4)
        end if
        total = computation + communication
        timing_gap = maximum_wall - total

        write(*,'(*(g0,:,","))') 'NODE_RAW', trim(method), trim(regime), &
            repeat_index, nodes, gpus_per_node, nprocs, local_size, block_size, &
            n_global, n_systems, n_global/nprocs, representative_phases, computation, &
            communication, total, maximum_wall, timing_gap
    end subroutine report_repeat

    subroutine synchronize_gpu(operation)
        character(len=*), intent(in) :: operation
        integer :: cuda_status
        cuda_status = cudaDeviceSynchronize()
        call check_cuda(cuda_status, operation)
    end subroutine synchronize_gpu

    subroutine read_integer_argument(position, value)
        integer, intent(in) :: position
        integer, intent(out) :: value
        character(len=64) :: argument
        integer :: status

        call get_command_argument(position, argument, status=status)
        if (status /= 0) call abort_run('Could not read an integer argument.')
        read(argument, *, iostat=status) value
        if (status /= 0) call abort_run('Warmups and repeats must be integers.')
    end subroutine read_integer_argument

    subroutine check_cuda(status, operation)
        integer, intent(in) :: status
        character(len=*), intent(in) :: operation
        if (status /= cudaSuccess) call abort_run(trim(operation) // ' failed.')
    end subroutine check_cuda

    subroutine check_mpi(status, operation)
        integer, intent(in) :: status
        character(len=*), intent(in) :: operation
        if (status /= MPI_SUCCESS) call abort_run(trim(operation) // ' failed.')
    end subroutine check_mpi

    subroutine abort_run(message)
        character(len=*), intent(in) :: message
        integer :: ierr, rank

        call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
        write(*,'(a,i0,a,a)') 'ERROR rank ', rank, ': ', trim(message)
        call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end subroutine abort_run

end module node_compare_support
