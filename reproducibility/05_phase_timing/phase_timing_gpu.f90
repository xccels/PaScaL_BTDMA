program phase_timing_gpu
    use mpi
    use cudafor
    use mod_btdma_gpu_v2
    implicit none

    integer, parameter :: m = 8
    integer, parameter :: n_global = 2048
    integer, parameter :: nsys = 128 * 128
    integer, parameter :: default_warmups = 1
    integer, parameter :: default_repeats = 5

    integer :: ierr, myrank, nprocs
    integer :: local_comm, local_rank, local_size
    integer :: device_count, device_id
    integer :: nlocal, nlocal_min, nlocal_max
    integer :: warmups, repeats, run
    integer :: cuda_stat
    real(8) :: time_start, total_local
    real(8) :: local_times(6), maximum_times(6)
    real(8) :: phase_sum, step3_fraction, phase_sum_fraction
    real(8) :: sample_error_local, sample_error_max
    real(8), device, allocatable :: a(:,:,:,:), b(:,:,:,:), c(:,:,:,:)
    real(8), device, allocatable :: x(:,:,:)
    type(BTDMA_PLAN_gpu_v2) :: plan

    call MPI_Init(ierr)
    call check_mpi(ierr, 'MPI_Init')
    call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
    call check_mpi(ierr, 'MPI_Comm_rank')
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
    call check_mpi(ierr, 'MPI_Comm_size')

    call read_positive_argument(1, default_warmups, warmups)
    call read_positive_argument(2, default_repeats, repeats)
    if (warmups < 0) call abort_run('WARMUPS must be zero or greater.')
    if (repeats < 1) call abort_run('REPEATS must be one or greater.')
    if (nprocs /= 2 .and. nprocs /= 4 .and. nprocs /= 8) then
        call abort_run('This fixed experiment accepts P=2, 4, or 8 only.')
    end if
    if (mod(n_global, nprocs) /= 0) then
        call abort_run('N=2048 must be divisible by the MPI process count.')
    end if

    call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, local_comm, ierr)
    call check_mpi(ierr, 'MPI_Comm_split_type')
    call MPI_Comm_rank(local_comm, local_rank, ierr)
    call check_mpi(ierr, 'MPI_Comm_rank(local)')
    call MPI_Comm_size(local_comm, local_size, ierr)
    call check_mpi(ierr, 'MPI_Comm_size(local)')

    cuda_stat = cudaGetDeviceCount(device_count)
    call check_cuda(cuda_stat, 'cudaGetDeviceCount')
    if (device_count < 1) call abort_run('No CUDA device is visible to this rank.')
    device_id = mod(local_rank, device_count)
    cuda_stat = cudaSetDevice(device_id)
    call check_cuda(cuda_stat, 'cudaSetDevice')

    nlocal = n_global / nprocs
    call MPI_Allreduce(nlocal, nlocal_min, 1, MPI_INTEGER, MPI_MIN, MPI_COMM_WORLD, ierr)
    call check_mpi(ierr, 'MPI_Allreduce(nlocal_min)')
    call MPI_Allreduce(nlocal, nlocal_max, 1, MPI_INTEGER, MPI_MAX, MPI_COMM_WORLD, ierr)
    call check_mpi(ierr, 'MPI_Allreduce(nlocal_max)')

    allocate(a(nsys,nlocal,m,m), b(nsys,nlocal,m,m), c(nsys,nlocal,m,m))
    allocate(x(nsys,nlocal,m))
    call btdma_makeplan_gpu_v2(plan, m, nsys, nlocal, MPI_COMM_WORLD)

    if (myrank == 0) then
        write(*,'(a)') 'PHASE_METADATA,backend=optimized_gpu_pascal_btdma,precision=fp64'
        write(*,'(a,i0,a,i0,a,i0,a,i0,a,i0)') &
            'PHASE_METADATA,m=', m, ',N=', n_global, ',nsys=', nsys, &
            ',warmups=', warmups, ',repeats=', repeats
        write(*,'(a,i0,a,i0,a,i0,a,i0)') &
            'PHASE_METADATA,p_total=', nprocs, ',p_dir=', nprocs, &
            ',local_ranks=', local_size, ',reduced_rows=', 2*nprocs
        write(*,'(a)') 'PHASE_SCHEMA,repeat,p_total,p_dir,reduced_rows,m,N,nsys,' // &
            'nlocal_min,nlocal_max,step1_s,step2_s,step3_s,step4_s,step5_s,' // &
            'total_s,step3_fraction,phase_sum_s,phase_sum_over_total'
    end if

    do run = 1, warmups
        call initialize_system(a, b, c, x, myrank, nprocs, nlocal)
        call reset_phase_timers()
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        call check_mpi(ierr, 'MPI_Barrier(warmup)')
        call btdma_many_mpi_gpu_v2(a, b, c, x, m, nsys, nlocal, plan)
        cuda_stat = cudaDeviceSynchronize()
        call check_cuda(cuda_stat, 'cudaDeviceSynchronize(warmup)')
    end do

    do run = 1, repeats
        call initialize_system(a, b, c, x, myrank, nprocs, nlocal)
        call reset_phase_timers()

        cuda_stat = cudaDeviceSynchronize()
        call check_cuda(cuda_stat, 'cudaDeviceSynchronize(before total)')
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        call check_mpi(ierr, 'MPI_Barrier(measured)')
        time_start = MPI_Wtime()

        call btdma_many_mpi_gpu_v2(a, b, c, x, m, nsys, nlocal, plan)

        cuda_stat = cudaDeviceSynchronize()
        call check_cuda(cuda_stat, 'cudaDeviceSynchronize(after total)')
        total_local = MPI_Wtime() - time_start

        local_times = [t1__b, t2__b, t3__b, t4__b, t5__b, total_local]
        call MPI_Reduce(local_times, maximum_times, 6, MPI_DOUBLE_PRECISION, MPI_MAX, 0, MPI_COMM_WORLD, ierr)
        call check_mpi(ierr, 'MPI_Reduce(phase maxima)')

        if (myrank == 0) then
            phase_sum = sum(maximum_times(1:5))
            step3_fraction = maximum_times(3) / maximum_times(6)
            phase_sum_fraction = phase_sum / maximum_times(6)
            write(*,'(a,9(i0,a),9(es24.16e3,:,a))') &
                'PHASE_RAW,', run, ',', nprocs, ',', nprocs, ',', 2*nprocs, ',', &
                m, ',', n_global, ',', nsys, ',', nlocal_min, ',', nlocal_max, ',', &
                maximum_times(1), ',', maximum_times(2), ',', maximum_times(3), ',', &
                maximum_times(4), ',', maximum_times(5), ',', maximum_times(6), ',', &
                step3_fraction, ',', phase_sum, ',', phase_sum_fraction
        end if
    end do

    sample_error_local = abs(x(1,1,1) - 1.0d0)
    call MPI_Reduce(sample_error_local, sample_error_max, 1, MPI_DOUBLE_PRECISION, &
                    MPI_MAX, 0, MPI_COMM_WORLD, ierr)
    call check_mpi(ierr, 'MPI_Reduce(sample error)')
    if (myrank == 0) then
        write(*,'(a,es24.16e3)') 'PHASE_CHECK,max_abs_sample_error=', sample_error_max
    end if

    call btdma_cleanplan_gpu_v2(plan)
    deallocate(a, b, c, x)
    call MPI_Comm_free(local_comm, ierr)
    call check_mpi(ierr, 'MPI_Comm_free')
    call MPI_Finalize(ierr)

contains

    subroutine initialize_system(a_d, b_d, c_d, x_d, rank, size, rows)
        real(8), device, intent(out) :: a_d(nsys,rows,m,m)
        real(8), device, intent(out) :: b_d(nsys,rows,m,m)
        real(8), device, intent(out) :: c_d(nsys,rows,m,m)
        real(8), device, intent(out) :: x_d(nsys,rows,m)
        integer, intent(in) :: rank, size, rows
        integer :: component, stat

        a_d = 0.0d0
        b_d = 0.0d0
        c_d = 0.0d0
        x_d = 1.5d0
        do component = 1, m
            a_d(:,:,component,component) = -0.25d0
            b_d(:,:,component,component) =  2.00d0
            c_d(:,:,component,component) = -0.25d0
        end do
        if (rank == 0) then
            a_d(:,1,:,:) = 0.0d0
            x_d(:,1,:) = 1.75d0
        end if
        if (rank == size - 1) then
            c_d(:,rows,:,:) = 0.0d0
            x_d(:,rows,:) = 1.75d0
        end if
        stat = cudaDeviceSynchronize()
        call check_cuda(stat, 'cudaDeviceSynchronize(initialization)')
    end subroutine initialize_system

    subroutine reset_phase_timers()
        t0__a = 0.0d0
        t0__b = 0.0d0
        t1__a = 0.0d0
        t1__b = 0.0d0
        t2__a = 0.0d0
        t2__b = 0.0d0
        t3__a = 0.0d0
        t3__b = 0.0d0
        t4__a = 0.0d0
        t4__b = 0.0d0
        t5__a = 0.0d0
        t5__b = 0.0d0
    end subroutine reset_phase_timers

    subroutine read_positive_argument(position, default_value, value)
        integer, intent(in) :: position, default_value
        integer, intent(out) :: value
        integer :: status
        character(len=64) :: text

        value = default_value
        if (command_argument_count() < position) return
        call get_command_argument(position, text, status=status)
        if (status /= 0) call abort_run('Could not read a command-line argument.')
        read(text, *, iostat=status) value
        if (status /= 0) call abort_run('Command-line arguments must be integers.')
    end subroutine read_positive_argument

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
        integer :: abort_error
        write(*,'(a,i0,a,a)') 'ERROR rank ', myrank, ': ', trim(message)
        call MPI_Abort(MPI_COMM_WORLD, 1, abort_error)
    end subroutine abort_run

end program phase_timing_gpu
