program throughput_gpu
    use mpi
    use cudafor
    use mod_btdma_gpu_v2, only: BTDMA_PLAN_gpu_v2, &
        btdma_makeplan_gpu_v2, btdma_many_mpi_gpu_v2, &
        btdma_cleanplan_gpu_v2, t0__a, t0__b, t1__a, t1__b, &
        t2__a, t2__b, t3__a, t3__b, t4__a, t4__b, t5__a, t5__b, &
        comm__tp__a, comm__tp__b, comm__tc__a, comm__tc__b, &
        comm__tu__a, comm__tu__b
    use throughput_common
    implicit none

    type(throughput_config) :: cfg
    type(BTDMA_PLAN_gpu_v2) :: plan
    integer :: ierr, myrank, nprocs, local_comm, local_rank
    integer :: device_count, device_id, cuda_status
    integer :: nlocal, iteration, repeat_index
    real(dp), device, allocatable :: a(:,:,:,:), b(:,:,:,:), c(:,:,:,:), d(:,:,:)
    real(dp) :: start_time, elapsed
    real(dp) :: timing(8)

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
    call read_config(cfg, "gpu", MPI_COMM_WORLD)
    call validate_mapping(cfg, "gpu", MPI_COMM_WORLD)

    call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, &
        MPI_INFO_NULL, local_comm, ierr)
    call MPI_Comm_rank(local_comm, local_rank, ierr)
    cuda_status = cudaGetDeviceCount(device_count)
    if (cuda_status /= 0 .or. device_count < 1) then
        call abort_gpu("no CUDA device is visible", MPI_COMM_WORLD)
    end if
    device_id = mod(local_rank, device_count)
    cuda_status = cudaSetDevice(device_id)
    if (cuda_status /= 0) then
        call abort_gpu("cudaSetDevice failed", MPI_COMM_WORLD)
    end if

    nlocal = local_extent(cfg%n_global, myrank, nprocs)
    allocate(a(cfg%nsys,nlocal,cfg%m,cfg%m))
    allocate(b(cfg%nsys,nlocal,cfg%m,cfg%m))
    allocate(c(cfg%nsys,nlocal,cfg%m,cfg%m))
    allocate(d(cfg%nsys,nlocal,cfg%m))
    call btdma_makeplan_gpu_v2(plan, cfg%m, cfg%nsys, nlocal, MPI_COMM_WORLD)

    call print_result_header(myrank)
    do iteration = 1, cfg%nwarmup + cfg%nrepeat
        call initialize_gpu_system(a, b, c, d, cfg%m, myrank, nprocs)
        call reset_gpu_timers()
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        call synchronize_gpu()
        start_time = MPI_Wtime()
        call btdma_many_mpi_gpu_v2(a, b, c, d, cfg%m, cfg%nsys, nlocal, plan)
        call synchronize_gpu()
        elapsed = MPI_Wtime() - start_time

        if (iteration > cfg%nwarmup) then
            repeat_index = iteration - cfg%nwarmup
            timing = (/ elapsed, t1__b + t3__b + t5__b, t2__b + t4__b, &
                t1__b, t2__b, t3__b, t4__b, t5__b /)
            call reduce_and_print(cfg, "gpu", repeat_index, timing, MPI_COMM_WORLD)
        end if
    end do

    call btdma_cleanplan_gpu_v2(plan)
    deallocate(a, b, c, d)
    call MPI_Comm_free(local_comm, ierr)
    call MPI_Finalize(ierr)

contains

    subroutine initialize_gpu_system(aa, bb, cc, dd, block_size, rank, ranks)
        real(dp), device, intent(out) :: aa(:,:,:,:), bb(:,:,:,:), cc(:,:,:,:)
        real(dp), device, intent(out) :: dd(:,:,:)
        integer, intent(in) :: block_size, rank, ranks
        integer :: component

        aa = 0.0_dp
        bb = 0.0_dp
        cc = 0.0_dp
        dd = 1.0_dp
        do component = 1, block_size
            aa(:,:,component,component) = -0.25_dp
            bb(:,:,component,component) =  2.00_dp
            cc(:,:,component,component) = -0.25_dp
        end do
        if (rank == 0) aa(:,1,:,:) = 0.0_dp
        if (rank == ranks - 1) cc(:,size(cc,2),:,:) = 0.0_dp
        call synchronize_gpu()
    end subroutine initialize_gpu_system

    subroutine reset_gpu_timers()
        t0__a=0.0_dp; t0__b=0.0_dp; t1__a=0.0_dp; t1__b=0.0_dp
        t2__a=0.0_dp; t2__b=0.0_dp; t3__a=0.0_dp; t3__b=0.0_dp
        t4__a=0.0_dp; t4__b=0.0_dp; t5__a=0.0_dp; t5__b=0.0_dp
        comm__tp__a=0.0_dp; comm__tp__b=0.0_dp
        comm__tc__a=0.0_dp; comm__tc__b=0.0_dp
        comm__tu__a=0.0_dp; comm__tu__b=0.0_dp
    end subroutine reset_gpu_timers

    subroutine synchronize_gpu()
        integer :: status

        status = cudaDeviceSynchronize()
        if (status /= 0) call abort_gpu("cudaDeviceSynchronize failed", MPI_COMM_WORLD)
    end subroutine synchronize_gpu

    subroutine abort_gpu(message, comm)
        character(len=*), intent(in) :: message
        integer, intent(in) :: comm
        integer :: error_code, rank

        call MPI_Comm_rank(comm, rank, error_code)
        if (rank == 0) write(*,'(A)') "ERROR: " // trim(message)
        call MPI_Abort(comm, 3, error_code)
    end subroutine abort_gpu

end program throughput_gpu
