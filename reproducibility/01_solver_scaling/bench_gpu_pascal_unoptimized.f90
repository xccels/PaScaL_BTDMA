program bench_gpu_pascal_unoptimized
    use mpi
    use cudafor
    use mpiutil
    use mod_btdma_gpu, only: BTDMA_PLAN_gpu, btdma_makeplan_gpu, &
        btdma_many_mpi_gpu, btdma_cleanplan_gpu
    use benchmark_support
    use gpu_benchmark_support
    implicit none

    type(benchmark_config) :: cfg
    type(BTDMA_PLAN_gpu) :: plan
    integer :: ierr, myrank, nprocs
    integer :: nlocal, iteration, repeat_index
    real(dp), device, allocatable :: a(:,:,:,:), b(:,:,:,:), c(:,:,:,:), d(:,:,:)
    real(dp) :: start_time, elapsed
    real(dp) :: timing(8)

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
    call read_config(cfg, MPI_COMM_WORLD)
    call validate_mapping(cfg, 1, "GPU", MPI_COMM_WORLD)
    call select_local_gpu(MPI_COMM_WORLD)

    nlocal = local_extent(cfg%n_global, myrank, nprocs)
    allocate(a(cfg%m,cfg%m,cfg%nsys,nlocal), b(cfg%m,cfg%m,cfg%nsys,nlocal))
    allocate(c(cfg%m,cfg%m,cfg%nsys,nlocal), d(cfg%m,cfg%nsys,nlocal))
    call btdma_makeplan_gpu(plan, cfg%m, cfg%nsys, nlocal, MPI_COMM_WORLD)

    call print_result_header(myrank)
    do iteration = 1, cfg%nwarmup + cfg%nrepeat
        call initialize_gpu_legacy(a, b, c, d, cfg%m, myrank, nprocs, MPI_COMM_WORLD)
        call reset_mpiutil_timers()
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        call synchronize_gpu(MPI_COMM_WORLD)
        start_time = MPI_Wtime()
        call btdma_many_mpi_gpu(a, b, c, d, cfg%m, cfg%nsys, nlocal, plan)
        call synchronize_gpu(MPI_COMM_WORLD)
        elapsed = MPI_Wtime() - start_time

        if (iteration > cfg%nwarmup) then
            repeat_index = iteration - cfg%nwarmup
            timing = (/ elapsed, t1_b + t3_b + t5_b, t2_b + t4_b, &
                t1_b, t2_b, t3_b, t4_b, t5_b /)
            call reduce_and_print(cfg, "pascal_unoptimized", "gpu", &
                repeat_index, timing, MPI_COMM_WORLD)
        end if
    end do

    call btdma_cleanplan_gpu(plan)
    deallocate(a, b, c, d)
    call MPI_Finalize(ierr)

contains

    subroutine reset_mpiutil_timers()
        t0_a=0.0_dp; t0_b=0.0_dp; t1_a=0.0_dp; t1_b=0.0_dp
        t2_a=0.0_dp; t2_b=0.0_dp; t3_a=0.0_dp; t3_b=0.0_dp
        t4_a=0.0_dp; t4_b=0.0_dp; t5_a=0.0_dp; t5_b=0.0_dp
        comm_tp_a=0.0_dp; comm_tp_b=0.0_dp
        comm_tc_a=0.0_dp; comm_tc_b=0.0_dp
        comm_tu_a=0.0_dp; comm_tu_b=0.0_dp
    end subroutine reset_mpiutil_timers

end program bench_gpu_pascal_unoptimized
