program bench_gpu_pascal_optimized
    use mpi
    use cudafor
    use mod_btdma_gpu_v2, only: BTDMA_PLAN_gpu_v2, btdma_makeplan_gpu_v2, &
        btdma_many_mpi_gpu_v2, btdma_cleanplan_gpu_v2, &
        t0__a, t0__b, t1__a, t1__b, t2__a, t2__b, t3__a, t3__b, &
        t4__a, t4__b, t5__a, t5__b, comm__tp__a, comm__tp__b, &
        comm__tc__a, comm__tc__b, comm__tu__a, comm__tu__b
    use benchmark_support
    use gpu_benchmark_support
    implicit none

    type(benchmark_config) :: cfg
    type(BTDMA_PLAN_gpu_v2) :: plan
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
    allocate(a(cfg%nsys,nlocal,cfg%m,cfg%m), b(cfg%nsys,nlocal,cfg%m,cfg%m))
    allocate(c(cfg%nsys,nlocal,cfg%m,cfg%m), d(cfg%nsys,nlocal,cfg%m))
    call btdma_makeplan_gpu_v2(plan, cfg%m, cfg%nsys, nlocal, MPI_COMM_WORLD)

    call print_result_header(myrank)
    do iteration = 1, cfg%nwarmup + cfg%nrepeat
        call initialize_gpu_v2(a, b, c, d, cfg%m, myrank, nprocs, MPI_COMM_WORLD)
        call reset_v2_timers()
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        call synchronize_gpu(MPI_COMM_WORLD)
        start_time = MPI_Wtime()
        call btdma_many_mpi_gpu_v2(a, b, c, d, cfg%m, cfg%nsys, nlocal, plan)
        call synchronize_gpu(MPI_COMM_WORLD)
        elapsed = MPI_Wtime() - start_time

        if (iteration > cfg%nwarmup) then
            repeat_index = iteration - cfg%nwarmup
            timing = (/ elapsed, t1__b + t3__b + t5__b, t2__b + t4__b, &
                t1__b, t2__b, t3__b, t4__b, t5__b /)
            call reduce_and_print(cfg, "pascal_optimized", "gpu", &
                repeat_index, timing, MPI_COMM_WORLD)
        end if
    end do

    call btdma_cleanplan_gpu_v2(plan)
    deallocate(a, b, c, d)
    call MPI_Finalize(ierr)

contains

    subroutine reset_v2_timers()
        t0__a=0.0_dp; t0__b=0.0_dp; t1__a=0.0_dp; t1__b=0.0_dp
        t2__a=0.0_dp; t2__b=0.0_dp; t3__a=0.0_dp; t3__b=0.0_dp
        t4__a=0.0_dp; t4__b=0.0_dp; t5__a=0.0_dp; t5__b=0.0_dp
        comm__tp__a=0.0_dp; comm__tp__b=0.0_dp
        comm__tc__a=0.0_dp; comm__tc__b=0.0_dp
        comm__tu__a=0.0_dp; comm__tu__b=0.0_dp
    end subroutine reset_v2_timers

end program bench_gpu_pascal_optimized
