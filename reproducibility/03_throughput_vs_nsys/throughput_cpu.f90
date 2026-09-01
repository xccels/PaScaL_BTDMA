program throughput_cpu
    use mpi
    use mpiutil, only: t0_a, t0_b, t1_a, t1_b, t2_a, t2_b, &
        t3_a, t3_b, t4_a, t4_b, t5_a, t5_b, comm_tp_a, comm_tp_b, &
        comm_tc_a, comm_tc_b, comm_tu_a, comm_tu_b
    use mod_btdma_cpu
    use throughput_common
    implicit none

    type(throughput_config) :: cfg
    type(BTDMA_PLAN) :: plan
    integer :: ierr, myrank, nprocs
    integer :: nlocal, iteration, repeat_index
    real(dp), allocatable :: a(:,:,:,:), b(:,:,:,:), c(:,:,:,:), d(:,:,:)
    real(dp) :: start_time, elapsed
    real(dp) :: timing(8)

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
    call read_config(cfg, "cpu", MPI_COMM_WORLD)
    call validate_mapping(cfg, "cpu", MPI_COMM_WORLD)

    nlocal = local_extent(cfg%n_global, myrank, nprocs)
    allocate(a(cfg%m,cfg%m,cfg%nsys,nlocal))
    allocate(b(cfg%m,cfg%m,cfg%nsys,nlocal))
    allocate(c(cfg%m,cfg%m,cfg%nsys,nlocal))
    allocate(d(cfg%m,cfg%nsys,nlocal))
    call btdma_makeplan(plan, cfg%m, cfg%nsys, nlocal, MPI_COMM_WORLD)

    call print_result_header(myrank)
    do iteration = 1, cfg%nwarmup + cfg%nrepeat
        call initialize_cpu_system(a, b, c, d, cfg%m, myrank, nprocs)
        call reset_cpu_timers()
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        start_time = MPI_Wtime()
        call btdma_many_mpi(a, b, c, d, cfg%m, cfg%nsys, nlocal, plan)
        elapsed = MPI_Wtime() - start_time

        if (iteration > cfg%nwarmup) then
            repeat_index = iteration - cfg%nwarmup
            timing = (/ elapsed, t1_b + t3_b + t5_b, t2_b + t4_b, &
                t1_b, t2_b, t3_b, t4_b, t5_b /)
            call reduce_and_print(cfg, "cpu", repeat_index, timing, MPI_COMM_WORLD)
        end if
    end do

    call btdma_cleanplan(plan)
    deallocate(a, b, c, d)
    call MPI_Finalize(ierr)

contains

    subroutine reset_cpu_timers()
        t0_a=0.0_dp; t0_b=0.0_dp; t1_a=0.0_dp; t1_b=0.0_dp
        t2_a=0.0_dp; t2_b=0.0_dp; t3_a=0.0_dp; t3_b=0.0_dp
        t4_a=0.0_dp; t4_b=0.0_dp; t5_a=0.0_dp; t5_b=0.0_dp
        comm_tp_a=0.0_dp; comm_tp_b=0.0_dp
        comm_tc_a=0.0_dp; comm_tc_b=0.0_dp
        comm_tu_a=0.0_dp; comm_tu_b=0.0_dp
    end subroutine reset_cpu_timers

end program throughput_cpu
