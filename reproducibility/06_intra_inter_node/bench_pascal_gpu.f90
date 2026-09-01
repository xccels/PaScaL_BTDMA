program bench_pascal_gpu
    use mpi
    use cudafor
    use mod_btdma_gpu_v2
    use node_compare_support
    implicit none

    character(len=16) :: regime
    integer :: ierr, myrank, nprocs, local_comm, local_rank, local_size
    integer :: warmups, repeats, run, nlocal
    real(dp), device, allocatable :: a(:,:,:,:), b(:,:,:,:), c(:,:,:,:), d(:,:,:)
    real(dp) :: wall_start, wall_elapsed, phases(5)
    type(BTDMA_PLAN_gpu_v2) :: plan

    call MPI_Init(ierr)
    call check_mpi(ierr, 'MPI_Init')
    call configure_run(regime, warmups, repeats, myrank, nprocs, &
        local_comm, local_rank, local_size)
    nlocal = n_global / nprocs

    allocate(a(n_systems,nlocal,block_size,block_size))
    allocate(b(n_systems,nlocal,block_size,block_size))
    allocate(c(n_systems,nlocal,block_size,block_size))
    allocate(d(n_systems,nlocal,block_size))
    call btdma_makeplan_gpu_v2(plan, block_size, n_systems, nlocal, MPI_COMM_WORLD)

    if (myrank == 0) then
        write(*,'(a)') 'NODE_SCHEMA,method,regime,repeat,nodes,gpus_per_node,mpi_ranks,' // &
            'local_ranks,m,N,nsys,Nlocal,phase1_s,phase2_s,phase3_s,phase4_s,' // &
            'phase5_s,computation_s,communication_s,total_s,wall_s,timing_gap_s'
    end if

    do run = 1, warmups
        call initialize_v2_system(a, b, c, d)
        call reset_timers()
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        call check_mpi(ierr, 'MPI_Barrier(warmup)')
        call btdma_many_mpi_gpu_v2(a, b, c, d, block_size, n_systems, nlocal, plan)
        call synchronize_gpu('cudaDeviceSynchronize(warmup)')
    end do

    do run = 1, repeats
        call initialize_v2_system(a, b, c, d)
        call reset_timers()
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        call check_mpi(ierr, 'MPI_Barrier(measured)')
        call synchronize_gpu('cudaDeviceSynchronize(before total)')
        wall_start = MPI_Wtime()

        call btdma_many_mpi_gpu_v2(a, b, c, d, block_size, n_systems, nlocal, plan)

        call synchronize_gpu('cudaDeviceSynchronize(after total)')
        wall_elapsed = MPI_Wtime() - wall_start
        phases = [t1__b, t2__b, t3__b, t4__b, t5__b]
        call report_repeat('pascal_btdma', regime, run, phases, wall_elapsed, &
            local_size, MPI_COMM_WORLD)
    end do

    call btdma_cleanplan_gpu_v2(plan)
    deallocate(a, b, c, d)
    call MPI_Comm_free(local_comm, ierr)
    call check_mpi(ierr, 'MPI_Comm_free')
    call MPI_Finalize(ierr)

contains

    subroutine reset_timers()
        t0__a = 0.0_dp; t0__b = 0.0_dp
        t1__a = 0.0_dp; t1__b = 0.0_dp
        t2__a = 0.0_dp; t2__b = 0.0_dp
        t3__a = 0.0_dp; t3__b = 0.0_dp
        t4__a = 0.0_dp; t4__b = 0.0_dp
        t5__a = 0.0_dp; t5__b = 0.0_dp
        comm__tp__a = 0.0_dp; comm__tp__b = 0.0_dp
        comm__tc__a = 0.0_dp; comm__tc__b = 0.0_dp
        comm__tu__a = 0.0_dp; comm__tu__b = 0.0_dp
    end subroutine reset_timers

end program bench_pascal_gpu
