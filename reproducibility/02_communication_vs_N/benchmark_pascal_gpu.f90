program benchmark_pascal_gpu
    use mpi
    use cudafor
    use mod_btdma_gpu_v2
    use communication_benchmark_common
    implicit none

    integer :: ierr, rank, nprocs, cuda_status
    integer :: n_global, m, warmups, repeats, p_label
    integer :: nrow_sub, iteration, diagonal_index
    real(kind=8), device, allocatable :: a(:,:,:,:), b(:,:,:,:), c(:,:,:,:), d(:,:,:)
    type(BTDMA_PLAN_gpu_v2) :: plan

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
    call parse_benchmark_args(n_global, m, warmups, repeats, p_label, MPI_COMM_WORLD)
    if (nprocs /= p_label) then
        call abort_benchmark('GPU P=8 requires one MPI rank per GPU and 8 ranks in total', &
                             MPI_COMM_WORLD)
    end if
    call select_local_gpu(rank)

    nrow_sub = n_global / nprocs
    if (rank < mod(n_global, nprocs)) nrow_sub = nrow_sub + 1
    call require_minimum_local_rows(nrow_sub, MPI_COMM_WORLD)

    allocate(a(nsys_fixed, nrow_sub, m, m))
    allocate(b(nsys_fixed, nrow_sub, m, m))
    allocate(c(nsys_fixed, nrow_sub, m, m))
    allocate(d(nsys_fixed, nrow_sub, m))
    call btdma_makeplan_gpu_v2(plan, m, nsys_fixed, nrow_sub, MPI_COMM_WORLD)

    call print_result_header(rank, &
        'PaScaL mod_btdma_gpu_v2 t2+t4; Step2 exchanges reduced A+B+C+RHS; ' // &
        'Step4 exchanges reduced solution only')

    do iteration = 1, warmups + repeats
        a = 0.0d0
        b = 0.0d0
        c = 0.0d0
        d = 1.0d0
        do diagonal_index = 1, m
            a(:, :, diagonal_index, diagonal_index) = -0.25d0
            b(:, :, diagonal_index, diagonal_index) =  2.00d0
            c(:, :, diagonal_index, diagonal_index) = -0.25d0
        end do
        if (rank == 0)        a(:, 1,        :, :) = 0.0d0
        if (rank == nprocs-1) c(:, nrow_sub, :, :) = 0.0d0

        call reset_gpu_timers()
        cuda_status = cudaDeviceSynchronize()
        if (cuda_status /= 0) call abort_benchmark('GPU initialisation failed', MPI_COMM_WORLD)
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        call btdma_many_mpi_gpu_v2(a, b, c, d, m, nsys_fixed, nrow_sub, plan)
        cuda_status = cudaDeviceSynchronize()
        if (cuda_status /= 0) call abort_benchmark('mod_btdma_gpu_v2 execution failed', MPI_COMM_WORLD)

        if (iteration > warmups) then
            call reduce_and_print('gpu', 'pascal', p_label, nprocs, m, n_global, &
                                  iteration-warmups, warmups, repeats, t2__b, t4__b, &
                                  MPI_COMM_WORLD)
        end if
    end do

    call btdma_cleanplan_gpu_v2(plan)
    deallocate(a, b, c, d)
    call MPI_Finalize(ierr)

contains

    subroutine reset_gpu_timers()
        t0__a = 0.0d0; t0__b = 0.0d0
        t1__a = 0.0d0; t1__b = 0.0d0
        t2__a = 0.0d0; t2__b = 0.0d0
        t3__a = 0.0d0; t3__b = 0.0d0
        t4__a = 0.0d0; t4__b = 0.0d0
        t5__a = 0.0d0; t5__b = 0.0d0
        comm__tp__a = 0.0d0; comm__tp__b = 0.0d0
        comm__tc__a = 0.0d0; comm__tc__b = 0.0d0
        comm__tu__a = 0.0d0; comm__tu__b = 0.0d0
    end subroutine reset_gpu_timers

    subroutine select_local_gpu(global_rank)
        integer, intent(in) :: global_rank
        integer :: local_rank, ngpu, device_id, env_status, ios, status_local
        character(len=64) :: local_rank_text

        local_rank = -1
        call get_environment_variable('SLURM_LOCALID', local_rank_text, status=env_status)
        if (env_status == 0) read(local_rank_text, *, iostat=ios) local_rank
        if (local_rank < 0) then
            call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK', local_rank_text, status=env_status)
            if (env_status == 0) read(local_rank_text, *, iostat=ios) local_rank
        end if
        if (local_rank < 0) local_rank = global_rank

        status_local = cudaGetDeviceCount(ngpu)
        if (status_local /= 0 .or. ngpu < 1) then
            call abort_benchmark('no visible CUDA device was found', MPI_COMM_WORLD)
        end if
        device_id = mod(local_rank, ngpu)
        status_local = cudaSetDevice(device_id)
        if (status_local /= 0) call abort_benchmark('cudaSetDevice failed', MPI_COMM_WORLD)
        status_local = cudaDeviceSynchronize()
        if (status_local /= 0) call abort_benchmark('initial CUDA synchronisation failed', MPI_COMM_WORLD)
    end subroutine select_local_gpu

end program benchmark_pascal_gpu
