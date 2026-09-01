program benchmark_alltoall_gpu
    use mpi
    use cudafor
    use mpiutil
    use communication_baseline_gpu, only: solve_local_baseline_gpu
    use communication_benchmark_common
    implicit none

    integer :: ierr, rank, nprocs, cuda_status
    integer :: n_global, m, warmups, repeats, p_label
    integer :: nrow_sub, nsys_sub, first_row, last_row, first_sys, last_sys
    integer :: iteration, diagonal_index
    real(kind=8) :: start_time, forward_time, backward_time

    real(kind=8), device, allocatable :: a_local(:,:,:), b_local(:,:,:)
    real(kind=8), device, allocatable :: c_local(:,:,:), d_local(:,:,:)
    real(kind=8), device, allocatable :: a_line(:,:,:,:), b_line(:,:,:,:)
    real(kind=8), device, allocatable :: c_line(:,:,:,:), d_line(:,:,:)
    real(kind=8), device, allocatable :: send_buffer(:), receive_buffer(:)
    type(a2a_plan) :: matrix_plan, vector_plan

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
    call parse_benchmark_args(n_global, m, warmups, repeats, p_label, MPI_COMM_WORLD)
    if (nprocs /= p_label) then
        call abort_benchmark('GPU P=8 requires one MPI rank per GPU and 8 ranks in total', &
                             MPI_COMM_WORLD)
    end if
    call select_local_gpu(rank)

    nrow_sub = mpiutil_para(1, n_global, rank, nprocs, first_row, last_row)
    nsys_sub = mpiutil_para(1, nsys_fixed, rank, nprocs, first_sys, last_sys)
    call require_minimum_local_rows(nrow_sub, MPI_COMM_WORLD)

    allocate(a_local(m*m, nsys_fixed, nrow_sub))
    allocate(b_local(m*m, nsys_fixed, nrow_sub))
    allocate(c_local(m*m, nsys_fixed, nrow_sub))
    allocate(d_local(m,   nsys_fixed, nrow_sub))
    allocate(a_line(m, m, nsys_sub, n_global))
    allocate(b_line(m, m, nsys_sub, n_global))
    allocate(c_line(m, m, nsys_sub, n_global))
    allocate(d_line(m,    nsys_sub, n_global))
    allocate(send_buffer(m*m*nsys_fixed*nrow_sub))
    allocate(receive_buffer(m*m*nsys_sub*n_global))

    matrix_plan%myrank = rank
    matrix_plan%nprocs = nprocs
    matrix_plan%mpi_comm = MPI_COMM_WORLD
    vector_plan%myrank = rank
    vector_plan%nprocs = nprocs
    vector_plan%mpi_comm = MPI_COMM_WORLD
    call mpiutil_a2aplan((/m*m, nsys_fixed, nrow_sub/), &
                         (/m*m, nsys_sub, n_global/), matrix_plan)
    call mpiutil_a2aplan((/m, nsys_fixed, nrow_sub/), &
                         (/m, nsys_sub, n_global/), vector_plan)

    call print_result_header(rank, &
        'alltoall forward=A+B+C+RHS full-row GPU pack+MPI_Alltoallv+GPU unpack; ' // &
        'backward=solution GPU pack+MPI_Alltoallv+GPU unpack')

    do iteration = 1, warmups + repeats
        a_local = 0.0d0
        b_local = 0.0d0
        c_local = 0.0d0
        d_local = 1.0d0
        do diagonal_index = 1, m
            a_local(diagonal_index + (diagonal_index-1)*m, :, :) = -0.25d0
            b_local(diagonal_index + (diagonal_index-1)*m, :, :) =  2.00d0
            c_local(diagonal_index + (diagonal_index-1)*m, :, :) = -0.25d0
        end do
        if (rank == 0)        a_local(:, :, 1)        = 0.0d0
        if (rank == nprocs-1) c_local(:, :, nrow_sub) = 0.0d0

        cuda_status = cudaDeviceSynchronize()
        if (cuda_status /= 0) call abort_benchmark('GPU initialisation failed', MPI_COMM_WORLD)
        call MPI_Barrier(MPI_COMM_WORLD, ierr)

        start_time = MPI_Wtime()
        call redistribute_forward_gpu()
        cuda_status = cudaDeviceSynchronize()
        if (cuda_status /= 0) call abort_benchmark('GPU forward redistribution failed', MPI_COMM_WORLD)
        forward_time = MPI_Wtime() - start_time

        call solve_local_baseline_gpu(n_global, nsys_sub, m, a_line, b_line, c_line, d_line)
        cuda_status = cudaDeviceSynchronize()
        if (cuda_status /= 0) call abort_benchmark('GPU local baseline solve failed', MPI_COMM_WORLD)

        start_time = MPI_Wtime()
        call redistribute_backward_gpu()
        cuda_status = cudaDeviceSynchronize()
        if (cuda_status /= 0) call abort_benchmark('GPU backward redistribution failed', MPI_COMM_WORLD)
        backward_time = MPI_Wtime() - start_time

        if (iteration > warmups) then
            call reduce_and_print('gpu', 'alltoall', p_label, nprocs, m, n_global, &
                                  iteration-warmups, warmups, repeats, forward_time, backward_time, &
                                  MPI_COMM_WORLD)
        end if
    end do

    call mpiutil_a2aplan_clean(matrix_plan)
    call mpiutil_a2aplan_clean(vector_plan)
    deallocate(a_local, b_local, c_local, d_local)
    deallocate(a_line, b_line, c_line, d_line)
    deallocate(send_buffer, receive_buffer)
    call MPI_Finalize(ierr)

contains

    subroutine redistribute_forward_gpu()
        call mpiutil_pack_gpu(a_local, send_buffer, matrix_plan%A, matrix_plan)
        call MPI_Alltoallv(send_buffer, matrix_plan%A%counts, matrix_plan%A%displs, MPI_DOUBLE_PRECISION, &
                           receive_buffer, matrix_plan%B%counts, matrix_plan%B%displs, MPI_DOUBLE_PRECISION, &
                           MPI_COMM_WORLD, ierr)
        call mpiutil_unpack_gpu(a_line, receive_buffer, matrix_plan%B, matrix_plan)

        call mpiutil_pack_gpu(b_local, send_buffer, matrix_plan%A, matrix_plan)
        call MPI_Alltoallv(send_buffer, matrix_plan%A%counts, matrix_plan%A%displs, MPI_DOUBLE_PRECISION, &
                           receive_buffer, matrix_plan%B%counts, matrix_plan%B%displs, MPI_DOUBLE_PRECISION, &
                           MPI_COMM_WORLD, ierr)
        call mpiutil_unpack_gpu(b_line, receive_buffer, matrix_plan%B, matrix_plan)

        call mpiutil_pack_gpu(c_local, send_buffer, matrix_plan%A, matrix_plan)
        call MPI_Alltoallv(send_buffer, matrix_plan%A%counts, matrix_plan%A%displs, MPI_DOUBLE_PRECISION, &
                           receive_buffer, matrix_plan%B%counts, matrix_plan%B%displs, MPI_DOUBLE_PRECISION, &
                           MPI_COMM_WORLD, ierr)
        call mpiutil_unpack_gpu(c_line, receive_buffer, matrix_plan%B, matrix_plan)

        call mpiutil_pack_gpu(d_local, send_buffer, vector_plan%A, vector_plan)
        call MPI_Alltoallv(send_buffer, vector_plan%A%counts, vector_plan%A%displs, MPI_DOUBLE_PRECISION, &
                           receive_buffer, vector_plan%B%counts, vector_plan%B%displs, MPI_DOUBLE_PRECISION, &
                           MPI_COMM_WORLD, ierr)
        call mpiutil_unpack_gpu(d_line, receive_buffer, vector_plan%B, vector_plan)
    end subroutine redistribute_forward_gpu

    subroutine redistribute_backward_gpu()
        call mpiutil_pack_gpu(d_line, receive_buffer, vector_plan%B, vector_plan)
        call MPI_Alltoallv(receive_buffer, vector_plan%B%counts, vector_plan%B%displs, MPI_DOUBLE_PRECISION, &
                           send_buffer, vector_plan%A%counts, vector_plan%A%displs, MPI_DOUBLE_PRECISION, &
                           MPI_COMM_WORLD, ierr)
        call mpiutil_unpack_gpu(d_local, send_buffer, vector_plan%A, vector_plan)
    end subroutine redistribute_backward_gpu

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

end program benchmark_alltoall_gpu
