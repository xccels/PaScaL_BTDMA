program bench_alltoall_gpu
    use mpi
    use cudafor
    use mpiutil
    use mod_btdma_gpu, only: btdma_many_gpu
    use node_compare_support
    implicit none

    character(len=16) :: regime
    integer :: ierr, myrank, nprocs, local_comm, local_rank, local_size
    integer :: warmups, repeats, run, nlocal, nsys_local
    real(dp), device, allocatable :: a(:,:,:,:), b(:,:,:,:), c(:,:,:,:), d(:,:,:)
    real(dp), device, allocatable :: a_line(:,:,:,:), b_line(:,:,:,:)
    real(dp), device, allocatable :: c_line(:,:,:,:), d_line(:,:,:)
    real(dp), device, allocatable :: center_matrix(:), line_matrix(:)
    real(dp), device, allocatable :: center_vector(:), line_vector(:)
    real(dp) :: wall_start, wall_elapsed, phases(5)
    type(a2a_plan) :: matrix_plan, vector_plan

    call MPI_Init(ierr)
    call check_mpi(ierr, 'MPI_Init')
    call configure_run(regime, warmups, repeats, myrank, nprocs, &
        local_comm, local_rank, local_size)
    nlocal = n_global / nprocs
    nsys_local = n_systems / nprocs

    allocate(a(block_size,block_size,n_systems,nlocal))
    allocate(b(block_size,block_size,n_systems,nlocal))
    allocate(c(block_size,block_size,n_systems,nlocal))
    allocate(d(block_size,n_systems,nlocal))
    allocate(a_line(block_size,block_size,nsys_local,n_global))
    allocate(b_line(block_size,block_size,nsys_local,n_global))
    allocate(c_line(block_size,block_size,nsys_local,n_global))
    allocate(d_line(block_size,nsys_local,n_global))
    allocate(center_matrix(block_size*block_size*n_systems*nlocal))
    allocate(line_matrix(block_size*block_size*nsys_local*n_global))
    allocate(center_vector(block_size*n_systems*nlocal))
    allocate(line_vector(block_size*nsys_local*n_global))

    call initialize_plan(matrix_plan, block_size*block_size, nlocal, nsys_local, myrank, nprocs)
    call initialize_plan(vector_plan, block_size, nlocal, nsys_local, myrank, nprocs)

    if (myrank == 0) then
        write(*,'(a)') 'NODE_SCHEMA,method,regime,repeat,nodes,gpus_per_node,mpi_ranks,' // &
            'local_ranks,m,N,nsys,Nlocal,phase1_s,phase2_s,phase3_s,phase4_s,' // &
            'phase5_s,computation_s,communication_s,total_s,wall_s,timing_gap_s'
    end if

    do run = 1, warmups
        call initialize_legacy_system(a, b, c, d)
        call reset_timers()
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        call check_mpi(ierr, 'MPI_Barrier(warmup)')
        call solve_once()
        call synchronize_gpu('cudaDeviceSynchronize(warmup)')
    end do

    do run = 1, repeats
        call initialize_legacy_system(a, b, c, d)
        call reset_timers()
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        call check_mpi(ierr, 'MPI_Barrier(measured)')
        call synchronize_gpu('cudaDeviceSynchronize(before total)')
        wall_start = MPI_Wtime()

        call solve_once()

        call synchronize_gpu('cudaDeviceSynchronize(after total)')
        wall_elapsed = MPI_Wtime() - wall_start
        phases = [0.0_dp, t2_b, t3_b, t4_b, 0.0_dp]
        call report_repeat('conventional_alltoall', regime, run, phases, &
            wall_elapsed, local_size, MPI_COMM_WORLD)
    end do

    call mpiutil_a2aplan_clean(matrix_plan)
    call mpiutil_a2aplan_clean(vector_plan)
    deallocate(a, b, c, d, a_line, b_line, c_line, d_line)
    deallocate(center_matrix, line_matrix, center_vector, line_vector)
    call MPI_Comm_free(local_comm, ierr)
    call check_mpi(ierr, 'MPI_Comm_free')
    call MPI_Finalize(ierr)

contains

    subroutine initialize_plan(plan, width, rows, local_systems, rank, ranks)
        type(a2a_plan), intent(inout) :: plan
        integer, intent(in) :: width, rows, local_systems, rank, ranks

        plan%myrank = rank
        plan%nprocs = ranks
        plan%mpi_comm = MPI_COMM_WORLD
        call mpiutil_a2aplan((/width,n_systems,rows/), &
            (/width,local_systems,n_global/), plan)
    end subroutine initialize_plan

    subroutine solve_once()
        call mpiutil_timecheck(t2_a, t2_b, 0)
        call redistribute_forward(a, a_line, center_matrix, line_matrix, matrix_plan)
        call redistribute_forward(b, b_line, center_matrix, line_matrix, matrix_plan)
        call redistribute_forward(c, c_line, center_matrix, line_matrix, matrix_plan)
        call redistribute_forward(d, d_line, center_vector, line_vector, vector_plan)
        call mpiutil_timecheck(t2_a, t2_b, 1)

        call mpiutil_timecheck(t3_a, t3_b, 0)
        call btdma_many_gpu(n_global, nsys_local, block_size, a_line, b_line, c_line, d_line)
        call mpiutil_timecheck(t3_a, t3_b, 1)

        call mpiutil_timecheck(t4_a, t4_b, 0)
        call redistribute_backward(d_line, d, line_vector, center_vector, vector_plan)
        call mpiutil_timecheck(t4_a, t4_b, 1)
    end subroutine solve_once

    subroutine redistribute_forward(source, destination, send_buffer, receive_buffer, plan)
        type(a2a_plan), intent(inout) :: plan
        real(dp), device, intent(in) :: source(1:plan%A%n(1),1:plan%A%n(2),1:plan%A%n(3))
        real(dp), device, intent(out) :: destination(1:plan%B%n(1),1:plan%B%n(2),1:plan%B%n(3))
        real(dp), device, intent(inout) :: send_buffer(:), receive_buffer(:)
        integer :: mpi_error

        call mpiutil_pack_gpu(source, send_buffer, plan%A, plan)
        call MPI_Alltoallv(send_buffer, plan%A%counts, plan%A%displs, MPI_DOUBLE_PRECISION, &
            receive_buffer, plan%B%counts, plan%B%displs, MPI_DOUBLE_PRECISION, &
            MPI_COMM_WORLD, mpi_error)
        call check_mpi(mpi_error, 'MPI_Alltoallv(forward)')
        call mpiutil_unpack_gpu(destination, receive_buffer, plan%B, plan)
    end subroutine redistribute_forward

    subroutine redistribute_backward(source, destination, send_buffer, receive_buffer, plan)
        type(a2a_plan), intent(inout) :: plan
        real(dp), device, intent(in) :: source(1:plan%B%n(1),1:plan%B%n(2),1:plan%B%n(3))
        real(dp), device, intent(out) :: destination(1:plan%A%n(1),1:plan%A%n(2),1:plan%A%n(3))
        real(dp), device, intent(inout) :: send_buffer(:), receive_buffer(:)
        integer :: mpi_error

        call mpiutil_pack_gpu(source, send_buffer, plan%B, plan)
        call MPI_Alltoallv(send_buffer, plan%B%counts, plan%B%displs, MPI_DOUBLE_PRECISION, &
            receive_buffer, plan%A%counts, plan%A%displs, MPI_DOUBLE_PRECISION, &
            MPI_COMM_WORLD, mpi_error)
        call check_mpi(mpi_error, 'MPI_Alltoallv(backward)')
        call mpiutil_unpack_gpu(destination, receive_buffer, plan%A, plan)
    end subroutine redistribute_backward

    subroutine reset_timers()
        t0_a = 0.0_dp; t0_b = 0.0_dp
        t1_a = 0.0_dp; t1_b = 0.0_dp
        t2_a = 0.0_dp; t2_b = 0.0_dp
        t3_a = 0.0_dp; t3_b = 0.0_dp
        t4_a = 0.0_dp; t4_b = 0.0_dp
        t5_a = 0.0_dp; t5_b = 0.0_dp
        comm_tp_a = 0.0_dp; comm_tp_b = 0.0_dp
        comm_tc_a = 0.0_dp; comm_tc_b = 0.0_dp
        comm_tu_a = 0.0_dp; comm_tu_b = 0.0_dp
    end subroutine reset_timers

end program bench_alltoall_gpu
