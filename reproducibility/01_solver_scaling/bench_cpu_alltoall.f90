program bench_cpu_alltoall
    use mpi
    use mpiutil
    use mod_btdma_cpu, only: btdma_many
    use benchmark_support
    implicit none

    type(benchmark_config) :: cfg
    type(a2a_plan) :: matrix_plan, vector_plan
    integer :: ierr, myrank, nprocs
    integer :: nlocal, nsys_local, iteration, repeat_index
    real(dp), allocatable :: a(:,:,:,:), b(:,:,:,:), c(:,:,:,:), d(:,:,:)
    real(dp), allocatable :: a_line(:,:,:,:), b_line(:,:,:,:), c_line(:,:,:,:), d_line(:,:,:)
    real(dp), allocatable :: center_matrix(:), line_matrix(:), center_vector(:), line_vector(:)
    real(dp) :: start_time, elapsed
    real(dp) :: timing(8)

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
    call read_config(cfg, MPI_COMM_WORLD)
    call validate_mapping(cfg, 24, "CPU", MPI_COMM_WORLD)

    nlocal = local_extent(cfg%n_global, myrank, nprocs)
    nsys_local = local_extent(cfg%nsys, myrank, nprocs)

    allocate(a(cfg%m,cfg%m,cfg%nsys,nlocal), b(cfg%m,cfg%m,cfg%nsys,nlocal))
    allocate(c(cfg%m,cfg%m,cfg%nsys,nlocal), d(cfg%m,cfg%nsys,nlocal))
    allocate(a_line(cfg%m,cfg%m,nsys_local,cfg%n_global))
    allocate(b_line(cfg%m,cfg%m,nsys_local,cfg%n_global))
    allocate(c_line(cfg%m,cfg%m,nsys_local,cfg%n_global))
    allocate(d_line(cfg%m,nsys_local,cfg%n_global))
    allocate(center_matrix(cfg%m*cfg%m*cfg%nsys*nlocal))
    allocate(line_matrix(cfg%m*cfg%m*nsys_local*cfg%n_global))
    allocate(center_vector(cfg%m*cfg%nsys*nlocal))
    allocate(line_vector(cfg%m*nsys_local*cfg%n_global))

    call initialize_plan(matrix_plan, cfg%m*cfg%m, cfg%nsys, nlocal, &
        nsys_local, cfg%n_global, myrank, nprocs)
    call initialize_plan(vector_plan, cfg%m, cfg%nsys, nlocal, &
        nsys_local, cfg%n_global, myrank, nprocs)

    call print_result_header(myrank)
    do iteration = 1, cfg%nwarmup + cfg%nrepeat
        call initialize_cpu_system(a, b, c, d, cfg%m, myrank, nprocs)
        call reset_mpiutil_timers()
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        start_time = MPI_Wtime()

        call mpiutil_timecheck(t2_a, t2_b, 0)
        call redistribute_forward(a, a_line, center_matrix, line_matrix, matrix_plan)
        call redistribute_forward(b, b_line, center_matrix, line_matrix, matrix_plan)
        call redistribute_forward(c, c_line, center_matrix, line_matrix, matrix_plan)
        call redistribute_forward(d, d_line, center_vector, line_vector, vector_plan)
        call mpiutil_timecheck(t2_a, t2_b, 1)

        call mpiutil_timecheck(t3_a, t3_b, 0)
        call btdma_many(cfg%n_global, nsys_local, cfg%m, a_line, b_line, c_line, d_line)
        call mpiutil_timecheck(t3_a, t3_b, 1)

        call mpiutil_timecheck(t4_a, t4_b, 0)
        call redistribute_backward(d_line, d, line_vector, center_vector, vector_plan)
        call mpiutil_timecheck(t4_a, t4_b, 1)

        elapsed = MPI_Wtime() - start_time
        if (iteration > cfg%nwarmup) then
            repeat_index = iteration - cfg%nwarmup
            timing = (/ elapsed, t3_b, t2_b + t4_b, 0.0_dp, t2_b, t3_b, t4_b, 0.0_dp /)
            call reduce_and_print(cfg, "alltoall", "cpu", repeat_index, timing, MPI_COMM_WORLD)
        end if
    end do

    call mpiutil_a2aplan_clean(matrix_plan)
    call mpiutil_a2aplan_clean(vector_plan)
    deallocate(a, b, c, d, a_line, b_line, c_line, d_line)
    deallocate(center_matrix, line_matrix, center_vector, line_vector)
    call MPI_Finalize(ierr)

contains

    subroutine initialize_plan(plan, width, nsys, nrow, nsys_part, nrow_global, rank, ranks)
        type(a2a_plan), intent(inout) :: plan
        integer, intent(in) :: width, nsys, nrow, nsys_part, nrow_global, rank, ranks
        plan%myrank = rank
        plan%nprocs = ranks
        plan%mpi_comm = MPI_COMM_WORLD
        call mpiutil_a2aplan((/width,nsys,nrow/), &
            (/width,nsys_part,nrow_global/), plan)
    end subroutine initialize_plan

    subroutine redistribute_forward(source, destination, send_buffer, receive_buffer, plan)
        type(a2a_plan), intent(inout) :: plan
        real(dp), intent(in) :: source(1:plan%A%n(1),1:plan%A%n(2),1:plan%A%n(3))
        real(dp), intent(out) :: destination(1:plan%B%n(1),1:plan%B%n(2),1:plan%B%n(3))
        real(dp), intent(inout) :: send_buffer(:), receive_buffer(:)
        integer :: mpi_error

        call mpiutil_pack(source, send_buffer, plan%A, plan)
        call MPI_Alltoallv(send_buffer, plan%A%counts, plan%A%displs, MPI_DOUBLE_PRECISION, &
            receive_buffer, plan%B%counts, plan%B%displs, MPI_DOUBLE_PRECISION, &
            MPI_COMM_WORLD, mpi_error)
        call mpiutil_unpack(destination, receive_buffer, plan%B, plan)
    end subroutine redistribute_forward

    subroutine redistribute_backward(source, destination, send_buffer, receive_buffer, plan)
        type(a2a_plan), intent(inout) :: plan
        real(dp), intent(in) :: source(1:plan%B%n(1),1:plan%B%n(2),1:plan%B%n(3))
        real(dp), intent(out) :: destination(1:plan%A%n(1),1:plan%A%n(2),1:plan%A%n(3))
        real(dp), intent(inout) :: send_buffer(:), receive_buffer(:)
        integer :: mpi_error

        call mpiutil_pack(source, send_buffer, plan%B, plan)
        call MPI_Alltoallv(send_buffer, plan%B%counts, plan%B%displs, MPI_DOUBLE_PRECISION, &
            receive_buffer, plan%A%counts, plan%A%displs, MPI_DOUBLE_PRECISION, &
            MPI_COMM_WORLD, mpi_error)
        call mpiutil_unpack(destination, receive_buffer, plan%A, plan)
    end subroutine redistribute_backward

    subroutine reset_mpiutil_timers()
        t0_a=0.0_dp; t0_b=0.0_dp; t1_a=0.0_dp; t1_b=0.0_dp
        t2_a=0.0_dp; t2_b=0.0_dp; t3_a=0.0_dp; t3_b=0.0_dp
        t4_a=0.0_dp; t4_b=0.0_dp; t5_a=0.0_dp; t5_b=0.0_dp
        comm_tp_a=0.0_dp; comm_tp_b=0.0_dp
        comm_tc_a=0.0_dp; comm_tc_b=0.0_dp
        comm_tu_a=0.0_dp; comm_tu_b=0.0_dp
    end subroutine reset_mpiutil_timers

end program bench_cpu_alltoall
