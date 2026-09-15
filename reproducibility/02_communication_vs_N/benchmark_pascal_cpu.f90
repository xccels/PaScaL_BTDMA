program benchmark_pascal_cpu
    use mpi
    use mpiutil
    use mod_btdma_cpu
    use communication_benchmark_common
    implicit none

    integer :: ierr, rank, nprocs
    integer :: n_global, m, warmups, repeats, p_label
    integer :: nrow_sub, first_row, last_row
    integer :: iteration, diagonal_index
    real(kind=8), allocatable :: a(:,:,:,:), b(:,:,:,:), c(:,:,:,:), d(:,:,:)
    type(BTDMA_PLAN) :: plan

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
    call parse_benchmark_args(n_global, m, warmups, repeats, p_label, MPI_COMM_WORLD)
    if (nprocs /= 24*p_label) then
        call abort_benchmark('CPU P=8 requires 24 MPI ranks per socket and 192 ranks in total', &
                             MPI_COMM_WORLD)
    end if

    nrow_sub = mpiutil_para(1, n_global, rank, nprocs, first_row, last_row)
    call require_minimum_local_rows(nrow_sub, MPI_COMM_WORLD)

    allocate(a(m, m, nsys_fixed, nrow_sub))
    allocate(b(m, m, nsys_fixed, nrow_sub))
    allocate(c(m, m, nsys_fixed, nrow_sub))
    allocate(d(m,    nsys_fixed, nrow_sub))
    call btdma_makeplan(plan, m, nsys_fixed, nrow_sub, MPI_COMM_WORLD)

    call print_result_header(rank, &
        'PaScaL t2+t4; current CPU core Step2 exchanges reduced A+B+C+RHS and Step4 exchanges reduced A+B+C+solution')

    do iteration = 1, warmups + repeats
        a = 0.0d0
        b = 0.0d0
        c = 0.0d0
        d = 1.0d0
        do diagonal_index = 1, m
            a(diagonal_index, diagonal_index, :, :) = -0.25d0
            b(diagonal_index, diagonal_index, :, :) =  2.00d0
            c(diagonal_index, diagonal_index, :, :) = -0.25d0
        end do
        if (rank == 0)        a(:, :, :, 1)        = 0.0d0
        if (rank == nprocs-1) c(:, :, :, nrow_sub) = 0.0d0

        call reset_cpu_timers()
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        call btdma_many_mpi(a, b, c, d, m, nsys_fixed, nrow_sub, plan)

        if (iteration > warmups) then
            call reduce_and_print('cpu', 'pascal', p_label, nprocs, m, n_global, &
                                  iteration-warmups, warmups, repeats, t2_b, t4_b, MPI_COMM_WORLD)
        end if
    end do

    call btdma_cleanplan(plan)
    deallocate(a, b, c, d)
    call MPI_Finalize(ierr)

contains

    subroutine reset_cpu_timers()
        t0_a = 0.0d0; t0_b = 0.0d0
        t1_a = 0.0d0; t1_b = 0.0d0
        t2_a = 0.0d0; t2_b = 0.0d0
        t3_a = 0.0d0; t3_b = 0.0d0
        t4_a = 0.0d0; t4_b = 0.0d0
        t5_a = 0.0d0; t5_b = 0.0d0
        comm_tp_a = 0.0d0; comm_tp_b = 0.0d0
        comm_tc_a = 0.0d0; comm_tc_b = 0.0d0
        comm_tu_a = 0.0d0; comm_tu_b = 0.0d0
    end subroutine reset_cpu_timers

end program benchmark_pascal_cpu
