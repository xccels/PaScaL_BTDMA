program singular_probe_cpu
    use mpi
    use accuracy_support
    use mod_btdma_cpu, only: BTDMA_PLAN, btdma_makeplan, btdma_many_mpi, btdma_cleanplan
    implicit none

    integer, parameter :: m = 2, nsys = 8, nrow_sub = 4
    integer :: ierr, rank, nprocs, n_global, probe, output_unit
    real(dp) :: residual_norm, relative_error
    logical :: nonfinite
    character(len=256) :: output_path
    real(dp), allocatable :: lower(:,:,:,:), diagonal(:,:,:,:), upper(:,:,:,:), rhs(:,:,:)
    real(dp), allocatable :: lower_original(:,:,:,:), diagonal_original(:,:,:,:), upper_original(:,:,:,:)
    real(dp), allocatable :: rhs_original(:,:,:), exact_local(:,:,:)
    type(BTDMA_PLAN) :: plan

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
    n_global = nrow_sub * nprocs

    if (command_argument_count() /= 1) then
        if (rank == 0) write(*,'(A)') 'Usage: singular_probe_cpu OUTPUT.csv'
        call MPI_Abort(MPI_COMM_WORLD, 2, ierr)
    end if
    call get_command_argument(1, output_path)

    allocate(lower(m,m,nsys,nrow_sub), diagonal(m,m,nsys,nrow_sub), &
             upper(m,m,nsys,nrow_sub), rhs(m,nsys,nrow_sub))
    allocate(lower_original(m,m,nsys,nrow_sub), diagonal_original(m,m,nsys,nrow_sub), &
             upper_original(m,m,nsys,nrow_sub), rhs_original(m,nsys,nrow_sub), exact_local(m,nsys,nrow_sub))

    if (rank == 0) then
        open(newunit=output_unit, file=trim(output_path), status='replace', action='write')
        write(output_unit,'(A)') 'probe,solver,m,N,n_sys,mpi_ranks,gpu_count,returned_from_call,' // &
            'api_status,status_available,runtime_status,residual_norm,relative_solution_error,nonfinite,warning_channel'
    else
        output_unit = -1
    end if

    call btdma_makeplan(plan, m, nsys, nrow_sub, MPI_COMM_WORLD)
    do probe = 1, 2
        call generate_probe(probe, rank, n_global, lower_original, diagonal_original, upper_original, &
                            rhs_original, exact_local)
        lower = lower_original
        diagonal = diagonal_original
        upper = upper_original
        rhs = rhs_original
        call btdma_many_mpi(lower, diagonal, upper, rhs, m, nsys, nrow_sub, plan)
        call probe_metrics(rank, nprocs, n_global, lower_original, diagonal_original, upper_original, &
                           rhs_original, exact_local, rhs, residual_norm, relative_error, nonfinite)
        if (rank == 0) then
            write(output_unit,'(*(g0,:,","))') trim(singular_probe_label(probe)), 'cpu-pascal-btdma', &
                m, n_global, nsys, nprocs, 0, 'yes', -999, 'no', 0, residual_norm, relative_error, &
                trim(merge('yes', 'no ', nonfinite)), 'none-in-public-api'
            flush(output_unit)
        end if
    end do

    call btdma_cleanplan(plan)
    if (rank == 0) close(output_unit)
    deallocate(lower, diagonal, upper, rhs, lower_original, diagonal_original, upper_original, rhs_original, exact_local)
    call MPI_Finalize(ierr)

contains

    subroutine generate_probe(probe_id, process_rank, global_rows, a, b, c, d, x)
        integer, intent(in) :: probe_id, process_rank, global_rows
        real(dp), intent(out) :: a(m,m,nsys,nrow_sub), b(m,m,nsys,nrow_sub), c(m,m,nsys,nrow_sub)
        real(dp), intent(out) :: d(m,nsys,nrow_sub), x(m,nsys,nrow_sub)
        integer :: local_row, system, global_row
        real(dp) :: x_previous(m), x_next(m)

        do local_row = 1, nrow_sub
            global_row = process_rank * nrow_sub + local_row
            do system = 1, nsys
                call build_singular_probe_blocks(m, global_rows, global_row, probe_id, &
                                                 a(:,:,system,local_row), b(:,:,system,local_row), &
                                                 c(:,:,system,local_row))
                call exact_vector(m, system, global_row, x(:,system,local_row))
                x_previous = 0.0_dp
                x_next = 0.0_dp
                if (global_row > 1) call exact_vector(m, system, global_row - 1, x_previous)
                if (global_row < global_rows) call exact_vector(m, system, global_row + 1, x_next)
                d(:,system,local_row) = matmul(a(:,:,system,local_row), x_previous) &
                                      + matmul(b(:,:,system,local_row), x(:,system,local_row)) &
                                      + matmul(c(:,:,system,local_row), x_next)
            end do
        end do
    end subroutine generate_probe

    subroutine probe_metrics(process_rank, process_count, global_rows, a, b, c, d, exact, computed, &
                             residual_value, error_value, has_nonfinite)
        integer, intent(in) :: process_rank, process_count, global_rows
        real(dp), intent(in) :: a(m,m,nsys,nrow_sub), b(m,m,nsys,nrow_sub), c(m,m,nsys,nrow_sub)
        real(dp), intent(in) :: d(m,nsys,nrow_sub), exact(m,nsys,nrow_sub), computed(m,nsys,nrow_sub)
        real(dp), intent(out) :: residual_value, error_value
        logical, intent(out) :: has_nonfinite
        integer :: left_rank, right_rank, local_row, system, global_row, nonfinite_local, nonfinite_global
        real(dp) :: left_ghost(m,nsys), right_ghost(m,nsys), previous_x(m), next_x(m), residual(m)
        real(dp) :: sums_local(4), sums_global(4)

        left_rank = process_rank - 1
        right_rank = process_rank + 1
        if (left_rank < 0) left_rank = MPI_PROC_NULL
        if (right_rank >= process_count) right_rank = MPI_PROC_NULL
        left_ghost = 0.0_dp
        right_ghost = 0.0_dp
        call MPI_Sendrecv(computed(:,:,nrow_sub), m * nsys, MPI_DOUBLE_PRECISION, right_rank, 421, &
                          left_ghost, m * nsys, MPI_DOUBLE_PRECISION, left_rank, 421, &
                          MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
        call MPI_Sendrecv(computed(:,:,1), m * nsys, MPI_DOUBLE_PRECISION, left_rank, 422, &
                          right_ghost, m * nsys, MPI_DOUBLE_PRECISION, right_rank, 422, &
                          MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)

        sums_local = 0.0_dp
        nonfinite_local = 0
        do local_row = 1, nrow_sub
            global_row = process_rank * nrow_sub + local_row
            do system = 1, nsys
                previous_x = 0.0_dp
                next_x = 0.0_dp
                if (global_row > 1) then
                    if (local_row > 1) then
                        previous_x = computed(:,system,local_row-1)
                    else
                        previous_x = left_ghost(:,system)
                    end if
                end if
                if (global_row < global_rows) then
                    if (local_row < nrow_sub) then
                        next_x = computed(:,system,local_row+1)
                    else
                        next_x = right_ghost(:,system)
                    end if
                end if
                residual = matmul(a(:,:,system,local_row), previous_x) &
                         + matmul(b(:,:,system,local_row), computed(:,system,local_row)) &
                         + matmul(c(:,:,system,local_row), next_x) - d(:,system,local_row)
                sums_local(1) = sums_local(1) + sum(residual**2)
                sums_local(2) = sums_local(2) + sum(d(:,system,local_row)**2)
                sums_local(3) = sums_local(3) + sum((computed(:,system,local_row)-exact(:,system,local_row))**2)
                sums_local(4) = sums_local(4) + sum(exact(:,system,local_row)**2)
                if (vector_has_nonfinite(computed(:,system,local_row))) nonfinite_local = 1
            end do
        end do
        call MPI_Reduce(sums_local, sums_global, 4, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
        call MPI_Reduce(nonfinite_local, nonfinite_global, 1, MPI_INTEGER, MPI_MAX, 0, MPI_COMM_WORLD, ierr)
        if (process_rank == 0) then
            residual_value = sqrt(sums_global(1) / sums_global(2))
            error_value = sqrt(sums_global(3) / sums_global(4))
            has_nonfinite = nonfinite_global /= 0 .or. .not. ieee_is_finite(residual_value) &
                            .or. .not. ieee_is_finite(error_value)
        end if
    end subroutine probe_metrics

end program singular_probe_cpu
