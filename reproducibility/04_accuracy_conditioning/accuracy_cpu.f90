program accuracy_cpu
    use mpi
    use accuracy_support
    use mod_btdma_cpu, only: BTDMA_PLAN, btdma_makeplan, btdma_many_mpi, btdma_cleanplan
    implicit none

    integer :: ierr, rank, nprocs
    integer :: m, log10_k, nsys_per_family, chunk_size
    integer :: family, first_system, system_in_chunk, system_id, family_offset
    integer :: output_unit, reference_status
    integer :: nrow_sub
    logical :: nonfinite, failed
    real(dp) :: kappa, residual_norm, relative_error
    character(len=256) :: output_path

    real(dp), allocatable :: lower_full(:,:,:), diagonal_full(:,:,:), upper_full(:,:,:)
    real(dp), allocatable :: lower_work_full(:,:,:), diagonal_work_full(:,:,:), upper_work_full(:,:,:)
    real(dp), allocatable :: rhs_full(:,:), rhs_work_full(:,:), exact_full(:,:)

    real(dp), allocatable :: lower(:,:,:,:), diagonal(:,:,:,:), upper(:,:,:,:), rhs(:,:,:)
    real(dp), allocatable :: lower_original(:,:,:,:), diagonal_original(:,:,:,:), upper_original(:,:,:,:)
    real(dp), allocatable :: rhs_original(:,:,:), exact_local(:,:,:)
    type(BTDMA_PLAN) :: plan

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

    call parse_arguments(m, log10_k, nsys_per_family, chunk_size, output_path)
    call validate_arguments(m, log10_k, nsys_per_family, chunk_size, nprocs, rank)

    nrow_sub = n_global_design / nprocs
    kappa = condition_value(log10_k)

    allocate(lower_full(m,m,n_global_design), diagonal_full(m,m,n_global_design), &
             upper_full(m,m,n_global_design))
    allocate(lower_work_full(m,m,n_global_design), diagonal_work_full(m,m,n_global_design), &
             upper_work_full(m,m,n_global_design))
    allocate(rhs_full(m,n_global_design), rhs_work_full(m,n_global_design), exact_full(m,n_global_design))

    allocate(lower(m,m,chunk_size,nrow_sub), diagonal(m,m,chunk_size,nrow_sub), &
             upper(m,m,chunk_size,nrow_sub), rhs(m,chunk_size,nrow_sub))
    allocate(lower_original(m,m,chunk_size,nrow_sub), diagonal_original(m,m,chunk_size,nrow_sub), &
             upper_original(m,m,chunk_size,nrow_sub), rhs_original(m,chunk_size,nrow_sub), &
             exact_local(m,chunk_size,nrow_sub))

    if (rank == 0) then
        open(newunit=output_unit, file=trim(output_path), status='replace', action='write')
        call write_accuracy_header(output_unit)
    else
        output_unit = -1
    end if

    call btdma_makeplan(plan, m, chunk_size, nrow_sub, MPI_COMM_WORLD)

    do family = 1, 2
        if (family == 1) then
            family_offset = 0
        else
            family_offset = nsys_family_design
        end if

        do first_system = 1, nsys_per_family, chunk_size
            if (rank == 0) then
                do system_in_chunk = 1, chunk_size
                    system_id = family_offset + first_system + system_in_chunk - 1
                    call generate_one_system(m, n_global_design, system_id, family, kappa, &
                                             lower_full, diagonal_full, upper_full, rhs_full, exact_full)
                    lower_work_full = lower_full
                    diagonal_work_full = diagonal_full
                    upper_work_full = upper_full
                    rhs_work_full = rhs_full
                    call sequential_block_thomas(m, n_global_design, lower_work_full, diagonal_work_full, &
                                                 upper_work_full, rhs_work_full, reference_status)
                    call system_metrics(m, n_global_design, lower_full, diagonal_full, upper_full, rhs_full, &
                                        rhs_work_full, exact_full, residual_norm, relative_error, nonfinite)
                    failed = reference_status /= 0 .or. nonfinite &
                             .or. residual_norm > residual_failure_tolerance
                    call write_accuracy_record(output_unit, m, log10_k, family, 2 * nsys_per_family, &
                                               system_id, 'sequential-cpu-bta', 1, 0, residual_norm, &
                                               relative_error, reference_status, .true., 0, nonfinite, failed)
                end do
            end if

            call generate_local_batch(m, family, kappa, family_offset + first_system, chunk_size, &
                                      rank, nrow_sub, lower_original, diagonal_original, upper_original, &
                                      rhs_original, exact_local)
            lower = lower_original
            diagonal = diagonal_original
            upper = upper_original
            rhs = rhs_original

            call MPI_Barrier(MPI_COMM_WORLD, ierr)
            call btdma_many_mpi(lower, diagonal, upper, rhs, m, chunk_size, nrow_sub, plan)
            call measure_distributed_cpu(m, log10_k, family, 2 * nsys_per_family, &
                                         family_offset + first_system, chunk_size, rank, nprocs, nrow_sub, &
                                         lower_original, diagonal_original, upper_original, rhs_original, &
                                         exact_local, rhs, output_unit)
            if (rank == 0) flush(output_unit)
        end do
    end do

    call btdma_cleanplan(plan)
    if (rank == 0) close(output_unit)

    deallocate(lower_full, diagonal_full, upper_full, lower_work_full, diagonal_work_full, upper_work_full)
    deallocate(rhs_full, rhs_work_full, exact_full)
    deallocate(lower, diagonal, upper, rhs, lower_original, diagonal_original, upper_original, rhs_original, exact_local)

    call MPI_Finalize(ierr)

contains

    subroutine parse_arguments(block_size, condition_exponent, family_count, chunk, path)
        integer, intent(out) :: block_size, condition_exponent, family_count, chunk
        character(len=*), intent(out) :: path
        character(len=64) :: argument

        if (command_argument_count() /= 5) then
            if (rank == 0) write(*,'(A)') &
                'Usage: accuracy_cpu M LOG10_K N_SYSTEMS_PER_FAMILY CHUNK_SIZE OUTPUT.csv'
            call MPI_Abort(MPI_COMM_WORLD, 2, ierr)
        end if
        call get_command_argument(1, argument)
        read(argument,*) block_size
        call get_command_argument(2, argument)
        read(argument,*) condition_exponent
        call get_command_argument(3, argument)
        read(argument,*) family_count
        call get_command_argument(4, argument)
        read(argument,*) chunk
        call get_command_argument(5, path)
    end subroutine parse_arguments

    subroutine validate_arguments(block_size, condition_exponent, family_count, chunk, process_count, process_rank)
        integer, intent(in) :: block_size, condition_exponent, family_count, chunk, process_count, process_rank
        logical :: valid

        valid = any(block_size == [2, 5, 8])
        valid = valid .and. any(condition_exponent == [2, 6, 10, 14])
        valid = valid .and. family_count > 0 .and. family_count <= nsys_family_design
        valid = valid .and. chunk > 0 .and. modulo(family_count, chunk) == 0
        valid = valid .and. chunk >= process_count
        valid = valid .and. modulo(n_global_design, process_count) == 0
        if (.not. valid) then
            if (process_rank == 0) write(*,'(A)') &
                'Invalid arguments: use m={2,5,8}, log10(K)={2,6,10,14}, a divisible chunk, and N mod MPI_ranks=0.'
            call MPI_Abort(MPI_COMM_WORLD, 3, ierr)
        end if
    end subroutine validate_arguments

    subroutine generate_local_batch(block_size, conditioning_family, target_condition, first_id, batch_size, &
                                    process_rank, local_rows, a, b, c, d, x)
        integer, intent(in) :: block_size, conditioning_family, first_id, batch_size, process_rank, local_rows
        real(dp), intent(in) :: target_condition
        real(dp), intent(out) :: a(block_size,block_size,batch_size,local_rows)
        real(dp), intent(out) :: b(block_size,block_size,batch_size,local_rows)
        real(dp), intent(out) :: c(block_size,block_size,batch_size,local_rows)
        real(dp), intent(out) :: d(block_size,batch_size,local_rows)
        real(dp), intent(out) :: x(block_size,batch_size,local_rows)
        integer :: local_row, batch_system, global_row, canonical_id

        do local_row = 1, local_rows
            global_row = process_rank * local_rows + local_row
            do batch_system = 1, batch_size
                canonical_id = first_id + batch_system - 1
                call build_blocks(block_size, n_global_design, canonical_id, global_row, conditioning_family, &
                                  target_condition, a(:,:,batch_system,local_row), &
                                  b(:,:,batch_system,local_row), c(:,:,batch_system,local_row))
                call build_rhs(block_size, n_global_design, canonical_id, global_row, &
                               a(:,:,batch_system,local_row), b(:,:,batch_system,local_row), &
                               c(:,:,batch_system,local_row), d(:,batch_system,local_row), &
                               x(:,batch_system,local_row))
            end do
        end do
    end subroutine generate_local_batch

    subroutine measure_distributed_cpu(block_size, condition_exponent, conditioning_family, executed_nsys, &
                                       first_id, batch_size, process_rank, process_count, local_rows, &
                                       a, b, c, original_d, exact, computed, csv_unit)
        integer, intent(in) :: block_size, condition_exponent, conditioning_family, executed_nsys
        integer, intent(in) :: first_id, batch_size, process_rank, process_count, local_rows, csv_unit
        real(dp), intent(in) :: a(block_size,block_size,batch_size,local_rows)
        real(dp), intent(in) :: b(block_size,block_size,batch_size,local_rows)
        real(dp), intent(in) :: c(block_size,block_size,batch_size,local_rows)
        real(dp), intent(in) :: original_d(block_size,batch_size,local_rows)
        real(dp), intent(in) :: exact(block_size,batch_size,local_rows)
        real(dp), intent(in) :: computed(block_size,batch_size,local_rows)

        integer :: left_rank, right_rank, local_row, batch_system, global_row, canonical_id
        integer :: nonfinite_local(batch_size), nonfinite_global(batch_size)
        real(dp) :: sums_local(4,batch_size), sums_global(4,batch_size)
        real(dp) :: left_ghost(block_size,batch_size), right_ghost(block_size,batch_size)
        real(dp) :: previous_x(block_size), next_x(block_size), residual(block_size)
        real(dp) :: residual_value, error_value
        logical :: record_nonfinite, record_failed

        left_rank = process_rank - 1
        right_rank = process_rank + 1
        if (left_rank < 0) left_rank = MPI_PROC_NULL
        if (right_rank >= process_count) right_rank = MPI_PROC_NULL
        left_ghost = 0.0_dp
        right_ghost = 0.0_dp

        call MPI_Sendrecv(computed(:,:,local_rows), block_size * batch_size, MPI_DOUBLE_PRECISION, &
                          right_rank, 401, left_ghost, block_size * batch_size, MPI_DOUBLE_PRECISION, &
                          left_rank, 401, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
        call MPI_Sendrecv(computed(:,:,1), block_size * batch_size, MPI_DOUBLE_PRECISION, &
                          left_rank, 402, right_ghost, block_size * batch_size, MPI_DOUBLE_PRECISION, &
                          right_rank, 402, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)

        sums_local = 0.0_dp
        nonfinite_local = 0
        do local_row = 1, local_rows
            global_row = process_rank * local_rows + local_row
            do batch_system = 1, batch_size
                previous_x = 0.0_dp
                next_x = 0.0_dp
                if (global_row > 1) then
                    if (local_row > 1) then
                        previous_x = computed(:,batch_system,local_row-1)
                    else
                        previous_x = left_ghost(:,batch_system)
                    end if
                end if
                if (global_row < n_global_design) then
                    if (local_row < local_rows) then
                        next_x = computed(:,batch_system,local_row+1)
                    else
                        next_x = right_ghost(:,batch_system)
                    end if
                end if
                residual = matmul(a(:,:,batch_system,local_row), previous_x) &
                         + matmul(b(:,:,batch_system,local_row), computed(:,batch_system,local_row)) &
                         + matmul(c(:,:,batch_system,local_row), next_x) - original_d(:,batch_system,local_row)
                sums_local(1,batch_system) = sums_local(1,batch_system) + sum(residual * residual)
                sums_local(2,batch_system) = sums_local(2,batch_system) &
                                            + sum(original_d(:,batch_system,local_row)**2)
                sums_local(3,batch_system) = sums_local(3,batch_system) &
                                            + sum((computed(:,batch_system,local_row) &
                                            - exact(:,batch_system,local_row))**2)
                sums_local(4,batch_system) = sums_local(4,batch_system) &
                                            + sum(exact(:,batch_system,local_row)**2)
                if (vector_has_nonfinite(computed(:,batch_system,local_row))) nonfinite_local(batch_system) = 1
            end do
        end do

        call MPI_Reduce(sums_local, sums_global, 4 * batch_size, MPI_DOUBLE_PRECISION, MPI_SUM, 0, &
                        MPI_COMM_WORLD, ierr)
        call MPI_Reduce(nonfinite_local, nonfinite_global, batch_size, MPI_INTEGER, MPI_MAX, 0, &
                        MPI_COMM_WORLD, ierr)

        if (process_rank == 0) then
            do batch_system = 1, batch_size
                canonical_id = first_id + batch_system - 1
                residual_value = sqrt(sums_global(1,batch_system) / sums_global(2,batch_system))
                error_value = sqrt(sums_global(3,batch_system) / sums_global(4,batch_system))
                record_nonfinite = nonfinite_global(batch_system) /= 0 &
                                   .or. .not. ieee_is_finite(residual_value) &
                                   .or. .not. ieee_is_finite(error_value)
                record_failed = record_nonfinite .or. residual_value > residual_failure_tolerance
                call write_accuracy_record(csv_unit, block_size, condition_exponent, conditioning_family, &
                                           executed_nsys, canonical_id, 'cpu-pascal-btdma', process_count, 0, &
                                           residual_value, error_value, -999, .false., 0, record_nonfinite, &
                                           record_failed)
            end do
        end if
    end subroutine measure_distributed_cpu

end program accuracy_cpu
