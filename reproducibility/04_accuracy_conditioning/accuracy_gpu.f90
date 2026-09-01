program accuracy_gpu
    use mpi
    use cudafor
    use accuracy_support
    use mod_btdma_gpu_v2, only: BTDMA_PLAN_gpu_v2, btdma_makeplan_gpu_v2, &
                                btdma_many_mpi_gpu_v2, btdma_cleanplan_gpu_v2
    implicit none

    integer :: ierr, rank, nprocs, local_comm, local_rank
    integer :: gpu_count_visible, device_id, cuda_status
    integer :: m, log10_k, nsys_per_family, chunk_size
    integer :: family, first_system, family_offset, nrow_sub, output_unit
    real(dp) :: kappa
    character(len=256) :: output_path

    real(dp), allocatable :: lower(:,:,:,:), diagonal(:,:,:,:), upper(:,:,:,:), rhs(:,:,:)
    real(dp), allocatable :: lower_original(:,:,:,:), diagonal_original(:,:,:,:), upper_original(:,:,:,:)
    real(dp), allocatable :: rhs_original(:,:,:), exact_local(:,:,:)
    real(dp), device, allocatable :: lower_device(:,:,:,:), diagonal_device(:,:,:,:)
    real(dp), device, allocatable :: upper_device(:,:,:,:), rhs_device(:,:,:)
    type(BTDMA_PLAN_gpu_v2) :: plan

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
    call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, local_comm, ierr)
    call MPI_Comm_rank(local_comm, local_rank, ierr)

    call parse_arguments(m, log10_k, nsys_per_family, chunk_size, output_path)
    call validate_arguments(m, log10_k, nsys_per_family, chunk_size, nprocs, rank)

    cuda_status = cudaGetDeviceCount(gpu_count_visible)
    if (cuda_status /= cudaSuccess .or. gpu_count_visible < 1) then
        if (rank == 0) write(*,'(A)') 'No CUDA device is visible.'
        call MPI_Abort(MPI_COMM_WORLD, 4, ierr)
    end if
    device_id = modulo(local_rank, gpu_count_visible)
    cuda_status = cudaSetDevice(device_id)
    if (cuda_status /= cudaSuccess) then
        write(*,'(A,I0,A,I0)') 'Rank ', rank, ' failed to select CUDA device ', device_id
        call MPI_Abort(MPI_COMM_WORLD, 5, ierr)
    end if

    nrow_sub = n_global_design / nprocs
    kappa = condition_value(log10_k)

    allocate(lower(chunk_size,nrow_sub,m,m), diagonal(chunk_size,nrow_sub,m,m), &
             upper(chunk_size,nrow_sub,m,m), rhs(chunk_size,nrow_sub,m))
    allocate(lower_original(chunk_size,nrow_sub,m,m), diagonal_original(chunk_size,nrow_sub,m,m), &
             upper_original(chunk_size,nrow_sub,m,m), rhs_original(chunk_size,nrow_sub,m), &
             exact_local(chunk_size,nrow_sub,m))
    allocate(lower_device(chunk_size,nrow_sub,m,m), diagonal_device(chunk_size,nrow_sub,m,m), &
             upper_device(chunk_size,nrow_sub,m,m), rhs_device(chunk_size,nrow_sub,m))

    if (rank == 0) then
        open(newunit=output_unit, file=trim(output_path), status='replace', action='write')
        call write_accuracy_header(output_unit)
    else
        output_unit = -1
    end if

    call btdma_makeplan_gpu_v2(plan, m, chunk_size, nrow_sub, MPI_COMM_WORLD)

    do family = 1, 2
        if (family == 1) then
            family_offset = 0
        else
            family_offset = nsys_family_design
        end if

        do first_system = 1, nsys_per_family, chunk_size
            call generate_local_batch_gpu_layout(m, family, kappa, family_offset + first_system, &
                                                 chunk_size, rank, nrow_sub, lower_original, &
                                                 diagonal_original, upper_original, rhs_original, exact_local)
            lower = lower_original
            diagonal = diagonal_original
            upper = upper_original
            rhs = rhs_original
            lower_device = lower
            diagonal_device = diagonal
            upper_device = upper
            rhs_device = rhs

            call MPI_Barrier(MPI_COMM_WORLD, ierr)
            call btdma_many_mpi_gpu_v2(lower_device, diagonal_device, upper_device, rhs_device, &
                                       m, chunk_size, nrow_sub, plan)
            cuda_status = cudaDeviceSynchronize()
            rhs = rhs_device

            call measure_distributed_gpu(m, log10_k, family, 2 * nsys_per_family, &
                                         family_offset + first_system, chunk_size, rank, nprocs, nrow_sub, &
                                         lower_original, diagonal_original, upper_original, rhs_original, &
                                         exact_local, rhs, cuda_status, output_unit)
            if (rank == 0) flush(output_unit)
        end do
    end do

    call btdma_cleanplan_gpu_v2(plan)
    if (rank == 0) close(output_unit)
    deallocate(lower, diagonal, upper, rhs, lower_original, diagonal_original, upper_original, rhs_original, exact_local)
    deallocate(lower_device, diagonal_device, upper_device, rhs_device)
    call MPI_Comm_free(local_comm, ierr)
    call MPI_Finalize(ierr)

contains

    subroutine parse_arguments(block_size, condition_exponent, family_count, chunk, path)
        integer, intent(out) :: block_size, condition_exponent, family_count, chunk
        character(len=*), intent(out) :: path
        character(len=64) :: argument

        if (command_argument_count() /= 5) then
            if (rank == 0) write(*,'(A)') &
                'Usage: accuracy_gpu M LOG10_K N_SYSTEMS_PER_FAMILY CHUNK_SIZE OUTPUT.csv'
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

    subroutine generate_local_batch_gpu_layout(block_size, conditioning_family, target_condition, first_id, &
                                                batch_size, process_rank, local_rows, a, b, c, d, x)
        integer, intent(in) :: block_size, conditioning_family, first_id, batch_size, process_rank, local_rows
        real(dp), intent(in) :: target_condition
        real(dp), intent(out) :: a(batch_size,local_rows,block_size,block_size)
        real(dp), intent(out) :: b(batch_size,local_rows,block_size,block_size)
        real(dp), intent(out) :: c(batch_size,local_rows,block_size,block_size)
        real(dp), intent(out) :: d(batch_size,local_rows,block_size)
        real(dp), intent(out) :: x(batch_size,local_rows,block_size)
        integer :: local_row, batch_system, global_row, canonical_id
        real(dp) :: lower_block(block_size,block_size), diagonal_block(block_size,block_size)
        real(dp) :: upper_block(block_size,block_size), rhs_vector(block_size), exact_solution(block_size)

        do local_row = 1, local_rows
            global_row = process_rank * local_rows + local_row
            do batch_system = 1, batch_size
                canonical_id = first_id + batch_system - 1
                call build_blocks(block_size, n_global_design, canonical_id, global_row, conditioning_family, &
                                  target_condition, lower_block, diagonal_block, upper_block)
                call build_rhs(block_size, n_global_design, canonical_id, global_row, lower_block, &
                               diagonal_block, upper_block, rhs_vector, exact_solution)
                a(batch_system,local_row,:,:) = lower_block
                b(batch_system,local_row,:,:) = diagonal_block
                c(batch_system,local_row,:,:) = upper_block
                d(batch_system,local_row,:) = rhs_vector
                x(batch_system,local_row,:) = exact_solution
            end do
        end do
    end subroutine generate_local_batch_gpu_layout

    subroutine measure_distributed_gpu(block_size, condition_exponent, conditioning_family, executed_nsys, &
                                       first_id, batch_size, process_rank, process_count, local_rows, &
                                       a, b, c, original_d, exact, computed, cuda_runtime_status, csv_unit)
        integer, intent(in) :: block_size, condition_exponent, conditioning_family, executed_nsys
        integer, intent(in) :: first_id, batch_size, process_rank, process_count, local_rows
        integer, intent(in) :: cuda_runtime_status, csv_unit
        real(dp), intent(in) :: a(batch_size,local_rows,block_size,block_size)
        real(dp), intent(in) :: b(batch_size,local_rows,block_size,block_size)
        real(dp), intent(in) :: c(batch_size,local_rows,block_size,block_size)
        real(dp), intent(in) :: original_d(batch_size,local_rows,block_size)
        real(dp), intent(in) :: exact(batch_size,local_rows,block_size)
        real(dp), intent(in) :: computed(batch_size,local_rows,block_size)

        integer :: left_rank, right_rank, local_row, batch_system, global_row, canonical_id
        integer :: nonfinite_local(batch_size), nonfinite_global(batch_size)
        real(dp) :: sums_local(4,batch_size), sums_global(4,batch_size)
        real(dp) :: send_first(batch_size,block_size), send_last(batch_size,block_size)
        real(dp) :: left_ghost(batch_size,block_size), right_ghost(batch_size,block_size)
        real(dp) :: previous_x(block_size), next_x(block_size), residual(block_size)
        real(dp) :: residual_value, error_value
        logical :: record_nonfinite, record_failed

        left_rank = process_rank - 1
        right_rank = process_rank + 1
        if (left_rank < 0) left_rank = MPI_PROC_NULL
        if (right_rank >= process_count) right_rank = MPI_PROC_NULL
        send_first = computed(:,1,:)
        send_last = computed(:,local_rows,:)
        left_ghost = 0.0_dp
        right_ghost = 0.0_dp

        call MPI_Sendrecv(send_last, batch_size * block_size, MPI_DOUBLE_PRECISION, right_rank, 411, &
                          left_ghost, batch_size * block_size, MPI_DOUBLE_PRECISION, left_rank, 411, &
                          MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
        call MPI_Sendrecv(send_first, batch_size * block_size, MPI_DOUBLE_PRECISION, left_rank, 412, &
                          right_ghost, batch_size * block_size, MPI_DOUBLE_PRECISION, right_rank, 412, &
                          MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)

        sums_local = 0.0_dp
        nonfinite_local = 0
        do local_row = 1, local_rows
            global_row = process_rank * local_rows + local_row
            do batch_system = 1, batch_size
                previous_x = 0.0_dp
                next_x = 0.0_dp
                if (global_row > 1) then
                    if (local_row > 1) then
                        previous_x = computed(batch_system,local_row-1,:)
                    else
                        previous_x = left_ghost(batch_system,:)
                    end if
                end if
                if (global_row < n_global_design) then
                    if (local_row < local_rows) then
                        next_x = computed(batch_system,local_row+1,:)
                    else
                        next_x = right_ghost(batch_system,:)
                    end if
                end if
                residual = matmul(a(batch_system,local_row,:,:), previous_x) &
                         + matmul(b(batch_system,local_row,:,:), computed(batch_system,local_row,:)) &
                         + matmul(c(batch_system,local_row,:,:), next_x) - original_d(batch_system,local_row,:)
                sums_local(1,batch_system) = sums_local(1,batch_system) + sum(residual * residual)
                sums_local(2,batch_system) = sums_local(2,batch_system) &
                                            + sum(original_d(batch_system,local_row,:)**2)
                sums_local(3,batch_system) = sums_local(3,batch_system) &
                                            + sum((computed(batch_system,local_row,:) &
                                            - exact(batch_system,local_row,:))**2)
                sums_local(4,batch_system) = sums_local(4,batch_system) &
                                            + sum(exact(batch_system,local_row,:)**2)
                if (vector_has_nonfinite(computed(batch_system,local_row,:))) nonfinite_local(batch_system) = 1
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
                record_failed = cuda_runtime_status /= cudaSuccess .or. record_nonfinite &
                                .or. residual_value > residual_failure_tolerance
                call write_accuracy_record(csv_unit, block_size, condition_exponent, conditioning_family, &
                                           executed_nsys, canonical_id, 'gpu-pascal-btdma', process_count, &
                                           process_count, residual_value, error_value, -999, .false., &
                                           cuda_runtime_status, record_nonfinite, record_failed)
            end do
        end if
    end subroutine measure_distributed_gpu

end program accuracy_gpu
