module communication_benchmark_common
    use mpi
    implicit none

    integer, parameter :: nsys_fixed = 128 * 128

contains

    subroutine parse_benchmark_args(n_global, m, warmups, repeats, p_label, comm)
        integer, intent(out) :: n_global, m, warmups, repeats, p_label
        integer, intent(in) :: comm
        integer :: nargs

        nargs = command_argument_count()
        if (nargs > 5) then
            call abort_benchmark('usage: executable [N] [m] [warmups] [repeats] [P_label]', comm)
        end if

        call read_integer_argument(1, 512, n_global, comm)
        call read_integer_argument(2, 2,   m,        comm)
        call read_integer_argument(3, 2,   warmups,  comm)
        call read_integer_argument(4, 5,   repeats,  comm)
        call read_integer_argument(5, 8,   p_label,  comm)

        if (.not. any(n_global == (/512, 1024, 2048, 4096/))) then
            call abort_benchmark('N must be one of 512, 1024, 2048, or 4096', comm)
        end if
        if (.not. any(m == (/2, 5, 8/))) then
            call abort_benchmark('m must be one of 2, 5, or 8', comm)
        end if
        if (warmups < 0) then
            call abort_benchmark('warmups must be non-negative', comm)
        end if
        if (repeats < 1) then
            call abort_benchmark('repeats must be positive', comm)
        end if
        if (p_label /= 8) then
            call abort_benchmark('this experiment fixes the manuscript resource label P at 8', comm)
        end if
    end subroutine parse_benchmark_args

    subroutine require_minimum_local_rows(nrow_sub, comm)
        integer, intent(in) :: nrow_sub, comm
        integer :: ierr, min_nrow_sub

        call MPI_Allreduce(nrow_sub, min_nrow_sub, 1, MPI_INTEGER, MPI_MIN, comm, ierr)
        if (min_nrow_sub < 2) then
            call abort_benchmark('every rank must own at least two rows', comm)
        end if
    end subroutine require_minimum_local_rows

    subroutine print_result_header(rank, timer_scope)
        integer, intent(in) :: rank
        character(len=*), intent(in) :: timer_scope

        if (rank == 0) then
            write(*,'(A)') 'RESULT_HEADER,platform,method,P_label,nprocs,nsys,m,N,repeat,warmups,repeats,' // &
                           'comm_forward_s,comm_backward_s,comm_total_s'
            write(*,'(A)') 'TIMER_SCOPE,' // trim(timer_scope)
        end if
    end subroutine print_result_header

    subroutine reduce_and_print(platform, method, p_label, nprocs, m, n_global, repeat_id, &
                                warmups, repeats, forward_local, backward_local, comm)
        character(len=*), intent(in) :: platform, method
        integer, intent(in) :: p_label, nprocs, m, n_global, repeat_id, warmups, repeats, comm
        real(kind=8), intent(in) :: forward_local, backward_local
        real(kind=8) :: local_values(3), max_values(3)
        integer :: ierr, rank

        local_values(1) = forward_local
        local_values(2) = backward_local
        local_values(3) = forward_local + backward_local

        call MPI_Reduce(local_values, max_values, 3, MPI_DOUBLE_PRECISION, MPI_MAX, 0, comm, ierr)
        call MPI_Comm_rank(comm, rank, ierr)

        if (rank == 0) then
            write(*,'(A,",",A,",",A,8(",",I0),3(",",ES24.16E3))') &
                'RESULT', trim(platform), trim(method), p_label, nprocs, nsys_fixed, m, n_global, &
                repeat_id, warmups, repeats, max_values(1), max_values(2), max_values(3)
        end if
    end subroutine reduce_and_print

    subroutine abort_benchmark(message, comm)
        character(len=*), intent(in) :: message
        integer, intent(in) :: comm
        integer :: ierr, rank

        call MPI_Comm_rank(comm, rank, ierr)
        if (rank == 0) write(*,'(A)') 'ERROR: ' // trim(message)
        call MPI_Abort(comm, 2, ierr)
    end subroutine abort_benchmark

    subroutine read_integer_argument(position, default_value, value, comm)
        integer, intent(in) :: position, default_value, comm
        integer, intent(out) :: value
        character(len=64) :: text
        integer :: ios

        value = default_value
        if (command_argument_count() < position) return

        call get_command_argument(position, text)
        read(text, *, iostat=ios) value
        if (ios /= 0) then
            call abort_benchmark('all command-line arguments must be integers', comm)
        end if
    end subroutine read_integer_argument

end module communication_benchmark_common
