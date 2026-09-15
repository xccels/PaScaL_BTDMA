module benchmark_support
    use iso_fortran_env, only: real64
    use mpi
    implicit none

    integer, parameter :: dp = real64

    type :: benchmark_config
        character(len=16) :: scaling = "strong"
        integer :: logical_p = 2
        integer :: n_global = 2048
        integer :: nsys = 128 * 128
        integer :: m = 8
        integer :: nwarmup = 2
        integer :: nrepeat = 5
    end type benchmark_config

contains

    subroutine read_config(cfg, comm)
        type(benchmark_config), intent(out) :: cfg
        integer, intent(in) :: comm
        character(len=64) :: arg
        integer :: ierr, myrank

        call MPI_Comm_rank(comm, myrank, ierr)
        if (command_argument_count() /= 7) then
            if (myrank == 0) then
                write(*,'(A)') "Usage: <exe> <strong|weak> <P> <N> <nsys> <m> <warmups> <repeats>"
            end if
            call MPI_Abort(comm, 2, ierr)
        end if

        call get_command_argument(1, arg)
        cfg%scaling = adjustl(arg)
        call read_integer_argument(2, "P", cfg%logical_p, comm)
        call read_integer_argument(3, "N", cfg%n_global, comm)
        call read_integer_argument(4, "nsys", cfg%nsys, comm)
        call read_integer_argument(5, "m", cfg%m, comm)
        call read_integer_argument(6, "warmups", cfg%nwarmup, comm)
        call read_integer_argument(7, "repeats", cfg%nrepeat, comm)

        if (trim(cfg%scaling) /= "strong" .and. trim(cfg%scaling) /= "weak") then
            call abort_config("scaling must be strong or weak", comm)
        end if
        if (cfg%logical_p /= 2 .and. cfg%logical_p /= 4 .and. cfg%logical_p /= 8) then
            call abort_config("P must be one of 2, 4, or 8", comm)
        end if
        if (cfg%m /= 2 .and. cfg%m /= 5 .and. cfg%m /= 8) then
            call abort_config("m must be one of 2, 5, or 8", comm)
        end if
        if (cfg%n_global <= 0 .or. cfg%nsys <= 0) then
            call abort_config("N and nsys must be positive", comm)
        end if
        if (cfg%nwarmup < 0 .or. cfg%nrepeat <= 0) then
            call abort_config("warmups must be nonnegative and repeats must be positive", comm)
        end if
    end subroutine read_config

    subroutine validate_mapping(cfg, ranks_per_p, backend, comm)
        type(benchmark_config), intent(in) :: cfg
        integer, intent(in) :: ranks_per_p, comm
        character(len=*), intent(in) :: backend
        integer :: ierr, nprocs, nlocal_min, nsys_local_min
        character(len=160) :: message

        call MPI_Comm_size(comm, nprocs, ierr)
        if (nprocs /= ranks_per_p * cfg%logical_p) then
            write(message,'(A,A,A,I0,A,I0)') trim(backend), &
                " benchmark requires MPI ranks = ", ranks_per_p, " * P; got ", nprocs
            call abort_config(trim(message), comm)
        end if

        nlocal_min = cfg%n_global / nprocs
        nsys_local_min = cfg%nsys / nprocs
        if (nlocal_min < 3) then
            call abort_config("every MPI rank must own at least three block rows", comm)
        end if
        if (nsys_local_min < 1) then
            call abort_config("every MPI rank must own at least one redistributed system", comm)
        end if
    end subroutine validate_mapping

    integer function local_extent(global_size, myrank, nprocs) result(local_size)
        integer, intent(in) :: global_size, myrank, nprocs
        local_size = global_size / nprocs
        if (myrank < mod(global_size, nprocs)) local_size = local_size + 1
    end function local_extent

    subroutine initialize_cpu_system(a, b, c, d, m, myrank, nprocs)
        real(dp), intent(out) :: a(:,:,:,:), b(:,:,:,:), c(:,:,:,:), d(:,:,:)
        integer, intent(in) :: m, myrank, nprocs
        integer :: i

        a = 0.0_dp
        b = 0.0_dp
        c = 0.0_dp
        d = 1.0_dp
        do i = 1, m
            a(i,i,:,:) = -0.25_dp
            b(i,i,:,:) =  2.00_dp
            c(i,i,:,:) = -0.25_dp
        end do
        if (myrank == 0) a(:,:,:,1) = 0.0_dp
        if (myrank == nprocs - 1) c(:,:,:,size(c,4)) = 0.0_dp
    end subroutine initialize_cpu_system

    subroutine print_result_header(myrank)
        integer, intent(in) :: myrank
        if (myrank /= 0) return
        write(*,'(A)') &
            "BTDMA_RESULT_HEADER,case,variant,backend,precision,logical_P,mpi_ranks," // &
            "N_global,Nlocal_min,Nlocal_max,nsys,m,warmups,repeats,repeat,total_s," // &
            "compute_s,communication_s,phase1_s,phase2_s,phase3_s,phase4_s,phase5_s"
    end subroutine print_result_header

    subroutine reduce_and_print(cfg, variant, backend, repeat_index, local_timing, comm)
        type(benchmark_config), intent(in) :: cfg
        character(len=*), intent(in) :: variant, backend
        integer, intent(in) :: repeat_index, comm
        real(dp), intent(in) :: local_timing(8)
        real(dp) :: global_timing(8)
        integer :: ierr, myrank, nprocs, nlocal_min, nlocal_max

        call MPI_Comm_rank(comm, myrank, ierr)
        call MPI_Comm_size(comm, nprocs, ierr)
        call MPI_Reduce(local_timing, global_timing, size(local_timing), &
            MPI_DOUBLE_PRECISION, MPI_MAX, 0, comm, ierr)

        if (myrank == 0) then
            nlocal_min = cfg%n_global / nprocs
            nlocal_max = nlocal_min
            if (mod(cfg%n_global, nprocs) /= 0) nlocal_max = nlocal_max + 1
            write(*,'(A,4(",",A),10(",",I0),8(",",ES24.16E3))') &
                "BTDMA_RESULT", trim(cfg%scaling), trim(variant), trim(backend), "FP64", &
                cfg%logical_p, nprocs, cfg%n_global, nlocal_min, nlocal_max, cfg%nsys, &
                cfg%m, cfg%nwarmup, cfg%nrepeat, repeat_index, global_timing
        end if
    end subroutine reduce_and_print

    subroutine read_integer_argument(index, name, value, comm)
        integer, intent(in) :: index, comm
        character(len=*), intent(in) :: name
        integer, intent(out) :: value
        character(len=64) :: arg
        character(len=128) :: message
        integer :: ios

        call get_command_argument(index, arg)
        read(arg, *, iostat=ios) value
        if (ios /= 0) then
            write(message,'(A,A,A)') "invalid integer for ", trim(name), ": " // trim(arg)
            call abort_config(trim(message), comm)
        end if
    end subroutine read_integer_argument

    subroutine abort_config(message, comm)
        character(len=*), intent(in) :: message
        integer, intent(in) :: comm
        integer :: ierr, myrank

        call MPI_Comm_rank(comm, myrank, ierr)
        if (myrank == 0) write(*,'(A)') "ERROR: " // trim(message)
        call MPI_Abort(comm, 2, ierr)
    end subroutine abort_config

end module benchmark_support

