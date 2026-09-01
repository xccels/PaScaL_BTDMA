module throughput_common
    use iso_fortran_env, only: real64
    use mpi
    implicit none

    integer, parameter :: dp = real64

    type :: throughput_config
        integer :: logical_p = 8
        integer :: n_global = 2048
        integer :: nsys = 128 * 128
        integer :: m = 8
        integer :: nwarmup = 2
        integer :: nrepeat = 5
    end type throughput_config

contains

    subroutine read_config(cfg, backend, comm)
        type(throughput_config), intent(out) :: cfg
        character(len=*), intent(in) :: backend
        integer, intent(in) :: comm

        if (command_argument_count() /= 6) then
            call abort_config( &
                "Usage: <exe> <P> <N> <nsys> <m> <warmups> <repeats>", comm)
        end if

        call read_integer_argument(1, "P", cfg%logical_p, comm)
        call read_integer_argument(2, "N", cfg%n_global, comm)
        call read_integer_argument(3, "nsys", cfg%nsys, comm)
        call read_integer_argument(4, "m", cfg%m, comm)
        call read_integer_argument(5, "warmups", cfg%nwarmup, comm)
        call read_integer_argument(6, "repeats", cfg%nrepeat, comm)

        if (cfg%n_global /= 2048) then
            call abort_config("the manuscript throughput case requires N=2048", comm)
        end if
        if (.not. valid_nsys(cfg%nsys)) then
            call abort_config("nsys must be 16^2, 32^2, 64^2, 128^2, or 256^2", comm)
        end if
        if (cfg%m /= 2 .and. cfg%m /= 5 .and. cfg%m /= 8) then
            call abort_config("m must be 2, 5, or 8", comm)
        end if
        if (cfg%nwarmup < 0 .or. cfg%nrepeat <= 0) then
            call abort_config("warmups must be nonnegative and repeats must be positive", comm)
        end if

        select case (trim(backend))
        case ("cpu")
            if (cfg%logical_p /= 8) then
                call abort_config("the CPU throughput panel requires P=8 sockets", comm)
            end if
        case ("gpu")
            if (cfg%logical_p /= 2 .and. cfg%logical_p /= 4 .and. &
                cfg%logical_p /= 8) then
                call abort_config("GPU P must be 2, 4, or 8", comm)
            end if
            if (cfg%logical_p /= 8 .and. cfg%m /= 8) then
                call abort_config("GPU P=2 and P=4 are used only for m=8", comm)
            end if
        case default
            call abort_config("backend must be cpu or gpu", comm)
        end select
    end subroutine read_config

    subroutine validate_mapping(cfg, backend, comm)
        type(throughput_config), intent(in) :: cfg
        character(len=*), intent(in) :: backend
        integer, intent(in) :: comm
        integer :: ierr, nprocs, expected_ranks
        character(len=160) :: message

        call MPI_Comm_size(comm, nprocs, ierr)
        if (trim(backend) == "cpu") then
            expected_ranks = 24 * cfg%logical_p
        else
            expected_ranks = cfg%logical_p
        end if

        if (nprocs /= expected_ranks) then
            write(message,'(A,A,I0,A,I0,A,I0)') trim(backend), &
                " requires ", expected_ranks, " MPI ranks for P=", cfg%logical_p, &
                "; received ", nprocs
            call abort_config(trim(message), comm)
        end if
        if (cfg%n_global / nprocs < 3) then
            call abort_config("every rank must own at least three block rows", comm)
        end if
        if (cfg%nsys < nprocs) then
            call abort_config("nsys must be at least the MPI rank count", comm)
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
            "THROUGHPUT_RESULT_HEADER,method,backend,precision,logical_P," // &
            "mpi_ranks,N_global,Nlocal_min,Nlocal_max,nsys,m,warmups,repeats," // &
            "repeat,total_s,compute_s,communication_s,phase1_s,phase2_s," // &
            "phase3_s,phase4_s,phase5_s"
    end subroutine print_result_header

    subroutine reduce_and_print(cfg, backend, repeat_index, local_timing, comm)
        type(throughput_config), intent(in) :: cfg
        character(len=*), intent(in) :: backend
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
            write(*,'(A,3(",",A),10(",",I0),8(",",ES24.16E3))') &
                "THROUGHPUT_RESULT", "pascal_btdma", trim(backend), "FP64", &
                cfg%logical_p, nprocs, cfg%n_global, nlocal_min, nlocal_max, &
                cfg%nsys, cfg%m, cfg%nwarmup, cfg%nrepeat, repeat_index, global_timing
        end if
    end subroutine reduce_and_print

    logical function valid_nsys(nsys)
        integer, intent(in) :: nsys

        valid_nsys = nsys == 16*16 .or. nsys == 32*32 .or. &
                     nsys == 64*64 .or. nsys == 128*128 .or. &
                     nsys == 256*256
    end function valid_nsys

    subroutine read_integer_argument(index, name, value, comm)
        integer, intent(in) :: index, comm
        character(len=*), intent(in) :: name
        integer, intent(out) :: value
        character(len=64) :: argument
        character(len=128) :: message
        integer :: ios

        call get_command_argument(index, argument)
        read(argument, *, iostat=ios) value
        if (ios /= 0) then
            write(message,'(A,A,A)') "invalid integer for ", trim(name), &
                ": " // trim(argument)
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

end module throughput_common
