module gpu_benchmark_support
    use mpi
    use cudafor
    use benchmark_support, only: dp
    implicit none

contains

    subroutine select_local_gpu(comm)
        integer, intent(in) :: comm
        integer :: ierr, myrank, local_rank, device_count, device_id

        call MPI_Comm_rank(comm, myrank, ierr)
        local_rank = environment_local_rank(myrank)

        ierr = cudaGetDeviceCount(device_count)
        if (ierr /= 0 .or. device_count < 1) then
            call abort_gpu("cudaGetDeviceCount failed or no visible GPU was found", comm)
        end if

        device_id = mod(local_rank, device_count)
        ierr = cudaSetDevice(device_id)
        if (ierr /= 0) call abort_gpu("cudaSetDevice failed", comm)
    end subroutine select_local_gpu

    subroutine synchronize_gpu(comm)
        integer, intent(in) :: comm
        integer :: ierr
        ierr = cudaDeviceSynchronize()
        if (ierr /= 0) call abort_gpu("cudaDeviceSynchronize failed", comm)
    end subroutine synchronize_gpu

    subroutine initialize_gpu_legacy(a, b, c, d, m, myrank, nprocs, comm)
        real(dp), device, intent(out) :: a(:,:,:,:), b(:,:,:,:), c(:,:,:,:), d(:,:,:)
        integer, intent(in) :: m, myrank, nprocs, comm
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
        call synchronize_gpu(comm)
    end subroutine initialize_gpu_legacy

    subroutine initialize_gpu_v2(a, b, c, d, m, myrank, nprocs, comm)
        real(dp), device, intent(out) :: a(:,:,:,:), b(:,:,:,:), c(:,:,:,:), d(:,:,:)
        integer, intent(in) :: m, myrank, nprocs, comm
        integer :: i

        a = 0.0_dp
        b = 0.0_dp
        c = 0.0_dp
        d = 1.0_dp
        do i = 1, m
            a(:,:,i,i) = -0.25_dp
            b(:,:,i,i) =  2.00_dp
            c(:,:,i,i) = -0.25_dp
        end do
        if (myrank == 0) a(:,1,:,:) = 0.0_dp
        if (myrank == nprocs - 1) c(:,size(c,2),:,:) = 0.0_dp
        call synchronize_gpu(comm)
    end subroutine initialize_gpu_v2

    integer function environment_local_rank(fallback_rank) result(local_rank)
        integer, intent(in) :: fallback_rank
        character(len=32) :: value
        integer :: status, ios

        local_rank = fallback_rank
        call get_environment_variable("SLURM_LOCALID", value, status=status)
        if (status == 0 .and. len_trim(value) > 0) then
            read(value, *, iostat=ios) local_rank
            if (ios == 0) return
        end if

        call get_environment_variable("OMPI_COMM_WORLD_LOCAL_RANK", value, status=status)
        if (status == 0 .and. len_trim(value) > 0) then
            read(value, *, iostat=ios) local_rank
            if (ios == 0) return
        end if

        call get_environment_variable("MV2_COMM_WORLD_LOCAL_RANK", value, status=status)
        if (status == 0 .and. len_trim(value) > 0) then
            read(value, *, iostat=ios) local_rank
        end if
    end function environment_local_rank

    subroutine abort_gpu(message, comm)
        character(len=*), intent(in) :: message
        integer, intent(in) :: comm
        integer :: ierr, myrank

        call MPI_Comm_rank(comm, myrank, ierr)
        if (myrank == 0) write(*,'(A)') "ERROR: " // trim(message)
        call MPI_Abort(comm, 3, ierr)
    end subroutine abort_gpu

end module gpu_benchmark_support

