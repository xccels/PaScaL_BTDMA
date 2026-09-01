module mpiutil
    use mpi
    implicit none
    
    type, public :: AB
        integer :: n(1:3)
        integer, allocatable, dimension(:,:) :: counts_cart, displs_cart
        integer, allocatable, dimension(:) :: counts, displs
        ! integer, device, allocatable, dimension(:) :: d_ccc
    end type AB

    type, public :: a2a_plan
        integer :: myrank, nprocs, mpi_comm
        type(AB) :: A,B
    end type a2a_plan
    
    real*8 :: t0_a=0.d0,t0_b=0.d0
    real*8 :: t1_a=0.d0,t1_b=0.d0
    real*8 :: t2_a=0.d0,t2_b=0.d0
    real*8 :: t3_a=0.d0,t3_b=0.d0
    real*8 :: t4_a=0.d0,t4_b=0.d0
    real*8 :: t5_a=0.d0,t5_b=0.d0
    real*8 :: comm_tp_a=0.d0,comm_tp_b=0.d0
    real*8 :: comm_tc_a=0.d0,comm_tc_b=0.d0
    real*8 :: comm_tu_a=0.d0,comm_tu_b=0.d0
contains
    function mpiutil_para(sta_g, end_g, myrank, nprocs, indx_a, indx_b)result(nsub)
        integer :: sta_g, end_g, myrank, nprocs, indx_a, indx_b
        integer :: nsub

        integer :: n, tmp1, tmp2, aa,bb

        n    = end_g-sta_g+1
        tmp1 = int(n/nprocs)
        tmp2 = mod(n,nprocs)
        
        if ( myrank<tmp2 ) then
            aa = myrank
            bb = 1
        else
            aa = tmp2
            bb = 0
        endif 

        indx_a = myrank*tmp1 + aa + sta_g;
        indx_b = indx_a + tmp1 + bb -1;
        nsub   = tmp1 + bb;
    end function mpiutil_para

    subroutine mpiutil_a2aplan(tmpA,tmpB,plan)
        implicit none
        integer :: tmpA(1:3),tmpB(1:3)
        type(a2a_plan) :: plan

        integer :: i,j,k,ierr     
        
        plan%A%n(1:3)=tmpA(1:3)
        plan%B%n(1:3)=tmpB(1:3)

        allocate(plan%A%counts_cart(0:plan%nprocs-1,0:1), plan%A%displs_cart(0:plan%nprocs-1,0:1))
        allocate(plan%B%counts_cart(0:plan%nprocs-1,0:1), plan%B%displs_cart(0:plan%nprocs-1,0:1))
        allocate(plan%A%counts(0:plan%nprocs-1), plan%A%displs(0:plan%nprocs-1))
        allocate(plan%B%counts(0:plan%nprocs-1), plan%B%displs(0:plan%nprocs-1))

        ! allocate(plan%A%d_ccc(0:plan%nprocs-1))
        ! allocate(plan%B%d_ccc(0:plan%nprocs-1))

        plan%A%counts_cart(:,1) = plan%A%n(3)
        plan%B%counts_cart(:,0) = plan%B%n(2)
        call MPI_ALLGATHER(plan%B%n(2), 1, MPI_INTEGER, plan%A%counts_cart(:,0), 1, MPI_INTEGER, plan%mpi_comm, ierr)
        call MPI_ALLGATHER(plan%A%n(3), 1, MPI_INTEGER, plan%B%counts_cart(:,1), 1, MPI_INTEGER, plan%mpi_comm, ierr)

        do i = 0, plan%nprocs-1
            plan%A%displs_cart(i,0) = sum(plan%A%counts_cart(0:i,0)) - plan%A%counts_cart(i,0) 
            plan%A%displs_cart(i,1) = 0
            plan%B%displs_cart(i,0) = 0
            plan%B%displs_cart(i,1) = sum(plan%B%counts_cart(0:i,1)) - plan%B%counts_cart(i,1) 
        end do

        plan%A%counts(:) = plan%A%n(1)*plan%A%counts_cart(:,0)*plan%A%n(3)
        plan%B%counts(:) = plan%B%n(1)*plan%B%n(2)*plan%B%counts_cart(:,1)
        do i = 0, plan%nprocs-1
            plan%A%displs(i) = sum(plan%A%counts(0:i)) - plan%A%counts(i) 
            plan%B%displs(i) = sum(plan%B%counts(0:i)) - plan%B%counts(i) 
        end do

        ! plan%A%d_ccc(0:plan%nprocs-1) = plan%A%displs(0:plan%nprocs-1)/plan%A%n(1)
        ! plan%B%d_ccc(0:plan%nprocs-1) = plan%B%displs(0:plan%nprocs-1)/plan%B%n(1)


    end subroutine mpiutil_a2aplan

    subroutine mpiutil_a2aplan_clean(plan)
        implicit none
        type(a2a_plan) :: plan

        deallocate(plan%A%counts_cart, plan%A%displs_cart)
        deallocate(plan%B%counts_cart, plan%B%displs_cart)
        deallocate(plan%A%counts, plan%A%displs)
        deallocate(plan%B%counts, plan%B%displs)
        ! deallocate(plan%A%d_ccc)
        ! deallocate(plan%B%d_ccc)
    end subroutine mpiutil_a2aplan_clean

    subroutine mpiutil_pack(arr_in,arr_out,sr,plan)
        implicit none
        type(a2a_plan) :: plan
        type(AB) :: sr
        real*8 :: arr_in (1:sr%n(1),1:sr%n(2),1:sr%n(3))
        real*8 :: arr_out(1:sr%n(1)*sr%n(2)*sr%n(3))

        integer :: i,j,rank,ccc

        call mpiutil_timecheck(comm_tp_a,comm_tp_b,0)
        ccc=0
        do rank = 0, plan%nprocs-1
            do j = 1, sr%counts_cart(rank,1)
            do i = 1, sr%counts_cart(rank,0)
                arr_out(ccc*sr%n(1)+1:(ccc+1)*sr%n(1)) =  &
                arr_in(1:sr%n(1),i+sr%displs_cart(rank,0),j+sr%displs_cart(rank,1))
                ccc=ccc+1
            end do
            end do
        end do
        call mpiutil_timecheck(comm_tp_a,comm_tp_b,1)

    end subroutine mpiutil_pack

    subroutine mpiutil_unpack(arr_in,arr_out,sr,plan)
        implicit none
        type(a2a_plan) :: plan
        type(AB) :: sr
        real*8 :: arr_in (1:sr%n(1),1:sr%n(2),1:sr%n(3))
        real*8 :: arr_out(1:sr%n(1)*sr%n(2)*sr%n(3))

        integer :: i,j,rank,ccc

        call mpiutil_timecheck(comm_tu_a,comm_tu_b,0)
        ccc=0
        do rank = 0, plan%nprocs-1
            do j = 1, sr%counts_cart(rank,1)
            do i = 1, sr%counts_cart(rank,0)
                arr_in(1:sr%n(1),i+sr%displs_cart(rank,0),j+sr%displs_cart(rank,1)) &
                = arr_out(ccc*sr%n(1)+1:(ccc+1)*sr%n(1)) 
                ccc=ccc+1
            end do
            end do
        end do
        call mpiutil_timecheck(comm_tu_a,comm_tu_b,1)
        
    end subroutine mpiutil_unpack
    
       
#ifdef GPU_ENABLED
    subroutine mpiutil_pack_gpu(arr_in,arr_out,sr,plan)
        use cudafor
        implicit none
        type(a2a_plan) :: plan
        type(AB) :: sr
        real*8, device :: arr_in (1:sr%n(1),1:sr%n(2),1:sr%n(3))
        real*8, device :: arr_out(1:sr%n(1)*sr%n(2)*sr%n(3))

        type(dim3) :: threads, blocks

        integer :: i,j,rank,ccc
        call mpiutil_timecheck(comm_tp_a,comm_tp_b,0)
        do rank = 0, plan%nprocs-1
            threads = dim3(32,2,1)
            blocks  = dim3(ceiling(dble( sr%counts_cart(rank,0) )/dble(threads%x))  &
                          ,ceiling(dble( sr%counts_cart(rank,1) )/dble(threads%y)), 1)
            call pack_gpu<<<blocks,threads>>>(arr_out,arr_in,sr%n(1),sr%n(2),sr%n(3)            &
                                             ,sr%counts_cart(rank,0),sr%counts_cart(rank,1)     &
                                             ,sr%displs_cart(rank,0),sr%displs_cart(rank,1),rank)
        end do
        call mpiutil_timecheck(comm_tp_a,comm_tp_b,1)
    end subroutine mpiutil_pack_gpu

    subroutine mpiutil_unpack_gpu(arr_in,arr_out,sr,plan)
        use cudafor
        implicit none
        type(a2a_plan) :: plan
        type(AB) :: sr
        real*8, device :: arr_in (1:sr%n(1),1:sr%n(2),1:sr%n(3))
        real*8, device :: arr_out(1:sr%n(1)*sr%n(2)*sr%n(3))

        type(dim3) :: threads, blocks

        integer :: i,j,rank,ccc

        call mpiutil_timecheck(comm_tu_a,comm_tu_b,0)
        do rank = 0, plan%nprocs-1
            threads = dim3(32,2,1)
            blocks  = dim3(ceiling(dble( sr%counts_cart(rank,0) )/dble(threads%x))  &
                          ,ceiling(dble( sr%counts_cart(rank,1) )/dble(threads%y)), 1)
            call unpack_gpu<<<blocks,threads>>>(arr_out,arr_in,sr%n(1),sr%n(2),sr%n(3)            &
                                               ,sr%counts_cart(rank,0),sr%counts_cart(rank,1)     &
                                               ,sr%displs_cart(rank,0),sr%displs_cart(rank,1),rank)
        end do
        call mpiutil_timecheck(comm_tu_a,comm_tu_b,1)

    end subroutine mpiutil_unpack_gpu

    attributes(global) subroutine pack_gpu(arr_out,arr_in,n1,n2,n3,counts_cart0,counts_cart1,displs_cart0,displs_cart1,rank)
        use cudafor
        implicit none
        integer, value :: n1,n2,n3
        integer, value :: counts_cart0,counts_cart1
        integer, value :: displs_cart0,displs_cart1,rank
        real*8, device :: arr_out(1:n1*n2*n3)
        real*8, device :: arr_in(1:n1,1:n2,1:n3)
        
        integer :: i, j, k, ccc

        i = (blockidx%x-1)*blockdim%x + (threadidx%x-1) + 1
        j = (blockidx%y-1)*blockdim%y + (threadidx%y-1) + 1

        if(i <= counts_cart0.and.j <= counts_cart1) then
            ccc = (i-1) + (j-1)*counts_cart0 + rank*counts_cart0*counts_cart1
            arr_out(ccc*n1+1:(ccc+1)*n1) = arr_in(1:n1,i+displs_cart0,j+displs_cart1)
        endif
    end subroutine pack_gpu

    attributes(global) subroutine unpack_gpu(arr_out,arr_in,n1,n2,n3,counts_cart0,counts_cart1,displs_cart0,displs_cart1,rank)
        use cudafor
        implicit none
        integer, value :: n1,n2,n3
        integer, value :: counts_cart0,counts_cart1
        integer, value :: displs_cart0,displs_cart1,rank
        real*8, device :: arr_out(1:n1*n2*n3)
        real*8, device :: arr_in(1:n1,1:n2,1:n3)
        
        integer :: i, j, k, ccc

        i = (blockidx%x-1)*blockdim%x + (threadidx%x-1) + 1
        j = (blockidx%y-1)*blockdim%y + (threadidx%y-1) + 1

        if(i <= counts_cart0.and.j <= counts_cart1) then
            ccc = (i-1) + (j-1)*counts_cart0 + rank*counts_cart0*counts_cart1
            arr_in(1:n1,i+displs_cart0,j+displs_cart1) = arr_out(ccc*n1+1:(ccc+1)*n1) 
        endif
    end subroutine unpack_gpu
#endif

    subroutine mpiutil_timecheck(ta,tb,tick)
#ifdef GPU_ENABLED
        use cudafor
#endif
        implicit none
        integer :: tick
        real*8 :: ta, tb
        integer :: ierr
        
#ifdef GPU_ENABLED
        ierr = CudaDeviceSynchronize()
#endif
        ta = tick*(MPI_WTIME() - ta) + (1-tick)*MPI_WTIME()
        tb = tb + ta*tick
    end subroutine mpiutil_timecheck



end module mpiutil