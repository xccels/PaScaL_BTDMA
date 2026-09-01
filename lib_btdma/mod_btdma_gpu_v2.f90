module mod_btdma_gpu_v2
    ! use mpiutil
    use cudafor
    use mod_cudatools
    implicit none

    private :: mpiutil_para
    private :: mpiutil_a2aplan
    private :: mpiutil_a2aplan_clean
    private :: btdma_many_a2av_forward_gpu_v2
    private :: btdma_many_a2av_backward_gpu_v2
    private :: mpiutil_pack_gpu
    private :: mpiutil_unpack_gpu
    private :: pack_gpu
    private :: unpack_gpu

    type, private :: AB
        integer :: n(1:3)
        integer, allocatable, dimension(:,:) :: counts_cart, displs_cart
        integer, allocatable, dimension(:) :: counts, displs
    end type AB

    type, private :: a2a_plan
        integer :: myrank, nprocs, mpi_comm
        type(AB) :: A,B
    end type a2a_plan

    type, public :: BTDMA_PLAN_gpu_v2
        integer :: nsys_sub
        real*8, device, allocatable, dimension(:,:,:,:) :: rdA,rdB,rdC
        real*8, device, allocatable, dimension(:,:,:,:) :: trA,trB,trC
        real*8, device, allocatable, dimension(:,:,:  ) :: rdD
        real*8, device, allocatable, dimension(:,:,:  ) :: trD
        real*8, device, allocatable, dimension(:) :: bufRD,bufTR
        type(a2a_plan) :: commM,commV
    end type BTDMA_PLAN_gpu_v2

    real*8 :: t0__a=0.d0,t0__b=0.d0
    real*8 :: t1__a=0.d0,t1__b=0.d0
    real*8 :: t2__a=0.d0,t2__b=0.d0
    real*8 :: t3__a=0.d0,t3__b=0.d0
    real*8 :: t4__a=0.d0,t4__b=0.d0
    real*8 :: t5__a=0.d0,t5__b=0.d0
    real*8 :: comm__tp__a=0.d0,comm__tp__b=0.d0
    real*8 :: comm__tc__a=0.d0,comm__tc__b=0.d0
    real*8 :: comm__tu__a=0.d0,comm__tu__b=0.d0
contains

    subroutine btdma_makeplan_gpu_v2(plan,m,nsys,nrow_sub,comm)
        use mpi
        implicit none
        type(BTDMA_PLAN_gpu_v2) :: plan
        integer, intent(in) :: m,nsys,nrow_sub,comm

        integer :: nprocs,myrank,ierr
        integer :: indx_tmpa,indx_tmpb;

        integer :: i
        real*8 :: tmpII(1:8,1:8)

        call MPI_COMM_SIZE(comm, nprocs, ierr)
        call MPI_COMM_RANK(comm, myrank, ierr)

        plan%commM%nprocs = nprocs
        plan%commM%myrank = myrank
        plan%commM%mpi_comm = comm

        plan%commV%nprocs = nprocs
        plan%commV%myrank = myrank
        plan%commV%mpi_comm = comm

        plan%nsys_sub = mpiutil_para(1, nsys, myrank, nprocs, indx_tmpa, indx_tmpb)

        call mpiutil_a2aplan((/nsys,2/),(/plan%nsys_sub,2*nprocs/),m*m,plan%commM)
        call mpiutil_a2aplan((/nsys,2/),(/plan%nsys_sub,2*nprocs/),m*1,plan%commV)

        allocate(plan%rdA(1:nsys,1:2,1:m,1:m))
        allocate(plan%rdB(1:nsys,1:2,1:m,1:m))
        allocate(plan%rdC(1:nsys,1:2,1:m,1:m))
        allocate(plan%rdD(1:nsys,1:2,1:m    ))
        allocate(plan%trA(1:plan%nsys_sub,1:2*nprocs,1:m,1:m))
        allocate(plan%trB(1:plan%nsys_sub,1:2*nprocs,1:m,1:m))
        allocate(plan%trC(1:plan%nsys_sub,1:2*nprocs,1:m,1:m))
        allocate(plan%trD(1:plan%nsys_sub,1:2*nprocs,1:m    ))
        allocate(plan%bufRD(1:m*m*nsys*2))
        allocate(plan%bufTR(1:m*m*plan%nsys_sub*2*nprocs))

    end subroutine btdma_makeplan_gpu_v2

    subroutine btdma_cleanplan_gpu_v2(plan)
        implicit none
        type(BTDMA_PLAN_gpu_v2) :: plan
    
        deallocate(plan%rdA)
        deallocate(plan%rdB)
        deallocate(plan%rdC)
        deallocate(plan%rdD)
        deallocate(plan%trA)
        deallocate(plan%trB)
        deallocate(plan%trC)
        deallocate(plan%trD)
        deallocate(plan%bufRD)
        deallocate(plan%bufTR)

        call mpiutil_a2aplan_clean(plan%commM)
        call mpiutil_a2aplan_clean(plan%commV)
        
    end subroutine btdma_cleanplan_gpu_v2

    subroutine btdma_many_mpi_gpu_v2(A,B,C,D,m,nsys,nrow_sub,plan)
        use mpi
        implicit none
        type(BTDMA_PLAN_gpu_v2) :: plan
        integer, intent(in) :: m,nsys,nrow_sub        
        real*8, device:: A(1:nsys,1:nrow_sub,1:m,1:m)
        real*8, device:: B(1:nsys,1:nrow_sub,1:m,1:m)
        real*8, device:: C(1:nsys,1:nrow_sub,1:m,1:m)
        real*8, device:: D(1:nsys,1:nrow_sub,1:m    )

        type(dim3) :: threads, blocks
        ! integer :: ierr

        threads = dim3(64,1,1)
        blocks  = dim3(ceiling(dble(nsys)/dble(threads%x)),1,1)

        call btdma_timecheck(t1__a,t1__b,0)
        call btdma_many_modi_gpu_v2<<<blocks,threads>>>(A,B,C,D,plan%rdA,plan%rdB,plan%rdC,plan%rdD,m,nsys,nrow_sub)
        call btdma_timecheck(t1__a,t1__b,1)

        call btdma_timecheck(t2__a,t2__b,0)
        call btdma_many_a2av_forward_gpu_v2(plan)
        call btdma_timecheck(t2__a,t2__b,1)

        call btdma_timecheck(t3__a,t3__b,0)
        call btdma_many_gpu_v2<<<blocks,threads>>>(2*plan%commM%nprocs,plan%nsys_sub,m,plan%trA,plan%trB,plan%trC,plan%trD)
        call btdma_timecheck(t3__a,t3__b,1)
        
        call btdma_timecheck(t4__a,t4__b,0)
        call btdma_many_a2av_backward_gpu_v2(plan)
        call btdma_timecheck(t4__a,t4__b,1)

        call btdma_timecheck(t5__a,t5__b,0)
        call btdma_many_update_gpu_v2<<<blocks,threads>>>(A,B,C,D,plan%rdD,m,nsys,nrow_sub)
        call btdma_timecheck(t5__a,t5__b,1)
                
    end subroutine btdma_many_mpi_gpu_v2

    subroutine btdma_many_cycl_mpi_gpu_v2(A,B,C,D,m,nsys,nrow_sub,plan)
        use mpi
        implicit none
        type(BTDMA_PLAN_gpu_v2) :: plan
        integer, intent(in) :: m,nsys,nrow_sub        
        real*8, device:: A(1:nsys,1:nrow_sub,1:m,1:m)
        real*8, device:: B(1:nsys,1:nrow_sub,1:m,1:m)
        real*8, device:: C(1:nsys,1:nrow_sub,1:m,1:m)
        real*8, device:: D(1:nsys,1:nrow_sub,1:m    )

        real*8, device, allocatable, dimension(:,:,:,:) :: reduceE

        type(dim3) :: threads, blocks

        threads = dim3(64,1,1)
        blocks  = dim3(ceiling(dble(nsys)/dble(threads%x)),1,1)

        call btdma_timecheck(t1__a,t1__b,0)
        call btdma_many_modi_gpu_v2<<<blocks,threads>>>(A,B,C,D,plan%rdA,plan%rdB,plan%rdC,plan%rdD,m,nsys,nrow_sub)
        call btdma_timecheck(t1__a,t1__b,1)

        call btdma_timecheck(t2__a,t2__b,0)
        call btdma_many_a2av_forward_gpu_v2(plan)
        call btdma_timecheck(t2__a,t2__b,1)

        call btdma_timecheck(t3__a,t3__b,0)
        allocate(reduceE(1:plan%nsys_sub,1:2*plan%commM%nprocs,1:m,1:m))
        call btdma_many_cycl_gpu_v2<<<blocks,threads>>>(2*plan%commM%nprocs,plan%nsys_sub,m,plan%trA,plan%trB,plan%trC,plan%trD,reduceE)
        deallocate(reduceE)
        call btdma_timecheck(t3__a,t3__b,1)

        call btdma_timecheck(t4__a,t4__b,0)
        call btdma_many_a2av_backward_gpu_v2(plan)
        call btdma_timecheck(t4__a,t4__b,1)

        call btdma_timecheck(t5__a,t5__b,0)
        call btdma_many_update_gpu_v2<<<blocks,threads>>>(A,B,C,D,plan%rdD,m,nsys,nrow_sub)
        call btdma_timecheck(t5__a,t5__b,1)

    end subroutine btdma_many_cycl_mpi_gpu_v2

    attributes(global) subroutine btdma_many_gpu_v2(n,nsys,m,A,B,C,D)
        implicit none 
        integer, value :: n,nsys,m
        real*8, device :: A(1:nsys,1:n,1:m,1:m)
        real*8, device :: B(1:nsys,1:n,1:m,1:m)
        real*8, device :: C(1:nsys,1:n,1:m,1:m)
        real*8, device :: D(1:nsys,1:n,1:m    )

        real*8 :: Rtmp(1:8,1:8)
        real*8 :: Atmp(1:8,1:8)
        real*8 :: Ctmp(1:8,1:8)
        real*8 :: Dtmp(1:8),Dtmp0(1:8)

        integer :: isys
        integer :: i,j,q

        Rtmp(1:8,1:8) = 0.d0
        Atmp(1:8,1:8) = 0.d0
        Ctmp(1:8,1:8) = 0.d0
        Dtmp(1:8) = 0.d0
        Dtmp0(1:8) = 0.d0

        isys = (blockidx%x-1)*blockdim%x + (threadidx%x-1) + 1
        if(isys <= nsys) then

            do j = 1, m
                do i = 1, m
                    Rtmp(i,j) = B(isys,1,i,j)   
                    Ctmp(i,j) = C(isys,1,i,j)
                enddo
                Dtmp(j) = D(isys,1,j)
            enddo
            
            call gesv_mrhs2(m,Rtmp,Ctmp,m,Dtmp)

            do j = 1, m
                do i = 1, m
                    C(isys,1,i,j) = Ctmp(i,j)
                enddo
                D(isys,1,j) = Dtmp(j)
            enddo

            do q=2,n
                !--Rtmp---
                do j = 1, m
                do i = 1, m
                    Atmp(i,j) = A(isys,q  ,i,j)
                    Ctmp(i,j) = C(isys,q-1,i,j)
                enddo
                enddo
                call gemm(Atmp,Ctmp,Rtmp,m)

                do j = 1, m
                do i = 1, m
                    Rtmp(i,j) = B(isys,q,i,j) - Rtmp(i,j)
                enddo
                enddo

                !--Ctmp---
                do j = 1, m
                do i = 1, m
                    Ctmp(i,j) = C(isys,q,i,j)
                enddo
                enddo

                !--Dtmp---
                do i = 1, m
                    Dtmp0(i) = D(isys,q-1,i)
                enddo
                call gemv(Atmp,Dtmp0,Dtmp,m)

                do i = 1, m
                    Dtmp(i) = D(isys,q,i) - Dtmp(i)
                enddo

                call gesv_mrhs2(m,Rtmp,Ctmp,m,Dtmp)

                do j = 1, m
                    do i = 1, m
                        C(isys,q,i,j) = Ctmp(i,j)
                    enddo
                    D(isys,q,j) = Dtmp(j)
                enddo

            enddo

            do q=n-1,1,-1
                !--Ctmp---
                do j = 1, m
                do i = 1, m
                    Ctmp(i,j) = C(isys,q,i,j)
                enddo
                enddo
                !--Dtmp---
                do i = 1, m
                    Dtmp0(i) = D(isys,q+1,i)
                enddo

                call gemv(Ctmp,Dtmp0,Dtmp,m)

                do i = 1, m
                    D(isys,q,i) = D(isys,q,i) - Dtmp(i)
                enddo
            enddo



        endif

    end subroutine btdma_many_gpu_v2

    attributes(global) subroutine btdma_many_cycl_gpu_v2(n,nsys,m,A,B,C,D,E)
        implicit none 
        integer, value :: n,nsys,m
        real*8, device :: A(1:nsys,1:n,1:m,1:m)
        real*8, device :: B(1:nsys,1:n,1:m,1:m)
        real*8, device :: C(1:nsys,1:n,1:m,1:m)
        real*8, device :: D(1:nsys,1:n,1:m    )
        real*8, device :: E(1:nsys,1:n,1:m,1:m)

        real*8 :: Rtmp(1:8,1:8)
        real*8 :: Atmp(1:8,1:8)
        real*8 :: Ctmp(1:8,1:8)
        real*8 :: Dtmp(1:8),Dtmp0(1:8)
        real*8 :: Etmp(1:8,1:8),Etmp0(1:8,1:8)

        integer :: isys
        integer :: i,j,q

        Rtmp(1:8,1:8) = 0.d0
        Atmp(1:8,1:8) = 0.d0
        Ctmp(1:8,1:8) = 0.d0
        Dtmp(1:8) = 0.d0
        Dtmp0(1:8) = 0.d0
        Etmp(1:8,1:8) = 0.d0
        Etmp0(1:8,1:8) = 0.d0

        isys = (blockidx%x-1)*blockdim%x + (threadidx%x-1) + 1
        if(isys <= nsys) then

            ! q = 2
            do j = 1, m
                do i = 1, m
                    Rtmp(i,j) = B(isys,2,i,j)   
                    Ctmp(i,j) = C(isys,2,i,j)
                    Etmp(i,j) =-A(isys,2,i,j)
                enddo
                Dtmp(j) = D(isys,2,j)
            enddo
            
            call gesv_mrhs3(m,Rtmp,Ctmp,m,Etmp,m,Dtmp)

            do j = 1, m
                do i = 1, m
                    C(isys,2,i,j) = Ctmp(i,j)
                    E(isys,2,i,j) = Etmp(i,j)
                enddo
                D(isys,2,j) = Dtmp(j)
            enddo

            do q=3,n-1
                !--Rtmp---
                do j = 1, m
                do i = 1, m
                    Atmp(i,j) = A(isys,q  ,i,j)
                    Ctmp(i,j) = C(isys,q-1,i,j)
                enddo
                enddo
                call gemm(Atmp,Ctmp,Rtmp,m)

                do j = 1, m
                do i = 1, m
                    Rtmp(i,j) = B(isys,q,i,j) - Rtmp(i,j)
                enddo
                enddo

                !--Ctmp---
                do j = 1, m
                do i = 1, m
                    Ctmp(i,j) = C(isys,q,i,j)
                enddo
                enddo
                !--Etmp---
                do j = 1, m
                do i = 1, m
                    Etmp0(i,j) = E(isys,q-1,i,j)
                enddo
                enddo
                call gemm(Atmp,Etmp0,Etmp,m)
                do j = 1, m
                do i = 1, m
                    Etmp(i,j) = - Etmp(i,j)
                enddo
                enddo

                !--Dtmp---
                do i = 1, m
                    Dtmp0(i) = D(isys,q-1,i)
                enddo
                call gemv(Atmp,Dtmp0,Dtmp,m)

                do i = 1, m
                    Dtmp(i) = D(isys,q,i) - Dtmp(i)
                enddo

                call gesv_mrhs3(m,Rtmp,Ctmp,m,Etmp,m,Dtmp)

                do j = 1, m
                    do i = 1, m
                        C(isys,q,i,j) = Ctmp(i,j)
                        E(isys,q,i,j) = Etmp(i,j)
                    enddo
                    D(isys,q,j) = Dtmp(j)
                enddo

            enddo
            q=n
                !--Rtmp---
                do j = 1, m
                do i = 1, m
                    Atmp(i,j) = A(isys,q  ,i,j)
                    Ctmp(i,j) = C(isys,q-1,i,j)
                enddo
                enddo
                call gemm(Atmp,Ctmp,Rtmp,m)

                do j = 1, m
                do i = 1, m
                    Rtmp(i,j) = B(isys,q,i,j) - Rtmp(i,j)
                enddo
                enddo

                !--Ctmp---
                do j = 1, m
                do i = 1, m
                    Ctmp(i,j) = C(isys,q,i,j)
                enddo
                enddo
                !--Etmp---
                do j = 1, m
                do i = 1, m
                    Etmp0(i,j) = E(isys,q-1,i,j)
                enddo
                enddo
                call gemm(Atmp,Etmp0,Etmp,m)
                do j = 1, m
                do i = 1, m
                    Etmp(i,j) = -Ctmp(i,j) - Etmp(i,j)
                enddo
                enddo

                !--Dtmp---
                do i = 1, m
                    Dtmp0(i) = D(isys,q-1,i)
                enddo
                call gemv(Atmp,Dtmp0,Dtmp,m)

                do i = 1, m
                    Dtmp(i) = D(isys,q,i) - Dtmp(i)
                enddo

                call gesv_mrhs3(m,Rtmp,Ctmp,m,Etmp,m,Dtmp)

                do j = 1, m
                    do i = 1, m
                        C(isys,q,i,j) = Ctmp(i,j)
                        E(isys,q,i,j) = Etmp(i,j)
                    enddo
                    D(isys,q,j) = Dtmp(j)
                enddo

            do q=n-1,2,-1
                !--Ctmp---
                do j = 1, m
                do i = 1, m
                    Ctmp(i,j) = C(isys,q,i,j)
                enddo
                enddo
                !--Etmp---
                do j = 1, m
                do i = 1, m
                    Etmp0(i,j) = E(isys,q+1,i,j)
                enddo
                enddo
                !--Dtmp---
                do i = 1, m
                    Dtmp0(i) = D(isys,q+1,i)
                enddo

                call gemm(Ctmp,Etmp0,Etmp,m)
                call gemv(Ctmp,Dtmp0,Dtmp,m)

                do j = 1, m
                    do i = 1, m
                        E(isys,q,i,j) = E(isys,q,i,j) - Etmp(i,j)
                    enddo
                    D(isys,q,j) = D(isys,q,j) - Dtmp(j)
                enddo
            enddo
            
            q=1 
                !--Ctmp---
                do j = 1, m
                do i = 1, m
                    Atmp(i,j) = A(isys,q,i,j)
                    Ctmp(i,j) = C(isys,q,i,j)
                    Rtmp(i,j) = B(isys,q,i,j)
                enddo
                enddo
                !--Etmp---
                do j = 1, m
                do i = 1, m
                    Etmp0(i,j) = E(isys,n,i,j)
                enddo
                enddo
                call gemm(Atmp,Etmp0,Etmp,m)

                do j = 1, m
                do i = 1, m
                    Rtmp(i,j) = Rtmp(i,j) + Etmp(i,j)
                enddo
                enddo
                !--Etmp---
                do j = 1, m
                do i = 1, m
                    Etmp0(i,j) = E(isys,2,i,j)
                enddo
                enddo
                call gemm(Ctmp,Etmp0,Etmp,m)

                do j = 1, m
                do i = 1, m
                    Rtmp(i,j) = Rtmp(i,j) + Etmp(i,j)
                enddo
                enddo

                !--Dtmp---
                do i = 1, m
                    Dtmp0(i) = D(isys,2,i)
                enddo
                call gemv(Ctmp,Dtmp0,Dtmp,m)
                do i = 1, m
                    Etmp(i,1) = D(isys,q,i) - Dtmp(i)
                enddo
                do i = 1, m
                    Dtmp0(i) = D(isys,n,i)
                enddo
                call gemv(Atmp,Dtmp0,Dtmp,m)
                do i = 1, m
                    Dtmp(i) = Etmp(i,1) - Dtmp(i)
                enddo

                call gesv(m,Rtmp,Dtmp)

            do i = 1, m
                Dtmp0(i) = Dtmp(i)
            enddo
            do i = 1, m
                D(isys,1,i) = Dtmp0(i)
            enddo
            do q=2,n
                !--Rtmp---
                do j = 1, m
                do i = 1, m
                    Etmp(i,j) = E(isys,q,i,j)
                enddo
                enddo
                call gemv(Etmp,Dtmp0,Dtmp,m)

                do i = 1, m
                    D(isys,q,i) = D(isys,q,i) + Dtmp(i)
                enddo
            enddo
        endif

    end subroutine btdma_many_cycl_gpu_v2

    attributes(global) subroutine btdma_many_modi_gpu_v2(A,B,C,D,rdA,rdB,rdC,rdD,m,nsys,nrow_sub)
        implicit none
        integer, value :: m,nsys,nrow_sub        
        real*8, device :: A(1:nsys,1:nrow_sub,1:m,1:m),rdA(1:nsys,1:2,1:m,1:m)
        real*8, device :: B(1:nsys,1:nrow_sub,1:m,1:m),rdB(1:nsys,1:2,1:m,1:m)
        real*8, device :: C(1:nsys,1:nrow_sub,1:m,1:m),rdC(1:nsys,1:2,1:m,1:m)
        real*8, device :: D(1:nsys,1:nrow_sub,1:m    ),rdD(1:nsys,1:2,1:m    )

        real*8 :: Rtmp(1:8,1:8)
        real*8 :: Atmp(1:8,1:8)
        real*8 :: Ctmp(1:8,1:8)
        real*8 :: Dtmp(1:8),Dtmp0(1:8)
        real*8 :: Mtmp(1:8,1:8)

        integer :: isys
        integer :: i,j,q

        Rtmp(1:8,1:8) = 0.d0
        Atmp(1:8,1:8) = 0.d0
        Ctmp(1:8,1:8) = 0.d0
        Dtmp(1:8) = 0.d0
        Dtmp0(1:8) = 0.d0
        Mtmp(1:8,1:8) = 0.d0

        isys = (blockidx%x-1)*blockdim%x + (threadidx%x-1) + 1
        if(isys <= nsys) then
            q = 1!****
            do j = 1, m
                do i = 1, m
                    Rtmp(i,j) = B(isys,1,i,j)   
                    Atmp(i,j) = A(isys,1,i,j)   
                    Ctmp(i,j) = C(isys,1,i,j)
                enddo
                Dtmp(j) = D(isys,1,j)
            enddo
            call gesv_mrhs3(m,Rtmp,Atmp,m,Ctmp,m,Dtmp)

            do j = 1, m
                do i = 1, m
                    A(isys,1,i,j) = Atmp(i,j)
                    C(isys,1,i,j) = Ctmp(i,j)
                enddo
                D(isys,1,j) = Dtmp(j)
            enddo
            
            q = 2!****
            do j = 1, m
                do i = 1, m
                    Rtmp(i,j) = B(isys,2,i,j)   
                    Atmp(i,j) = A(isys,2,i,j)   
                    Ctmp(i,j) = C(isys,2,i,j)
                enddo
                Dtmp(j) = D(isys,2,j)
            enddo
            call gesv_mrhs3(m,Rtmp,Atmp,m,Ctmp,m,Dtmp)

            do j = 1, m
                do i = 1, m
                    A(isys,2,i,j) = Atmp(i,j)
                    C(isys,2,i,j) = Ctmp(i,j)
                enddo
                D(isys,2,j) = Dtmp(j)
            enddo

            do q=3,nrow_sub!****
                !--Rtmp---
                do j = 1, m
                do i = 1, m
                    Mtmp(i,j) = A(isys,q  ,i,j)
                    Ctmp(i,j) = C(isys,q-1,i,j)
                enddo
                enddo
                call gemm(Mtmp,Ctmp,Rtmp,m)

                do j = 1, m
                do i = 1, m
                    Rtmp(i,j) = B(isys,q,i,j) - Rtmp(i,j)
                enddo
                enddo

                !--Dtmp---
                do i = 1, m
                    Dtmp0(i) = D(isys,q-1,i)
                enddo
                call gemv(Mtmp,Dtmp0,Dtmp,m)

                do i = 1, m
                    Dtmp(i) = D(isys,q,i) - Dtmp(i)
                enddo

                !--Atmp---
                do j = 1, m
                do i = 1, m
                    Ctmp(i,j) = A(isys,q-1,i,j)
                enddo
                enddo
                call gemm(Mtmp,Ctmp,Atmp,m)
                do j = 1, m
                do i = 1, m
                     Atmp(i,j) = -Atmp(i,j)
                enddo
                enddo

                !--Ctmp---
                do j = 1, m
                do i = 1, m
                    Ctmp(i,j) = C(isys,q,i,j)
                enddo
                enddo
                
                call gesv_mrhs3(m,Rtmp,Atmp,m,Ctmp,m,Dtmp)

                do j = 1, m
                    do i = 1, m
                        A(isys,q,i,j) = Atmp(i,j)
                        C(isys,q,i,j) = Ctmp(i,j)
                    enddo
                    D(isys,q,j) = Dtmp(j)
                enddo

            enddo


            do q=nrow_sub-2, 2, -1!****
                !--Ctmp---
                do j = 1, m
                do i = 1, m
                    Mtmp(i,j) = C(isys,q,i,j)
                enddo
                enddo

                !--Dtmp---
                do i = 1, m
                    Dtmp0(i) = D(isys,q+1,i)
                enddo
                call gemv(Mtmp,Dtmp0,Dtmp,m)

                do i = 1, m
                    D(isys,q,i) = D(isys,q,i) - Dtmp(i)
                enddo

                !--Atmp---
                do j = 1, m
                do i = 1, m
                    Rtmp(i,j) = A(isys,q+1,i,j)
                enddo
                enddo
                call gemm(Mtmp,Rtmp,Atmp,m)
                do j = 1, m
                do i = 1, m 
                    A(isys,q,i,j) = A(isys,q,i,j)-Atmp(i,j)
                enddo
                enddo

                !--Ctmp---
                do j = 1, m
                do i = 1, m
                    Rtmp(i,j) = C(isys,q+1,i,j)
                enddo
                enddo
                call gemm(Mtmp,Rtmp,Ctmp,m)
                do j = 1, m
                do i = 1, m
                     C(isys,q,i,j) = -Ctmp(i,j)
                enddo
                enddo

            enddo

            q = 1!**** 
            !--Rtmp---
            do j = 1, m
            do i = 1, m
                Atmp(i,j) = A(isys,2,i,j)
                Mtmp(i,j) = C(isys,1,i,j)
            enddo
            enddo
            call gemm(Mtmp,Atmp,Rtmp,m)

            ! do j = 1, m
            ! do i = 1, m
            !     Rtmp(i,j) = B(isys,1,i,j) - Rtmp(i,j)
            ! enddo
            ! enddo
            do j = 1, m
            do i = 1, m
                Rtmp(i,j) = - Rtmp(i,j)
            enddo
                Rtmp(j,j) = Rtmp(j,j) +1.d0
            enddo

            !--Dtmp---
            do i = 1, m
                Dtmp0(i) = D(isys,2,i)
            enddo
            call gemv(Mtmp,Dtmp0,Dtmp,m)

            do i = 1, m
                Dtmp(i) = D(isys,1,i) - Dtmp(i)
            enddo

            !--Ctmp---
            do j = 1, m
            do i = 1, m
                Atmp(i,j) = C(isys,2,i,j)
            enddo
            enddo
            call gemm(Mtmp,Atmp,Ctmp,m)
            do j = 1, m
            do i = 1, m
                Ctmp(i,j) = -Ctmp(i,j)
            enddo
            enddo
            
            !--Atmp---
            do j = 1, m
            do i = 1, m
                Atmp(i,j) = A(isys,1,i,j)
            enddo
            enddo

            call gesv_mrhs3(m,Rtmp,Atmp,m,Ctmp,m,Dtmp)

            do j = 1, m
                do i = 1, m
                    A(isys,1,i,j) = Atmp(i,j)
                    C(isys,1,i,j) = Ctmp(i,j)
                enddo
                D(isys,1,j) = Dtmp(j)
            enddo

            do j = 1, m
                do i = 1, m
                    rdA(isys,1,i,j) = A(isys,1       ,i,j)
                    rdA(isys,2,i,j) = A(isys,nrow_sub,i,j)
                    rdC(isys,1,i,j) = C(isys,1       ,i,j)
                    rdC(isys,2,i,j) = C(isys,nrow_sub,i,j)
                    rdB(isys,1,i,j) = 0.d0
                    rdB(isys,2,i,j) = 0.d0
                enddo
                rdB(isys,1,j,j) = 1.d0
                rdB(isys,2,j,j) = 1.d0
                rdD(isys,1,j) = D(isys,1       ,j)
                rdD(isys,2,j) = D(isys,nrow_sub,j)
            enddo

        endif

    end subroutine btdma_many_modi_gpu_v2

    attributes(global) subroutine btdma_many_update_gpu_v2(A,B,C,D,rdD,m,nsys,nrow_sub)
        implicit none
        integer, value :: m,nsys,nrow_sub        
        real*8, device:: A(1:nsys,1:nrow_sub,1:m,1:m)
        real*8, device:: B(1:nsys,1:nrow_sub,1:m,1:m)
        real*8, device:: C(1:nsys,1:nrow_sub,1:m,1:m)
        real*8, device:: D(1:nsys,1:nrow_sub,1:m    ),rdD(1:nsys,1:2,1:m)
    
        integer :: isys
        integer :: i,j,q

        real*8 :: Atmp(1:8,1:8)
        real*8 :: Ctmp(1:8,1:8)
        real*8 :: Dtmp1(1:8),Dtmp2(1:8)
        real*8 :: MDtmp1(1:8)
        real*8 :: MDtmp2(1:8)

        isys = (blockidx%x-1)*blockdim%x + (threadidx%x-1) + 1
        if(isys <= nsys) then

            do i = 1, m
                Dtmp1(i) = rdD(isys,1,i)
                Dtmp2(i) = rdD(isys,2,i)
            end do

            do i = 1, m
                D(isys,1       ,i) = Dtmp1(i)
                D(isys,nrow_sub,i) = Dtmp2(i)
            end do

            do q = 2, nrow_sub-1

                do j = 1, m
                    do i = 1, m
                        Atmp(i,j)=A(isys,q,i,j)
                        Ctmp(i,j)=C(isys,q,i,j)
                    end do
                end do

                call gemv(Atmp,Dtmp1,MDtmp1,m)
                call gemv(Ctmp,Dtmp2,MDtmp2,m)

                do i = 1, m
                    D(isys,q,i) = D(isys,q,i) - MDtmp1(i) - MDtmp2(i)
                    ! D(isys,q,i) = - MDtmp1(i) - MDtmp2(i)
                end do

            end do
        endif

        
    end subroutine btdma_many_update_gpu_v2

    attributes(device) subroutine gesv(m,A,x2)
        implicit none
        integer, value :: m

        real*8 :: A(1:8,1:8)
        real*8 :: x2(1:8)
        
        integer :: i,j,k,irhs
        real*8 :: factor(1:8)

        ! Gaussian elimination
        do k=1,m-1
            do j = k+1, m
                factor(j) = A(j,k) / A(k,k)
                do i = k, m
                    A(j,i) = A(j,i) - factor(j) * A(k,i)
                enddo
            enddo

            !x2
            do i = k+1, m
                x2(i) = x2(i) - factor(i) * x2(k) 
            enddo
        enddo

        ! Back substitution
        x2(m) = x2(m) / A(m,m)
        do i = m-1, 1, -1
            do j = i+1, m
                x2(i) = x2(i) - A(i,j) * x2(j)
            enddo
            x2(i) = x2(i) / A(i,i)
        enddo

    end subroutine gesv

    attributes(device) subroutine gesv_mrhs2(m,A,x1,nrhs1,x2)
        implicit none
        integer, value :: m
        integer, value :: nrhs1

        real*8 :: A(1:8,1:8)
        real*8 :: x1(1:8,1:8)
        real*8 :: x2(1:8)
        
        integer :: i,j,k,irhs
        real*8 :: factor(1:8)

        ! Gaussian elimination
        do k=1,m-1
            do j = k+1, m
                factor(j) = A(j,k) / A(k,k)
                do i = k, m
                    A(j,i) = A(j,i) - factor(j) * A(k,i)
                enddo
            enddo

            !x1
            do irhs = 1, nrhs1
                do i = k+1, m
                    x1(i,irhs) = x1(i,irhs) - factor(i) * x1(k,irhs) 
                enddo
            enddo
            !x2
            do i = k+1, m
                x2(i) = x2(i) - factor(i) * x2(k) 
            enddo
        enddo

        ! Back substitution
        !x1
        do irhs = 1, nrhs1
            x1(m,irhs) = x1(m,irhs) / A(m,m)
            do i = m-1, 1, -1
                do j = i+1, m
                    x1(i,irhs) = x1(i,irhs) - A(i,j) * x1(j,irhs)
                enddo
                x1(i,irhs) = x1(i,irhs) / A(i,i)
            enddo
        enddo
        x2(m) = x2(m) / A(m,m)
        do i = m-1, 1, -1
            do j = i+1, m
                x2(i) = x2(i) - A(i,j) * x2(j)
            enddo
            x2(i) = x2(i) / A(i,i)
        enddo

    end subroutine gesv_mrhs2

    attributes(device) subroutine gemm(A,B,C,m)
        implicit none
        integer, value :: m
        real*8 :: A(1:8,1:8)
        real*8 :: B(1:8,1:8)
        real*8 :: C(1:8,1:8)

        integer :: i,j,l
        real*8 :: sum

        do j = 1, m
            do i = 1, m
                sum = 0.d0
                do l = 1, m
                    sum = sum + A(i,l) * B(l,j)
                enddo
                C(i,j) = sum
            enddo
        enddo

    end subroutine gemm

    attributes(device) subroutine gemv(A,x,b,m)
        implicit none
        integer, value :: m
        real*8 :: A(1:8,1:8)
        real*8 :: x(1:8)
        real*8 :: b(1:8)

        integer :: i,j
        real*8 :: sum

        do i = 1, m
            sum = 0.d0
            do j = 1, m
                sum = sum + A(i,j) * x(j)
            enddo
            b(i) = sum
        enddo

    end subroutine gemv

    attributes(device) subroutine gesv_mrhs3(m,A,x1,nrhs1,x2,nrhs2,x3)
        implicit none
        integer, value :: m
        integer, value :: nrhs1,nrhs2

        real*8 :: A(1:8,1:8)
        real*8 :: x1(1:8,1:8)
        real*8 :: x2(1:8,1:8)
        real*8 :: x3(1:8)
        
        integer :: i,j,k,irhs
        real*8 :: factor(1:8)

        ! Gaussian elimination
        do k=1,m-1
            do j = k+1, m
                factor(j) = A(j,k) / A(k,k)
                do i = k, m
                    A(j,i) = A(j,i) - factor(j) * A(k,i)
                enddo
            enddo

            !x1
            do irhs = 1, nrhs1
                do i = k+1, m
                    x1(i,irhs) = x1(i,irhs) - factor(i) * x1(k,irhs) 
                enddo
            enddo
            !x2
            do irhs = 1, nrhs2
                do i = k+1, m
                    x2(i,irhs) = x2(i,irhs) - factor(i) * x2(k,irhs) 
                enddo
            enddo
            !x3
            do i = k+1, m
                x3(i) = x3(i) - factor(i) * x3(k) 
            enddo
        enddo

        ! Back substitution
        !x1
        do irhs = 1, nrhs1
            x1(m,irhs) = x1(m,irhs) / A(m,m)
            do i = m-1, 1, -1
                do j = i+1, m
                    x1(i,irhs) = x1(i,irhs) - A(i,j) * x1(j,irhs)
                enddo
                x1(i,irhs) = x1(i,irhs) / A(i,i)
            enddo
        enddo
        !x2
        do irhs = 1, nrhs2
            x2(m,irhs) = x2(m,irhs) / A(m,m)
            do i = m-1, 1, -1
                do j = i+1, m
                    x2(i,irhs) = x2(i,irhs) - A(i,j) * x2(j,irhs)
                enddo
                x2(i,irhs) = x2(i,irhs) / A(i,i)
            enddo
        enddo
        !x3
        x3(m) = x3(m) / A(m,m)
        do i = m-1, 1, -1
            do j = i+1, m
                x3(i) = x3(i) - A(i,j) * x3(j)
            enddo
            x3(i) = x3(i) / A(i,i)
        enddo

    end subroutine gesv_mrhs3

    !---tools--------------------------------------------------------------------
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

        subroutine mpiutil_a2aplan(tmpA,tmpB,stride,plan)
            use mpi
            implicit none
            integer :: tmpA(1:2),tmpB(1:2)
            integer :: stride
            type(a2a_plan) :: plan

            integer :: i,j,k,ierr     
            
            plan%A%n(1:2)=tmpA(1:2);plan%A%n(3)=stride
            plan%B%n(1:2)=tmpB(1:2);plan%B%n(3)=stride

            allocate(plan%A%counts_cart(0:plan%nprocs-1,0:1), plan%A%displs_cart(0:plan%nprocs-1,0:1))
            allocate(plan%B%counts_cart(0:plan%nprocs-1,0:1), plan%B%displs_cart(0:plan%nprocs-1,0:1))
            allocate(plan%A%counts(0:plan%nprocs-1), plan%A%displs(0:plan%nprocs-1))
            allocate(plan%B%counts(0:plan%nprocs-1), plan%B%displs(0:plan%nprocs-1))

            ! allocate(plan%A%d_ccc(0:plan%nprocs-1))
            ! allocate(plan%B%d_ccc(0:plan%nprocs-1))

            plan%A%counts_cart(:,1) = tmpA(2)
            plan%B%counts_cart(:,0) = tmpB(1)
            call MPI_ALLGATHER(tmpB(1), 1, MPI_INTEGER, plan%A%counts_cart(:,0), 1, MPI_INTEGER, plan%mpi_comm, ierr)
            call MPI_ALLGATHER(tmpA(2), 1, MPI_INTEGER, plan%B%counts_cart(:,1), 1, MPI_INTEGER, plan%mpi_comm, ierr)

            do i = 0, plan%nprocs-1
                plan%A%displs_cart(i,0) = sum(plan%A%counts_cart(0:i,0)) - plan%A%counts_cart(i,0) 
                plan%A%displs_cart(i,1) = 0
                plan%B%displs_cart(i,0) = 0
                plan%B%displs_cart(i,1) = sum(plan%B%counts_cart(0:i,1)) - plan%B%counts_cart(i,1) 
            end do

            plan%A%counts(:) = stride*plan%A%counts_cart(:,0)*tmpA(2)
            plan%B%counts(:) = stride*tmpB(1)*plan%B%counts_cart(:,1)
            do i = 0, plan%nprocs-1
                plan%A%displs(i) = sum(plan%A%counts(0:i)) - plan%A%counts(i) 
                plan%B%displs(i) = sum(plan%B%counts(0:i)) - plan%B%counts(i) 
            end do

            ! plan%A%d_ccc(0:plan%nprocs-1) = plan%A%displs(0:plan%nprocs-1)/stride
            ! plan%B%d_ccc(0:plan%nprocs-1) = plan%B%displs(0:plan%nprocs-1)/stride


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

        subroutine mpiutil_pack_gpu(arr_in,arr_out,sr,plan)
            use cudafor
            implicit none
            type(a2a_plan) :: plan
            type(AB) :: sr
            real*8, device :: arr_in (1:sr%n(1),1:sr%n(2),1:sr%n(3))
            real*8, device :: arr_out(1:sr%n(1)*sr%n(2)*sr%n(3))

            type(dim3) :: threads, blocks

            integer :: i,j,rank,ccc
            integer :: width

            integer :: ierr

            width = 0
            do rank = 0, plan%nprocs-1
                threads = dim3(32,2,1)
                blocks  = dim3(ceiling(dble( sr%counts_cart(rank,0) )/dble(threads%x))  &
                              ,ceiling(dble( sr%counts_cart(rank,1) )/dble(threads%y)), 1)
                call pack_gpu<<<blocks,threads>>>(arr_out,arr_in,sr%n(1),sr%n(2),sr%n(3)           &
                                                ,sr%counts_cart(rank,0),sr%counts_cart(rank,1)     &
                                                ,sr%displs_cart(rank,0),sr%displs_cart(rank,1),width,rank)
                width = width + sr%counts_cart(rank,0)*sr%counts_cart(rank,1)
            end do

            ierr = cudaDeviceSynchronize()

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
            integer :: width

            integer :: ierr

            width = 0
            do rank = 0, plan%nprocs-1
                threads = dim3(32,2,1)
                blocks  = dim3(ceiling(dble( sr%counts_cart(rank,0) )/dble(threads%x))  &
                              ,ceiling(dble( sr%counts_cart(rank,1) )/dble(threads%y)), 1)
                call unpack_gpu<<<blocks,threads>>>(arr_out,arr_in,sr%n(1),sr%n(2),sr%n(3)         &
                                                ,sr%counts_cart(rank,0),sr%counts_cart(rank,1)     &
                                                ,sr%displs_cart(rank,0),sr%displs_cart(rank,1),width,rank)
                width = width + sr%counts_cart(rank,0)*sr%counts_cart(rank,1)
            end do

            ierr = cudaDeviceSynchronize()
        end subroutine mpiutil_unpack_gpu

        attributes(global) subroutine pack_gpu(arr_out,arr_in,n1,n2,stride,counts_cart0,counts_cart1,displs_cart0,displs_cart1,width,rank)
            use cudafor
            implicit none
            integer, value :: n1,n2,stride,width
            integer, value :: counts_cart0,counts_cart1
            integer, value :: displs_cart0,displs_cart1,rank
            real*8, device :: arr_out(1:n1*n2*stride)
            real*8, device :: arr_in(1:n1,1:n2,1:stride)
            
            integer :: i, j, k, ccc

            i = (blockidx%x-1)*blockdim%x + (threadidx%x-1) + 1
            j = (blockidx%y-1)*blockdim%y + (threadidx%y-1) + 1

            if(i <= counts_cart0 .and. j <= counts_cart1) then
                ccc = (i-1) + (j-1)*counts_cart0 + 1
                do k = 1, stride
                    arr_out(ccc + counts_cart0*counts_cart1*(k-1) + width*stride) = arr_in(i+displs_cart0,j+displs_cart1,k)
                end do
            end if
        end subroutine pack_gpu

        attributes(global) subroutine unpack_gpu(arr_out,arr_in,n1,n2,stride,counts_cart0,counts_cart1,displs_cart0,displs_cart1,width,rank)
            use cudafor
            implicit none
            integer, value :: n1,n2,stride,width
            integer, value :: counts_cart0,counts_cart1
            integer, value :: displs_cart0,displs_cart1,rank
            real*8, device :: arr_out(1:n1*n2*stride)
            real*8, device :: arr_in(1:n1,1:n2,1:stride)
            
            integer :: i, j, k, ccc

            i = (blockidx%x-1)*blockdim%x + (threadidx%x-1) + 1
            j = (blockidx%y-1)*blockdim%y + (threadidx%y-1) + 1

            if(i <= counts_cart0 .and. j <= counts_cart1) then
                ccc = (i-1) + (j-1)*counts_cart0 + 1
                do k = 1, stride
                     arr_in(i+displs_cart0,j+displs_cart1,k) = arr_out(ccc + counts_cart0*counts_cart1*(k-1)  + width*stride)
                end do
            endif
        end subroutine unpack_gpu

        subroutine btdma_many_a2av_forward_gpu_v2(plan)
            use mpi
            implicit none
            type(BTDMA_PLAN_gpu_v2) :: plan
            integer :: ierr


            ! commM,commV
            call mpiutil_pack_gpu(plan%rdA, plan%bufRD, plan%commM%A, plan%commM)
            call MPI_ALLTOALLV(plan%bufRD, plan%commM%A%counts, plan%commM%A%displs, MPI_DOUBLE_PRECISION,  &
                            plan%bufTR, plan%commM%B%counts, plan%commM%B%displs, MPI_DOUBLE_PRECISION,  &
                            plan%commM%mpi_comm, ierr)
            call mpiutil_unpack_gpu(plan%trA, plan%bufTR, plan%commM%B, plan%commM)

            call mpiutil_pack_gpu(plan%rdB, plan%bufRD, plan%commM%A, plan%commM)
            call MPI_ALLTOALLV(plan%bufRD, plan%commM%A%counts, plan%commM%A%displs, MPI_DOUBLE_PRECISION,  &
                            plan%bufTR, plan%commM%B%counts, plan%commM%B%displs, MPI_DOUBLE_PRECISION,  &
                            plan%commM%mpi_comm, ierr)
            call mpiutil_unpack_gpu(plan%trB, plan%bufTR, plan%commM%B, plan%commM)

            call mpiutil_pack_gpu(plan%rdC, plan%bufRD, plan%commM%A, plan%commM)
            call MPI_ALLTOALLV(plan%bufRD, plan%commM%A%counts, plan%commM%A%displs, MPI_DOUBLE_PRECISION,  &
                            plan%bufTR, plan%commM%B%counts, plan%commM%B%displs, MPI_DOUBLE_PRECISION,  &
                            plan%commM%mpi_comm, ierr)
            call mpiutil_unpack_gpu(plan%trC, plan%bufTR, plan%commM%B, plan%commM) 

            call mpiutil_pack_gpu(plan%rdD, plan%bufRD, plan%commV%A, plan%commV)
            call MPI_ALLTOALLV(plan%bufRD, plan%commV%A%counts, plan%commV%A%displs, MPI_DOUBLE_PRECISION,  &
                            plan%bufTR, plan%commV%B%counts, plan%commV%B%displs, MPI_DOUBLE_PRECISION,  &
                            plan%commV%mpi_comm, ierr)
            call mpiutil_unpack_gpu(plan%trD, plan%bufTR, plan%commV%B, plan%commV)


        end subroutine btdma_many_a2av_forward_gpu_v2

        subroutine btdma_many_a2av_backward_gpu_v2(plan)
            use mpi
            implicit none
            type(BTDMA_PLAN_gpu_v2) :: plan
            integer :: ierr
            ! commM,commV
            ! call mpiutil_pack_gpu(plan%trA, plan%bufTR, plan%commM%B, plan%commM)
            ! call MPI_ALLTOALLV(plan%bufTR, plan%commM%B%counts, plan%commM%B%displs, MPI_DOUBLE_PRECISION,  &
            !                 plan%bufRD, plan%commM%A%counts, plan%commM%A%displs, MPI_DOUBLE_PRECISION,  &                           
            !                 plan%commM%mpi_comm, ierr)
            ! call mpiutil_unpack_gpu(plan%rdA, plan%bufRD, plan%commM%A, plan%commM)

            ! call mpiutil_pack_gpu(plan%trB, plan%bufTR, plan%commM%B, plan%commM)
            ! call MPI_ALLTOALLV(plan%bufTR, plan%commM%B%counts, plan%commM%B%displs, MPI_DOUBLE_PRECISION,  &
            !                 plan%bufRD, plan%commM%A%counts, plan%commM%A%displs, MPI_DOUBLE_PRECISION,  &                           
            !                 plan%commM%mpi_comm, ierr)
            ! call mpiutil_unpack_gpu(plan%rdB, plan%bufRD, plan%commM%A, plan%commM)

            ! call mpiutil_pack_gpu(plan%trC, plan%bufTR, plan%commM%B, plan%commM)
            ! call MPI_ALLTOALLV(plan%bufTR, plan%commM%B%counts, plan%commM%B%displs, MPI_DOUBLE_PRECISION,  &
            !                 plan%bufRD, plan%commM%A%counts, plan%commM%A%displs, MPI_DOUBLE_PRECISION,  &                           
            !                 plan%commM%mpi_comm, ierr)
            ! call mpiutil_unpack_gpu(plan%rdC, plan%bufRD, plan%commM%A, plan%commM)

            call mpiutil_pack_gpu(plan%trD, plan%bufTR, plan%commV%B, plan%commV)
            call MPI_ALLTOALLV(plan%bufTR, plan%commV%B%counts, plan%commV%B%displs, MPI_DOUBLE_PRECISION,  &
                               plan%bufRD, plan%commV%A%counts, plan%commV%A%displs, MPI_DOUBLE_PRECISION,  &                           
                               plan%commV%mpi_comm, ierr)
            call mpiutil_unpack_gpu(plan%rdD, plan%bufRD, plan%commV%A, plan%commV)

        end subroutine btdma_many_a2av_backward_gpu_v2


        subroutine btdma_timecheck(ta,tb,tick)
            use mpi
            use cudafor
            implicit none
            integer :: tick
            real*8 :: ta, tb
            integer :: ierr
            
            ierr = CudaDeviceSynchronize()
            ta = tick*(MPI_WTIME() - ta) + (1-tick)*MPI_WTIME()
            tb = tb + ta*tick
        end subroutine btdma_timecheck
    !-----------------------------------------------------------------------------
end module mod_btdma_gpu_v2
