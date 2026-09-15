module mod_btdma_gpu
    use mpiutil
    use cudafor
    use mod_cudatools
    implicit none

    real*8, device :: II(1:32,1:32)

    type, public :: BTDMA_PLAN_gpu
        integer :: nsys_sub
        real*8, device, allocatable, dimension(:,:,:,:) :: rdA,rdB,rdC
        real*8, device, allocatable, dimension(:,:,:,:) :: trA,trB,trC
        real*8, device, allocatable, dimension(:  ,:,:) :: rdD
        real*8, device, allocatable, dimension(:  ,:,:) :: trD
        real*8, device, allocatable, dimension(:) :: bufRD,bufTR
        type(a2a_plan) :: commM,commV
    end type BTDMA_PLAN_gpu

contains

    subroutine btdma_makeplan_gpu(plan,m,nsys,nrow_sub,comm)
        use mpi
        implicit none
        type(BTDMA_PLAN_gpu) :: plan
        integer, intent(in) :: m,nsys,nrow_sub,comm

        integer :: nprocs,myrank,ierr
        integer :: indx_tmpa,indx_tmpb;

        integer :: i
        real*8 :: tmpII(1:32,1:32)

        call MPI_COMM_SIZE(comm, nprocs, ierr)
        call MPI_COMM_RANK(comm, myrank, ierr)

        plan%commM%nprocs = nprocs
        plan%commM%myrank = myrank
        plan%commM%mpi_comm = comm

        plan%commV%nprocs = nprocs
        plan%commV%myrank = myrank
        plan%commV%mpi_comm = comm

        plan%nsys_sub = mpiutil_para(1, nsys, myrank, nprocs, indx_tmpa, indx_tmpb)

        call mpiutil_a2aplan((/m*m,nsys,2/),(/m*m,plan%nsys_sub,2*nprocs/),plan%commM)
        call mpiutil_a2aplan((/m*1,nsys,2/),(/m*1,plan%nsys_sub,2*nprocs/),plan%commV)

        allocate(plan%rdA(1:m,1:m,1:nsys,1:2))
        allocate(plan%rdB(1:m,1:m,1:nsys,1:2))
        allocate(plan%rdC(1:m,1:m,1:nsys,1:2))
        allocate(plan%rdD(1:m,    1:nsys,1:2))
        allocate(plan%trA(1:m,1:m,1:plan%nsys_sub,1:2*nprocs))
        allocate(plan%trB(1:m,1:m,1:plan%nsys_sub,1:2*nprocs))
        allocate(plan%trC(1:m,1:m,1:plan%nsys_sub,1:2*nprocs))
        allocate(plan%trD(1:m,    1:plan%nsys_sub,1:2*nprocs))
        allocate(plan%bufRD(1:m*m*nsys*2))
        allocate(plan%bufTR(1:m*m*plan%nsys_sub*2*nprocs))

        tmpII = 0.d0
        do i = 1, 32
            tmpII(i,i) = 1.d0
        end do
        II = tmpII

    end subroutine btdma_makeplan_gpu

    subroutine btdma_cleanplan_gpu(plan)
        implicit none
        type(BTDMA_PLAN_gpu) :: plan
    
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
        
    end subroutine btdma_cleanplan_gpu

    subroutine btdma_many_gpu(n,nsys,m,A,B,C,D)
        implicit none 
        integer, value :: n,nsys,m
        real*8, device :: A(1:m,1:m,1:nsys,1:n)
        real*8, device :: B(1:m,1:m,1:nsys,1:n)
        real*8, device :: C(1:m,1:m,1:nsys,1:n)
        real*8, device :: D(1:m,    1:nsys,1:n)
        
        integer :: isys
        integer :: i,j,q
        integer :: optimal_thread
    
        type(dim3) :: threads, blocks

        !optimal_thread =ceiling((16384.d0/dble(8*m+9))/32.d0)*32
        optimal_thread = 64
        threads = dim3(optimal_thread,1,1)
        blocks  = dim3(ceiling(dble(nsys)/dble(threads%x)),1,1)

        q=1
        call gesv_gpu_batch_multi2<<<blocks,threads>>>(B(1,1, 1,q), m, D(1,   1,q), 1, C(1,1, 1,q), m, nsys)
        
        do q = 2, n
            call gemm_gpu_batch<<<blocks,threads>>>(A(1,1, 1,q), m, C(1,1, 1,q-1), B(1,1, 1,q), m, 1, -1, nsys)
            call gemm_gpu_batch<<<blocks,threads>>>(A(1,1, 1,q), m, D(1,   1,q-1), D(1,   1,q), 1, 1, -1, nsys)

            call gesv_gpu_batch_multi2<<<blocks,threads>>>(B(1,1, 1,q), m, D(1,   1,q), 1, C(1,1, 1,q), m, nsys)
        end do

        do q = n-1, 1, -1
            call gemm_gpu_batch<<<blocks,threads>>>(C(1,1, 1,q), m, D(1,  1,q+1), D(1,  1,q), 1, 1, -1, nsys)
        end do

    end subroutine btdma_many_gpu

    subroutine btdma_many_cycl_gpu(n,nsys,m,A,B,C,D)
        implicit none 
        integer, value :: n,nsys,m
        real*8, device :: A(1:m,1:m,1:nsys,1:n)
        real*8, device :: B(1:m,1:m,1:nsys,1:n)
        real*8, device :: C(1:m,1:m,1:nsys,1:n)
        real*8, device :: D(1:m,    1:nsys,1:n)
        
        integer :: isys
        integer :: i,j,q
        integer :: optimal_thread
    
        type(dim3) :: threads, blocks

        real*8, device, allocatable, dimension(:,:,:,:) :: E

        allocate(E(m,m,nsys,n))
        threads = dim3(32,32,1)
        blocks  = dim3(ceiling(dble(nsys)/dble(threads%x)),ceiling(dble(n)/dble(threads%y)),1)
        call init_array<<<blocks, threads>>>(E, m, m, nsys, n, 0.0d0)

        ! optimal_thread =ceiling((16384.d0/dble(8*m+9))/32.d0)*32
        optimal_thread = 64
        threads = dim3(optimal_thread,1,1)
        blocks  = dim3(ceiling(dble(nsys)/dble(threads%x)),1,1)
        
        q=1
        call replace_gpu_batch<<<blocks,threads>>>(A(1,1,1,2),E(1,1,1,2),m,m,nsys,II,-1,0)
        call replace_gpu_batch<<<blocks,threads>>>(C(1,1,1,n),E(1,1,1,n),m,m,nsys,II,-1,0)

        q=2
        call gesv_gpu_batch_multi3<<<blocks,threads>>>(B(1,1,1,q), m, D(1,1,q), 1, C(1,1,1,q), m, E(1,1,1,q), m, nsys)

        do q = 3, n
            call gemm_gpu_batch<<<blocks,threads>>>(A(1,1,1,q), m, C(1,1,1,q-1), B(1,1,1,q), m, 1, -1, nsys)
            call gemm_gpu_batch<<<blocks,threads>>>(A(1,1,1,q), m, D(1,  1,q-1), D(1,  1,q), 1, 1, -1, nsys)
            call gemm_gpu_batch<<<blocks,threads>>>(A(1,1,1,q), m, E(1,1,1,q-1), E(1,1,1,q), m, 1, -1, nsys)
            call gesv_gpu_batch_multi3<<<blocks,threads>>>(B(1,1,1,q), m, D(1,1,q), 1, C(1,1,1,q), m, E(1,1,1,q), m, nsys)
        end do

        do q = n-1, 2, -1
            call gemm_gpu_batch<<<blocks,threads>>>(C(1,1,1,q), m, D(1,  1,q+1), D(1,  1,q), 1, 1, -1, nsys)
            call gemm_gpu_batch<<<blocks,threads>>>(C(1,1,1,q), m, E(1,1,1,q+1), E(1,1,1,q), m, 1, -1, nsys)
        end do

        q=1
        call gemm_gpu_batch<<<blocks,threads>>>(A(1,1,1,q), m, E(1,1,1,n), B(1,1,1,q), m, 1, 1, nsys)
        call gemm_gpu_batch<<<blocks,threads>>>(C(1,1,1,q), m, E(1,1,1,2), B(1,1,1,q), m, 1, 1, nsys)

        call gemm_gpu_batch<<<blocks,threads>>>(A(1,1,1,q), m, D(1,  1,n), D(1,  1,q), 1, 1,-1, nsys)
        call gemm_gpu_batch<<<blocks,threads>>>(C(1,1,1,q), m, D(1,  1,2), D(1,  1,q), 1, 1,-1, nsys)

        call gesv_gpu_batch_single<<<blocks,threads>>>(B(1,1,1,q), m, D(1,  1,q), 1, nsys)

        do q = 2, n
            call gemm_gpu_batch<<<blocks,threads>>>(E(1,1,1,q), m, D(1,1,1), D(1,1,q), 1, 1, 1, nsys)
        end do

        deallocate(E)
    end subroutine btdma_many_cycl_gpu

    subroutine btdma_many_mpi_gpu(A,B,C,D,m,nsys,nrow_sub,plan)
        type(BTDMA_PLAN_gpu) :: plan
        integer, intent(in) :: m,nsys,nrow_sub        
        real*8, device:: A(1:m,1:m,1:nsys,1:nrow_sub)
        real*8, device:: B(1:m,1:m,1:nsys,1:nrow_sub)
        real*8, device:: C(1:m,1:m,1:nsys,1:nrow_sub)
        real*8, device:: D(1:m,    1:nsys,1:nrow_sub)

        call mpiutil_timecheck(t1_a,t1_b,0)
        call btdma_many_modi_gpu(A,B,C,D,plan%rdA,plan%rdB,plan%rdC,plan%rdD,m,nsys,nrow_sub,plan)
        call mpiutil_timecheck(t1_a,t1_b,1)

        call mpiutil_timecheck(t2_a,t2_b,0)
        call btdma_many_a2av_forward_gpu(plan)
        call mpiutil_timecheck(t2_a,t2_b,1)

        call mpiutil_timecheck(t3_a,t3_b,0)
        call btdma_many_gpu(2*plan%commM%nprocs,plan%nsys_sub,m,plan%trA,plan%trB,plan%trC,plan%trD)
        call mpiutil_timecheck(t3_a,t3_b,1)

        call mpiutil_timecheck(t4_a,t4_b,0)
        call btdma_many_a2av_backward_gpu(plan)
        call mpiutil_timecheck(t4_a,t4_b,1)

        call mpiutil_timecheck(t5_a,t5_b,0)
        call btdma_many_update_gpu(A,B,C,D,plan%rdD,m,nsys,nrow_sub,plan)
        call mpiutil_timecheck(t5_a,t5_b,1)

    end subroutine btdma_many_mpi_gpu

    subroutine btdma_many_cycl_mpi_gpu(A,B,C,D,m,nsys,nrow_sub,plan)
        type(BTDMA_PLAN_gpu) :: plan
        integer, intent(in) :: m,nsys,nrow_sub        
        real*8, device:: A(1:m,1:m,1:nsys,1:nrow_sub)
        real*8, device:: B(1:m,1:m,1:nsys,1:nrow_sub)
        real*8, device:: C(1:m,1:m,1:nsys,1:nrow_sub)
        real*8, device:: D(1:m,    1:nsys,1:nrow_sub)

        call mpiutil_timecheck(t1_a,t1_b,0)
        call btdma_many_modi_gpu(A,B,C,D,plan%rdA,plan%rdB,plan%rdC,plan%rdD,m,nsys,nrow_sub,plan)
        call mpiutil_timecheck(t1_a,t1_b,1)

        call mpiutil_timecheck(t2_a,t2_b,0)
        call btdma_many_a2av_forward_gpu(plan)
        call mpiutil_timecheck(t2_a,t2_b,1)

        call mpiutil_timecheck(t3_a,t3_b,0)
        call btdma_many_cycl_gpu(2*plan%commM%nprocs,plan%nsys_sub,m,plan%trA,plan%trB,plan%trC,plan%trD)
        call mpiutil_timecheck(t3_a,t3_b,1)

        call mpiutil_timecheck(t4_a,t4_b,0)
        call btdma_many_a2av_backward_gpu(plan)
        call mpiutil_timecheck(t4_a,t4_b,1)

        call mpiutil_timecheck(t5_a,t5_b,0)
        call btdma_many_update_gpu(A,B,C,D,plan%rdD,m,nsys,nrow_sub,plan)
        call mpiutil_timecheck(t5_a,t5_b,1)

    end subroutine btdma_many_cycl_mpi_gpu

    subroutine btdma_many_modi_gpu(A,B,C,D,rdA,rdB,rdC,rdD,m,nsys,nrow_sub,plan)
        implicit none
        type(BTDMA_PLAN_gpu) :: plan
        integer, intent(in) :: m,nsys,nrow_sub        
        real*8, device :: A(1:m,1:m,1:nsys,1:nrow_sub),rdA(1:m,1:m,1:nsys,1:2)
        real*8, device :: B(1:m,1:m,1:nsys,1:nrow_sub),rdB(1:m,1:m,1:nsys,1:2)
        real*8, device :: C(1:m,1:m,1:nsys,1:nrow_sub),rdC(1:m,1:m,1:nsys,1:2)
        real*8, device :: D(1:m,    1:nsys,1:nrow_sub),rdD(1:m,    1:nsys,1:2)

        integer :: isys
        integer :: i,j,q
        integer :: optimal_thread
    
        type(dim3) :: threads, blocks

        ! optimal_thread =ceiling((16384.d0/dble(8*m+9))/32.d0)*32
        optimal_thread = 64
        threads = dim3(optimal_thread,1,1)
        blocks  = dim3(ceiling(dble(nsys)/dble(threads%x)),1,1)


        q=1
        call gesv_gpu_batch_multi3<<<blocks,threads>>>(B(1,1, 1,q), m, D(1,1,q), 1, C(1,1, 1,q), m, A(1,1, 1,q), m, nsys)
        
        q=2
        call gesv_gpu_batch_multi3<<<blocks,threads>>>(B(1,1, 1,q), m, D(1,1,q), 1, C(1,1, 1,q), m, A(1,1, 1,q), m, nsys)

        do q = 3, nrow_sub            
            call gemm_gpu_batch<<<blocks,threads>>>(A(1,1, 1,q), m, C(1,1, 1,q-1), B(1,1, 1,q), m, 1, -1, nsys)
            call gemm_gpu_batch<<<blocks,threads>>>(A(1,1, 1,q), m, D(1,   1,q-1), D(1,   1,q), 1, 1, -1, nsys)
            call gemm_gpu_batch<<<blocks,threads>>>(A(1,1, 1,q), m, A(1,1, 1,q-1), A(1,1, 1,q), m, 0, -1, nsys)
            call gesv_gpu_batch_multi3<<<blocks,threads>>>(B(1,1, 1,q), m, D(1,1,q), 1, C(1,1, 1,q), m, A(1,1, 1,q), m, nsys)
        end do
        
        do q = nrow_sub-2, 2, -1
            call gemm_gpu_batch<<<blocks,threads>>>(C(1,1, 1,q), m, D(1,   1,q+1), D(1,   1,q), 1, 1, -1, nsys)
            call gemm_gpu_batch<<<blocks,threads>>>(C(1,1, 1,q), m, A(1,1, 1,q+1), A(1,1, 1,q), m, 1, -1, nsys)
            call gemm_gpu_batch<<<blocks,threads>>>(C(1,1, 1,q), m, C(1,1, 1,q+1), C(1,1, 1,q), m, 0, -1, nsys)
        end do

        q=1
        call gemm_gpu_batch_delta<<<blocks,threads>>>(C(1,1, 1,q), A(1,1, 1,q+1), B(1,1, 1,q), II, m, -1, nsys)
        call gemm_gpu_batch<<<blocks,threads>>>(C(1,1, 1,q), m, D(1,   1,q+1), D(1,   1,q), 1, 1, -1, nsys)
        call gemm_gpu_batch<<<blocks,threads>>>(C(1,1, 1,q), m, C(1,1, 1,q+1), C(1,1, 1,q), m, 0, -1, nsys)
        call gesv_gpu_batch_multi3<<<blocks,threads>>>(B(1,1, 1,q), m, D(1,1,q), 1, C(1,1, 1,q), m, A(1,1, 1,q), m, nsys)

        !! reduced
        call replace_gpu_batch<<<blocks,threads>>>(A(1,1,1,1       ),rdA(1,1,1,1),m,m,nsys,II,1,0)
        call replace_gpu_batch<<<blocks,threads>>>(A(1,1,1,nrow_sub),rdA(1,1,1,2),m,m,nsys,II,1,0)
        call replace_gpu_batch<<<blocks,threads>>>(C(1,1,1,1       ),rdC(1,1,1,1),m,m,nsys,II,1,0)
        call replace_gpu_batch<<<blocks,threads>>>(C(1,1,1,nrow_sub),rdC(1,1,1,2),m,m,nsys,II,1,0)
        call replace_gpu_batch<<<blocks,threads>>>(D(1,  1,1       ),rdD(1,  1,1),m,1,nsys,II,1,0)
        call replace_gpu_batch<<<blocks,threads>>>(D(1,  1,nrow_sub),rdD(1,  1,2),m,1,nsys,II,1,0)
        call replace_gpu_batch<<<blocks,threads>>>(B(1,1,1,1       ),rdB(1,1,1,1),m,m,nsys,II,0,1)
        call replace_gpu_batch<<<blocks,threads>>>(B(1,1,1,nrow_sub),rdB(1,1,1,2),m,m,nsys,II,0,1)

    end subroutine btdma_many_modi_gpu

    subroutine btdma_many_update_gpu(A,B,C,D,rdD,m,nsys,nrow_sub,plan)
        implicit none
        type(BTDMA_PLAN_gpu) :: plan
        integer, intent(in) :: m,nsys,nrow_sub        
        real*8, device:: A(1:m,1:m,1:nsys,1:nrow_sub)
        real*8, device:: B(1:m,1:m,1:nsys,1:nrow_sub)
        real*8, device:: C(1:m,1:m,1:nsys,1:nrow_sub)
        real*8, device:: D(1:m,    1:nsys,1:nrow_sub),rdD(1:m,    1:nsys,1:2)
    
        integer :: sys,q

        integer :: optimal_thread    
        type(dim3) :: threads, blocks

        ! optimal_thread =ceiling((16384.d0/dble(8*m+9))/32.d0)*32
        optimal_thread = 64
        threads = dim3(optimal_thread,1,1)
        blocks  = dim3(ceiling(dble(nsys)/dble(threads%x)),1,1)
        !! reduced
        call replace_gpu_batch<<<blocks,threads>>>(rdD(1,1,1),D(1,1,1       ),m,1,nsys,II,1,0)
        call replace_gpu_batch<<<blocks,threads>>>(rdD(1,1,2),D(1,1,nrow_sub),m,1,nsys,II,1,0)
        
        do q = 2, nrow_sub-1
            call gemm_gpu_batch<<<blocks,threads>>>(A(1,1, 1,q), m, D(1,1,1       ), D(1,1,q), 1, 1, -1, nsys)
            call gemm_gpu_batch<<<blocks,threads>>>(C(1,1, 1,q), m, D(1,1,nrow_sub), D(1,1,q), 1, 1, -1, nsys)
            ! call gemm_gpu_batch<<<blocks,threads>>>(A(1,1, 1,q), m, D(1,1,1       ), D(1,1,q), 1, 0,-1, nsys)
            ! call gemm_gpu_batch<<<blocks,threads>>>(C(1,1, 1,q), m, D(1,1,nrow_sub), D(1,1,q), 1, 0,-1, nsys)
        end do
        
    end subroutine btdma_many_update_gpu

    !---tools--------------------------------------------------------------------
        subroutine btdma_many_a2av_forward_gpu(plan)
            implicit none
            type(BTDMA_PLAN_gpu) :: plan
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


        end subroutine btdma_many_a2av_forward_gpu

        subroutine btdma_many_a2av_backward_gpu(plan)
            implicit none
            type(BTDMA_PLAN_gpu) :: plan
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

        end subroutine btdma_many_a2av_backward_gpu
    !-----------------------------------------------------------------------------
end module mod_btdma_gpu
