module mod_btdma_cpu
    use mpiutil
    implicit none
    
    type, public :: BTDMA_PLAN
        integer :: nsys_sub
        real*8, allocatable, dimension(:,:,:,:) :: rdA,rdB,rdC
        real*8, allocatable, dimension(:,:,:,:) :: trA,trB,trC
        real*8, allocatable, dimension(:  ,:,:) :: rdD
        real*8, allocatable, dimension(:  ,:,:) :: trD
        real*8, allocatable, dimension(:) :: bufRD,bufTR
        type(a2a_plan) :: commM,commV
    end type BTDMA_PLAN
contains
    subroutine btdma_many(n,nsys,m,a,b,c,d)
        implicit none
        integer, intent(in) :: n,nsys,m
        real*8, intent(inout) :: a(m,m,nsys,n), b(m,m,nsys,n), c(m,m,nsys,n), d(m,nsys,n)

        integer :: i,j,sys,q
        ! // LAPACK_COL_MAJOR
        real*8 :: Sol(m,m+1);            !//todo: 나중에 동적할당으로 수정 필요
        real*8 :: RR(m,m),AkCkm(m,m);    !//todo: 나중에 동적할당으로 수정 필요
        real*8 :: AkDkm(m),CkDkp(m);     !//todo: 나중에 동적할당으로 수정 필요
        integer :: ij,ijp;
        real*8 :: alpha = 1.0, beta = 0.0;

        integer :: info,ipiv(m);

        ! q=1
        do sys = 1, nsys
            ! timestemp_a
            Sol(1:m,1    ) = d(1:m    ,sys,1)
            Sol(1:m,2:m+1) = c(1:m,1:m,sys,1)
            RR (1:m,1:m  ) = b(1:m,1:m,sys,1)
            ! timestemp_b
            ! timestemp_a
            call dgesv(m,m+1, RR, m, ipiv, Sol, m, info)
            ! timestemp_b
            ! timestemp_a
            d(1:m    ,sys,1) = Sol(1:m,1    )
            c(1:m,1:m,sys,1) = Sol(1:m,2:m+1)
            ! timestemp_b
        end do

        do q = 2, n
            do sys = 1, nsys
                ! timestemp_a
                call dgemm('N', 'N', m, m, m, 1.0d0, a(:,:,sys, q), m, c(:,:,sys, (q-1)), m, 0.d0, AkCkm, m);
                call dgemv('N',      m, m,    1.0d0, a(:,:,sys, q), m, d(:  ,sys, (q-1)), 1, 0.d0, AkDkm, 1);
                ! timestemp_b
                ! timestemp_a
                Sol(1:m,1    ) = d(1:m    ,sys,q) - AkDkm(1:m)
                Sol(1:m,2:m+1) = c(1:m,1:m,sys,q)
                RR (1:m,1:m  ) = b(1:m,1:m,sys,q) - AkCkm(1:m,1:m)
                ! timestemp_b
                ! timestemp_a
                call dgesv(m,m+1, RR, m, ipiv, Sol, m, info)
                ! timestemp_b
                ! timestemp_a
                d(1:m    ,sys,q) = Sol(1:m,1    )
                c(1:m,1:m,sys,q) = Sol(1:m,2:m+1)
                ! timestemp_b
            end do
        end do

        do q = n-1, 1, -1
            do sys = 1, nsys
                ! timestemp_a
                call dgemv('N', m, m, 1.0d0, c(:,:,sys, q), m, d(:,sys,(q+1)), 1, 0.d0, CkDkp, 1);
                ! timestemp_b
                ! timestemp_a
                d(1:m,sys,q) = d(1:m,sys,q) - CkDkp(1:m);
                ! timestemp_b
            enddo
        enddo
        
    end subroutine btdma_many

    subroutine btdma_many_cycl(n,nsys,m,a,b,c,d)
        implicit none
        integer, intent(in) :: n,nsys,m
        real*8, intent(inout) :: a(m,m,nsys,n), b(m,m,nsys,n), c(m,m,nsys,n), d(m,nsys,n)

        integer :: i,j,sys,q
        ! // LAPACK_COL_MAJOR
        real*8 :: Sol(m,2*m+1);                     !//todo: 나중에 동적할당으로 수정 필요
        real*8 :: RR(m,m),AkCkm(m,m)                !//todo: 나중에 동적할당으로 수정 필요
        real*8 :: AkEkm(m,m),CkEkm(m,m),CkEkp(m,m); !//todo: 나중에 동적할당으로 수정 필요
        real*8 :: AkDkm(m),CkDkp(m),EkDk(m);        !//todo: 나중에 동적할당으로 수정 필요
        integer :: ij,ijp;
        real*8 :: alpha = 1.0, beta = 0.0;

        integer :: info,ipiv(m);

        real*8, allocatable, dimension(:,:,:,:) :: e

        allocate(e(m,m,nsys,n))
        
        e(:,:,:,:)=0.d0
        do sys=1,nsys
            e(1:m,1:m,sys,2)=-a(1:m,1:m,sys,2)
            e(1:m,1:m,sys,n)=-c(1:m,1:m,sys,n)
        enddo

        ! q=2
        do sys = 1, nsys
            ! timestemp_a
            Sol(1:m,1)          =d(1:m    ,sys,2)
            Sol(1:m,1+1  :m+1)  =e(1:m,1:m,sys,2)
            Sol(1:m,1+m+1:m+m+1)=c(1:m,1:m,sys,2)
            RR (1:m,1:m  )      =b(1:m,1:m,sys,2)
            ! timestemp_b
            ! timestemp_a
            call dgesv(m,2*m+1, RR, m, ipiv, Sol, m, info)
            ! timestemp_b
            ! timestemp_a
            d(1:m    ,sys,2) = Sol(1:m,1)          
            e(1:m,1:m,sys,2) = Sol(1:m,1+1  :m+1)  
            c(1:m,1:m,sys,2) = Sol(1:m,1+m+1:m+m+1)
            ! timestemp_b
        end do

        do q = 3, n
            do sys = 1, nsys
                ! timestemp_a
                call dgemm('N', 'N', m, m, m, 1.0d0, a(:,:,sys, q), m, c(:,:,sys, (q-1)), m, 0.d0, AkCkm, m);
                call dgemv('N',      m, m,    1.0d0, a(:,:,sys, q), m, d(:  ,sys, (q-1)), 1, 0.d0, AkDkm, 1);
                call dgemm('N', 'N', m, m, m, 1.0d0, a(:,:,sys, q), m, e(:,:,sys, (q-1)), m, 0.d0, AkEkm, m);
                ! timestemp_b
                ! timestemp_a
                Sol(1:m,1)          =d(1:m    ,sys,q)-AkDkm(1:m    )
                Sol(1:m,1+1  :m+1)  =e(1:m,1:m,sys,q)-AkEkm(1:m,1:m)
                Sol(1:m,1+m+1:m+m+1)=c(1:m,1:m,sys,q)
                RR (1:m,1:m  )      =b(1:m,1:m,sys,q)-AkCkm(1:m,1:m)
                ! timestemp_b
                ! timestemp_a
                call dgesv(m,2*m+1, RR, m, ipiv, Sol, m, info)
                ! timestemp_b
                ! timestemp_a
                d(1:m    ,sys,q)=Sol(1:m,1)          
                e(1:m,1:m,sys,q)=Sol(1:m,1+1  :m+1)  
                c(1:m,1:m,sys,q)=Sol(1:m,1+m+1:m+m+1)
                ! timestemp_b
            end do
        end do

        do q = n-1, 2, -1
            do sys = 1, nsys
                ! timestemp_a
                call dgemv('N',      m, m,    1.0d0, c(:,:,sys, q), m, d(:  ,sys,(q+1)), 1, 0.d0, CkDkp, 1);
                call dgemm('N', 'N', m, m, m, 1.0d0, c(:,:,sys, q), m, e(:,:,sys,(q+1)), m, 0.d0, CkEkp, m);
                ! timestemp_b
                ! timestemp_a
                d(1:m    ,sys,q)= d(1:m,    sys,q) - CkDkp(1:m);
                e(1:m,1:m,sys,q)= e(1:m,1:m,sys,q) - CkEkp(1:m,1:m)
                ! timestemp_b
            enddo
        enddo
        
        do sys=1,nsys
            call dgemm('N', 'N', m, m, m, 1.0d0, a(:,:,sys,1), m, e(:,:,sys,n), m, 0.d0, AkEkm, m);
            call dgemm('N', 'N', m, m, m, 1.0d0, c(:,:,sys,1), m, e(:,:,sys,2), m, 0.d0, CkEkp, m);
            
            call dgemv('N',      m, m,    1.0d0, a(:,:,sys,1), m, d(1:m    ,sys,n), 1, 0.d0, AkDkm, 1);
            call dgemv('N',      m, m,    1.0d0, c(:,:,sys,1), m, d(1:m    ,sys,2), 1, 0.d0, CkDkp, 1);
            
            RR(1:m,1:m) = b(1:m,1:m,sys,1)+AkEkm(1:m,1:m)+CkEkp(1:m,1:m)
            
            d(1:m,sys,1)= d(1:m,sys,1)-AkDkm(1:m)-CkDkp(1:m)

            call dgesv(m,1, RR, m, ipiv, d(:,sys,1), m, info)
                
        enddo

        do q = 2,n
            do sys=1,nsys
                call dgemv('N',      m, m,    1.0d0, e(:,:,sys,q), m, d(1:m,sys,1), 1, 0.d0, EkDk, 1);
                d(1:m,sys,q) = d(1:m,sys,q) + EkDk(1:m)
            enddo
        end do
        deallocate(e)
    end subroutine btdma_many_cycl

    subroutine btdma_makeplan(plan,m,nsys,nrow_sub,comm)
        use mpi
        implicit none
        type(BTDMA_PLAN) :: plan
        integer, intent(in) :: m,nsys,nrow_sub,comm

        integer :: nprocs,myrank,ierr
        integer :: indx_tmpa,indx_tmpb;

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
    end subroutine btdma_makeplan

    subroutine btdma_cleanplan(plan)
        implicit none
        type(BTDMA_PLAN) :: plan
    
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
        
    end subroutine btdma_cleanplan

    subroutine btdma_many_mpi(A,B,C,D,m,nsys,nrow_sub,plan)
        type(BTDMA_PLAN) :: plan
        integer, intent(in) :: m,nsys,nrow_sub        
        real*8 :: A(1:m,1:m,1:nsys,1:nrow_sub)
        real*8 :: B(1:m,1:m,1:nsys,1:nrow_sub)
        real*8 :: C(1:m,1:m,1:nsys,1:nrow_sub)
        real*8 :: D(1:m,    1:nsys,1:nrow_sub)
    
        
        call mpiutil_timecheck(t1_a,t1_b,0)
        call btdma_many_modi(A,B,C,D,plan%rdA,plan%rdB,plan%rdC,plan%rdD,m,nsys,nrow_sub,plan)
        call mpiutil_timecheck(t1_a,t1_b,1)
        
        call mpiutil_timecheck(t2_a,t2_b,0)
        call btdma_many_a2av_forward(plan)
        call mpiutil_timecheck(t2_a,t2_b,1)
        
        call mpiutil_timecheck(t3_a,t3_b,0)
        call btdma_many(2*plan%commM%nprocs,plan%nsys_sub,m,plan%trA,plan%trB,plan%trC,plan%trD)
        call mpiutil_timecheck(t3_a,t3_b,1)
        
        call mpiutil_timecheck(t4_a,t4_b,0)
        call btdma_many_a2av_backward(plan)
        call mpiutil_timecheck(t4_a,t4_b,1)
        
        call mpiutil_timecheck(t5_a,t5_b,0)
        call btdma_many_update(A,B,C,D,plan%rdD,m,nsys,nrow_sub,plan)
        call mpiutil_timecheck(t5_a,t5_b,1)

    end subroutine btdma_many_mpi

    subroutine btdma_many_cyclic_mpi(A,B,C,D,m,nsys,nrow_sub,plan)
        type(BTDMA_PLAN) :: plan
        integer, intent(in) :: m,nsys,nrow_sub        
        real*8 :: A(1:m,1:m,1:nsys,1:nrow_sub)
        real*8 :: B(1:m,1:m,1:nsys,1:nrow_sub)
        real*8 :: C(1:m,1:m,1:nsys,1:nrow_sub)
        real*8 :: D(1:m,    1:nsys,1:nrow_sub)
    
        
        call mpiutil_timecheck(t1_a,t1_b,0)
        call btdma_many_modi(A,B,C,D,plan%rdA,plan%rdB,plan%rdC,plan%rdD,m,nsys,nrow_sub,plan)
        call mpiutil_timecheck(t1_a,t1_b,1)
        
        call mpiutil_timecheck(t2_a,t2_b,0)
        call btdma_many_a2av_forward(plan)
        call mpiutil_timecheck(t2_a,t2_b,1)
        
        call mpiutil_timecheck(t3_a,t3_b,0)
        call btdma_many_cycl(2*plan%commM%nprocs,plan%nsys_sub,m,plan%trA,plan%trB,plan%trC,plan%trD)
        call mpiutil_timecheck(t3_a,t3_b,1)
        
        call mpiutil_timecheck(t4_a,t4_b,0)
        call btdma_many_a2av_backward(plan)
        call mpiutil_timecheck(t4_a,t4_b,1)
        
        call mpiutil_timecheck(t5_a,t5_b,0)
        call btdma_many_update(A,B,C,D,plan%rdD,m,nsys,nrow_sub,plan)
        call mpiutil_timecheck(t5_a,t5_b,1)

    end subroutine btdma_many_cyclic_mpi
    

    subroutine btdma_many_modi(A,B,C,D,rdA,rdB,rdC,rdD,m,nsys,nrow_sub,plan)
        implicit none
        type(BTDMA_PLAN) :: plan
        integer, intent(in) :: m,nsys,nrow_sub        
        real*8 :: A(1:m,1:m,1:nsys,1:nrow_sub),rdA(1:m,1:m,1:nsys,1:2)
        real*8 :: B(1:m,1:m,1:nsys,1:nrow_sub),rdB(1:m,1:m,1:nsys,1:2)
        real*8 :: C(1:m,1:m,1:nsys,1:nrow_sub),rdC(1:m,1:m,1:nsys,1:2)
        real*8 :: D(1:m,    1:nsys,1:nrow_sub),rdD(1:m,    1:nsys,1:2)

        integer :: sys,q,i,j
        real*8 :: Sol(m,(1+2*m));          !//todo: m 나중에 동적할당으로 수정 필요
        real*8 :: RR(m,m);                 !//todo: m 나중에 동적할당으로 수정 필요
        real*8 :: AqDqm(m),CqDqp(m);       !//todo: m 나중에 동적할당으로 수정 필요
        real*8 :: AqAqm(m,m),AqCqm(m,m);   !//todo: m 나중에 동적할당으로 수정 필요
        real*8 :: CqAqp(m,m),CqCqp(m,m);   !//todo: m 나중에 동적할당으로 수정 필요
        real*8 :: AqpCq(m,m);
        real*8 :: delta_ij(m,m);
        
        integer :: info,ipiv(m);
        real*8 :: alpha = 1.0, beta = 0.0;
        
        delta_ij(1:m,1:m) = 0.d0
        do i=1,m
            delta_ij(i,i) =1.d0
        enddo
        ! q=1
        do sys = 1, nsys
            ! timestemp_a
            Sol(1:m,1        ) = D(1:m    ,sys,1)
            Sol(1:m,1+1:  m+1) = A(1:m,1:m,sys,1)
            Sol(1:m,m+2:2*m+1) = C(1:m,1:m,sys,1)
            RR (1:m,1:m      ) = B(1:m,1:m,sys,1)
            ! timestemp_b
            ! timestemp_a
            call dgesv(m,2*m+1, RR, m, ipiv, Sol, m, info)
            ! timestemp_b
            ! timestemp_a
            D(1:m    ,sys,1) = Sol(1:m,1        )
            A(1:m,1:m,sys,1) = Sol(1:m,1+1:  m+1)
            C(1:m,1:m,sys,1) = Sol(1:m,m+2:2*m+1)
            ! timestemp_b
        end do

        ! q=2
        do sys = 1, nsys
            ! timestemp_a
            Sol(1:m,1        ) = D(1:m    ,sys,2)
            Sol(1:m,2  :  m+1) = A(1:m,1:m,sys,2)
            Sol(1:m,2+m:2*m+1) = C(1:m,1:m,sys,2)
            RR (1:m,1:m      ) = B(1:m,1:m,sys,2)
            ! timestemp_b
            ! timestemp_a
            call dgesv(m,2*m+1, RR, m, ipiv, Sol, m, info)
            ! timestemp_b
            ! timestemp_a
            D(1:m    ,sys,2) = Sol(1:m,1        )
            A(1:m,1:m,sys,2) = Sol(1:m,2  :  m+1)
            C(1:m,1:m,sys,2) = Sol(1:m,2+m:2*m+1)
            ! timestemp_b
        end do

        ! q=3~nrow_sub
        do q = 3, nrow_sub
            do sys = 1, nsys
                ! timestemp_a
                call dgemv('N',      m, m,    1.0d0, A(:,:,sys, q), m, D(:  ,sys, (q-1)), 1, 0.d0, AqDqm, 1);
                call dgemm('N', 'N', m, m, m, 1.0d0, A(:,:,sys, q), m, A(:,:,sys, (q-1)), m, 0.d0, AqAqm, m);
                call dgemm('N', 'N', m, m, m, 1.0d0, A(:,:,sys, q), m, C(:,:,sys, (q-1)), m, 0.d0, AqCqm, m);
                ! timestemp_b
                ! timestemp_a
                Sol(1:m,1        ) = D(1:m    ,sys,q) - AqDqm(1:m)
                Sol(1:m,2  :  m+1) =-AqAqm(1:m,1:m)
                Sol(1:m,2+m:2*m+1) = C(1:m,1:m,sys,q)
                RR (1:m,1:m      ) = B(1:m,1:m,sys,q) - AqCqm(1:m,1:m);
                ! timestemp_b
                ! timestemp_a
                call dgesv(m,2*m+1, RR, m, ipiv, Sol, m, info)
                ! timestemp_b
                ! timestemp_a
                D(1:m    ,sys,q) = Sol(1:m,1        )
                A(1:m,1:m,sys,q) = Sol(1:m,2  :  m+1)
                C(1:m,1:m,sys,q) = Sol(1:m,2+m:2*m+1)
                ! timestemp_b
            end do
        end do
        
        ! q=nrow_sub-1~2
        do q = nrow_sub-2, 2, -1
            do sys = 1, nsys
                ! timestemp_a
                call dgemv('N',      m, m,    1.0d0, C(:,:,sys, q), m, D(:  ,sys, (q+1)), 1, 0.d0, CqDqp, 1);
                call dgemm('N', 'N', m, m, m, 1.0d0, C(:,:,sys, q), m, A(:,:,sys, (q+1)), m, 0.d0, CqAqp, m);
                call dgemm('N', 'N', m, m, m, 1.0d0, C(:,:,sys, q), m, C(:,:,sys, (q+1)), m, 0.d0, CqCqp, m);
                ! timestemp_b
                ! timestemp_a
                D(1:m    ,sys,q) = D(1:m    ,sys,q) - CqDqp(1:m)
                A(1:m,1:m,sys,q) = A(1:m,1:m,sys,q) - CqAqp(1:m,1:m)
                C(1:m,1:m,sys,q) =-CqCqp(1:m,1:m)
                ! timestemp_b
            end do
        end do

        ! q=1
        do sys = 1, nsys
            ! timestemp_a
            call dgemv('N',      m, m,    1.0d0, C(:,:,sys, 1), m, D(:  ,sys, 2), 1, 0.d0, CqDqp, 1);
            call dgemm('N', 'N', m, m, m, 1.0d0, C(:,:,sys, 1), m, C(:,:,sys, 2), m, 0.d0, CqCqp, m);
            call dgemm('N', 'N', m, m, m, 1.0d0, C(:,:,sys, 1), m, A(:,:,sys, 2), m, 0.d0, CqAqp, m);
            ! timestemp_b
            ! timestemp_a
            Sol(1:m,1        ) = D(1:m    ,sys,1) - CqDqp(1:m)
            Sol(1:m,2  :  m+1) = A(1:m,1:m,sys,1)
            Sol(1:m,2+m:2*m+1) =-CqCqp(1:m,1:m)
            RR (1:m,1:m      ) = delta_ij(1:m,1:m) - CqAqp(1:m,1:m);
            ! timestemp_b
            ! timestemp_a
            call dgesv(m,2*m+1, RR, m, ipiv, Sol, m, info)
            ! timestemp_b
            ! timestemp_a
            D(1:m    ,sys,1) = Sol(1:m,1        )
            A(1:m,1:m,sys,1) = Sol(1:m,2  :  m+1)
            C(1:m,1:m,sys,1) = Sol(1:m,2+m:2*m+1)
            ! timestemp_b
        end do

        !! reduced
        ! timestemp_a
        rdA(1:m,1:m,1:nsys,1) = A(1:m,1:m,1:nsys,1       )
        rdA(1:m,1:m,1:nsys,2) = A(1:m,1:m,1:nsys,nrow_sub)
        rdC(1:m,1:m,1:nsys,1) = C(1:m,1:m,1:nsys,1       )
        rdC(1:m,1:m,1:nsys,2) = C(1:m,1:m,1:nsys,nrow_sub)
        rdD(1:m,    1:nsys,1) = D(1:m,    1:nsys,1       )
        rdD(1:m,    1:nsys,2) = D(1:m,    1:nsys,nrow_sub)
        do sys = 1, nsys
            rdB(1:m,1:m,sys,1) = delta_ij(1:m,1:m)
            rdB(1:m,1:m,sys,2) = delta_ij(1:m,1:m)
        end do
        ! timestemp_b
    end subroutine btdma_many_modi

    subroutine btdma_many_a2av_forward(plan)
        implicit none
        type(BTDMA_PLAN) :: plan
        integer :: ierr

        ! commM,commV
        call mpiutil_pack(plan%rdA, plan%bufRD, plan%commM%A, plan%commM)
        call MPI_ALLTOALLV(plan%bufRD, plan%commM%A%counts, plan%commM%A%displs, MPI_DOUBLE_PRECISION,  &
                           plan%bufTR, plan%commM%B%counts, plan%commM%B%displs, MPI_DOUBLE_PRECISION,  &
                           plan%commM%mpi_comm, ierr)
        call mpiutil_unpack(plan%trA, plan%bufTR, plan%commM%B, plan%commM)


        call mpiutil_pack(plan%rdB, plan%bufRD, plan%commM%A, plan%commM)
        call MPI_ALLTOALLV(plan%bufRD, plan%commM%A%counts, plan%commM%A%displs, MPI_DOUBLE_PRECISION,  &
                           plan%bufTR, plan%commM%B%counts, plan%commM%B%displs, MPI_DOUBLE_PRECISION,  &
                           plan%commM%mpi_comm, ierr)
        call mpiutil_unpack(plan%trB, plan%bufTR, plan%commM%B, plan%commM)

        call mpiutil_pack(plan%rdC, plan%bufRD, plan%commM%A, plan%commM)
        call MPI_ALLTOALLV(plan%bufRD, plan%commM%A%counts, plan%commM%A%displs, MPI_DOUBLE_PRECISION,  &
                           plan%bufTR, plan%commM%B%counts, plan%commM%B%displs, MPI_DOUBLE_PRECISION,  &
                           plan%commM%mpi_comm, ierr)
        call mpiutil_unpack(plan%trC, plan%bufTR, plan%commM%B, plan%commM)

        call mpiutil_pack(plan%rdD, plan%bufRD, plan%commV%A, plan%commV)
        call MPI_ALLTOALLV(plan%bufRD, plan%commV%A%counts, plan%commV%A%displs, MPI_DOUBLE_PRECISION,  &
                           plan%bufTR, plan%commV%B%counts, plan%commV%B%displs, MPI_DOUBLE_PRECISION,  &
                           plan%commV%mpi_comm, ierr)
        call mpiutil_unpack(plan%trD, plan%bufTR, plan%commV%B, plan%commV)


    end subroutine btdma_many_a2av_forward

    subroutine btdma_many_a2av_backward(plan)
        implicit none
        type(BTDMA_PLAN) :: plan
        integer :: ierr
        ! commM,commV
        call mpiutil_pack(plan%trA, plan%bufTR, plan%commM%B, plan%commM)
        call MPI_ALLTOALLV(plan%bufTR, plan%commM%B%counts, plan%commM%B%displs, MPI_DOUBLE_PRECISION,  &
                           plan%bufRD, plan%commM%A%counts, plan%commM%A%displs, MPI_DOUBLE_PRECISION,  &                           
                           plan%commM%mpi_comm, ierr)
        call mpiutil_unpack(plan%rdA, plan%bufRD, plan%commM%A, plan%commM)

        call mpiutil_pack(plan%trB, plan%bufTR, plan%commM%B, plan%commM)
        call MPI_ALLTOALLV(plan%bufTR, plan%commM%B%counts, plan%commM%B%displs, MPI_DOUBLE_PRECISION,  &
                           plan%bufRD, plan%commM%A%counts, plan%commM%A%displs, MPI_DOUBLE_PRECISION,  &                           
                           plan%commM%mpi_comm, ierr)
        call mpiutil_unpack(plan%rdB, plan%bufRD, plan%commM%A, plan%commM)

        call mpiutil_pack(plan%trC, plan%bufTR, plan%commM%B, plan%commM)
        call MPI_ALLTOALLV(plan%bufTR, plan%commM%B%counts, plan%commM%B%displs, MPI_DOUBLE_PRECISION,  &
                           plan%bufRD, plan%commM%A%counts, plan%commM%A%displs, MPI_DOUBLE_PRECISION,  &                           
                           plan%commM%mpi_comm, ierr)
        call mpiutil_unpack(plan%rdC, plan%bufRD, plan%commM%A, plan%commM)

        call mpiutil_pack(plan%trD, plan%bufTR, plan%commV%B, plan%commV)
        call MPI_ALLTOALLV(plan%bufTR, plan%commV%B%counts, plan%commV%B%displs, MPI_DOUBLE_PRECISION,  &
                           plan%bufRD, plan%commV%A%counts, plan%commV%A%displs, MPI_DOUBLE_PRECISION,  &                           
                           plan%commV%mpi_comm, ierr)
        call mpiutil_unpack(plan%rdD, plan%bufRD, plan%commV%A, plan%commV)


    end subroutine btdma_many_a2av_backward

    subroutine btdma_many_update(A,B,C,D,rdD,m,nsys,nrow_sub,plan)
        implicit none
        type(BTDMA_PLAN) :: plan
        integer, intent(in) :: m,nsys,nrow_sub        
        real*8 :: A(1:m,1:m,1:nsys,1:nrow_sub)
        real*8 :: B(1:m,1:m,1:nsys,1:nrow_sub)
        real*8 :: C(1:m,1:m,1:nsys,1:nrow_sub)
        real*8 :: D(1:m,    1:nsys,1:nrow_sub),rdD(1:m,    1:nsys,1:2)
    
        integer :: sys,q
        real*8 :: AqD0(m),CqDn(m); !//todo: 나중에 동적할당으로 수정 필요

        !! reduced
        D(1:m,    1:nsys,1       ) = rdD(1:m,    1:nsys,1)
        D(1:m,    1:nsys,nrow_sub) = rdD(1:m,    1:nsys,2)
        
        do q = 2, nrow_sub-1
            do sys = 1, nsys
                call dgemv('N', m, m, 1.0d0, A(:,:,sys, q), m, D(:,sys,       1), 1, 0.d0, AqD0, 1);
                call dgemv('N', m, m, 1.0d0, C(:,:,sys, q), m, D(:,sys,nrow_sub), 1, 0.d0, CqDn, 1);
                D(1:m,sys,q) = D(1:m,sys,q) - AqD0(1:m) - CqDn(1:m)
            end do
        end do
        
    end subroutine btdma_many_update

    subroutine btdma_test_cyclic(A,B,C,D,m,nsys,nrow_sub,plan)
        type(BTDMA_PLAN) :: plan
        integer, intent(in) :: m,nsys,nrow_sub        
        real*8 :: A(1:m,1:m,1:nsys,1:nrow_sub)
        real*8 :: B(1:m,1:m,1:nsys,1:nrow_sub)
        real*8 :: C(1:m,1:m,1:nsys,1:nrow_sub)
        real*8 :: D(1:m,    1:nsys,1:nrow_sub)
    
        call btdma_many_cycl(nrow_sub,nsys,m,A,B,C,D)
        ! call btdma_many(nrow_sub,nsys,m,A,B,C,D)
        

    end subroutine btdma_test_cyclic
end module mod_btdma_cpu
!!check