program example_cpu
    use mpi
    use mod_btdma_cpu
    implicit none

    integer, parameter :: m = 2
    integer, parameter :: nsys = 16
    integer, parameter :: nrow_sub = 8
    real(8), parameter :: tolerance = 1.0d-10

    integer :: ierr, myrank, nprocs
    integer :: q, sys, i, j, grow, nrow_global
    integer :: exit_code
    real(8) :: x_prev(m), x_next(m), residual(m)
    real(8) :: sums_local(4), sums_global(4)
    real(8) :: normalized_residual, relative_error

    real(8), allocatable :: a(:,:,:,:), b(:,:,:,:), c(:,:,:,:)
    real(8), allocatable :: rhs(:,:,:), x_ref(:,:,:)
    real(8), allocatable :: a0(:,:,:,:), b0(:,:,:,:), c0(:,:,:,:), rhs0(:,:,:)
    real(8), allocatable :: x_all(:,:,:,:)
    type(BTDMA_PLAN) :: plan

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

    if (nprocs > nsys) then
        if (myrank == 0) write(*,'(A,I0,A,I0)') &
            'ERROR: this example supports at most ', nsys, ' MPI ranks; requested ', nprocs
        call MPI_Abort(MPI_COMM_WORLD, 2, ierr)
    end if

    nrow_global = nrow_sub * nprocs

    allocate(a(m,m,nsys,nrow_sub), b(m,m,nsys,nrow_sub), c(m,m,nsys,nrow_sub))
    allocate(rhs(m,nsys,nrow_sub), x_ref(m,nsys,nrow_sub))
    allocate(a0(m,m,nsys,nrow_sub), b0(m,m,nsys,nrow_sub), c0(m,m,nsys,nrow_sub))
    allocate(rhs0(m,nsys,nrow_sub))
    allocate(x_all(m,nsys,nrow_sub,0:nprocs-1))

    a = 0.0d0
    b = 0.0d0
    c = 0.0d0
    rhs = 0.0d0

    do q = 1, nrow_sub
        grow = myrank * nrow_sub + q
        do sys = 1, nsys
            do i = 1, m
                a(i,i,sys,q) = -0.20d0
                b(i,i,sys,q) =  2.20d0 + 0.002d0 * dble(sys)
                c(i,i,sys,q) = -0.15d0
                x_ref(i,sys,q) = exact_value(i, sys, grow)
            end do
            b(1,2,sys,q) = 0.05d0
            b(2,1,sys,q) = 0.05d0

            if (grow == 1) a(:,:,sys,q) = 0.0d0
            if (grow == nrow_global) c(:,:,sys,q) = 0.0d0

            x_prev = 0.0d0
            x_next = 0.0d0
            if (grow > 1) then
                do i = 1, m
                    x_prev(i) = exact_value(i, sys, grow - 1)
                end do
            end if
            if (grow < nrow_global) then
                do i = 1, m
                    x_next(i) = exact_value(i, sys, grow + 1)
                end do
            end if

            rhs(:,sys,q) = matmul(a(:,:,sys,q), x_prev) &
                         + matmul(b(:,:,sys,q), x_ref(:,sys,q)) &
                         + matmul(c(:,:,sys,q), x_next)
        end do
    end do

    a0 = a
    b0 = b
    c0 = c
    rhs0 = rhs

    call btdma_makeplan(plan, m, nsys, nrow_sub, MPI_COMM_WORLD)
    call btdma_many_mpi(a, b, c, rhs, m, nsys, nrow_sub, plan)

    call MPI_Allgather(rhs, m*nsys*nrow_sub, MPI_DOUBLE_PRECISION, &
                       x_all, m*nsys*nrow_sub, MPI_DOUBLE_PRECISION, &
                       MPI_COMM_WORLD, ierr)

    sums_local = 0.0d0
    do q = 1, nrow_sub
        grow = myrank * nrow_sub + q
        do sys = 1, nsys
            x_prev = 0.0d0
            x_next = 0.0d0
            if (grow > 1) then
                if (q > 1) then
                    x_prev = rhs(:,sys,q-1)
                else
                    x_prev = x_all(:,sys,nrow_sub,myrank-1)
                end if
            end if
            if (grow < nrow_global) then
                if (q < nrow_sub) then
                    x_next = rhs(:,sys,q+1)
                else
                    x_next = x_all(:,sys,1,myrank+1)
                end if
            end if

            residual = matmul(a0(:,:,sys,q), x_prev) &
                     + matmul(b0(:,:,sys,q), rhs(:,sys,q)) &
                     + matmul(c0(:,:,sys,q), x_next) - rhs0(:,sys,q)

            sums_local(1) = sums_local(1) + sum(residual**2)
            sums_local(2) = sums_local(2) + sum(rhs0(:,sys,q)**2)
            sums_local(3) = sums_local(3) + sum((rhs(:,sys,q)-x_ref(:,sys,q))**2)
            sums_local(4) = sums_local(4) + sum(x_ref(:,sys,q)**2)
        end do
    end do

    call MPI_Allreduce(sums_local, sums_global, 4, MPI_DOUBLE_PRECISION, MPI_SUM, &
                       MPI_COMM_WORLD, ierr)

    normalized_residual = sqrt(sums_global(1) / sums_global(2))
    relative_error = sqrt(sums_global(3) / sums_global(4))

    exit_code = 0
    if (normalized_residual > tolerance .or. relative_error > tolerance) exit_code = 1

    if (myrank == 0) then
        write(*,'(A)') 'PaScaL_BTDMA minimal CPU example'
        write(*,'(A,I0)') 'MPI ranks: ', nprocs
        write(*,'(A,I0)') 'Global block rows: ', nrow_global
        write(*,'(A,ES14.6)') 'Normalized residual: ', normalized_residual
        write(*,'(A,ES14.6)') 'Relative solution error: ', relative_error
        if (exit_code == 0) then
            write(*,'(A)') 'RESULT: PASS'
        else
            write(*,'(A)') 'RESULT: FAIL'
        end if
    end if

    call btdma_cleanplan(plan)
    deallocate(a, b, c, rhs, x_ref, a0, b0, c0, rhs0, x_all)
    call MPI_Finalize(ierr)

    if (exit_code /= 0) stop 1

contains

    pure real(8) function exact_value(component, system_id, global_row)
        integer, intent(in) :: component, system_id, global_row
        exact_value = sin(0.017d0 * dble(11*global_row + 3*system_id + component)) &
                    + 0.001d0 * dble(component + system_id)
    end function exact_value

end program example_cpu

