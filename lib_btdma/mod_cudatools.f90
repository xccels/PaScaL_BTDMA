module mod_cudatools
    use cudafor
    implicit none
contains
    attributes(global) subroutine gesv_gpu_batch_single(A, m, x1, nrhs1, nsys)
        implicit none
        integer, value :: m, nrhs1, nsys
        real*8, device :: A (1:m,1:m     ,1:nsys)
        real*8, device :: x1(1:m,1:nrhs1 ,1:nsys)
        integer :: i_sys
        integer :: p, q, r, irhs

        real*8 :: factor(1:32)

        i_sys = (blockidx%x-1)*blockdim%x + (threadidx%x-1) + 1

        if(i_sys <= nsys) then
            ! Gaussian elimination
            do r = 1, m-1
                do p = r+1, m
                    factor(p) = A(p,r,i_sys) / A(r,r,i_sys)
                    do q = r, m
                        A(p,q,i_sys) = A(p,q,i_sys) - factor(p) * A(r,q,i_sys)
                    end do
                end do
                !x1
                do irhs = 1, nrhs1
                    do p = r+1, m
                        x1(p,irhs,i_sys) = x1(p,irhs,i_sys) - factor(p) * x1(r,irhs,i_sys) 
                    enddo
                end do
            end do

            ! Back substitution
            !x1
            do irhs = 1, nrhs1
                x1(m,irhs,i_sys) = x1(m,irhs,i_sys) / A(m,m,i_sys)
                do p = m-1, 1, -1
                    x1(p,irhs,i_sys) = x1(p,irhs,i_sys)
                    do q = p+1, m
                        x1(p,irhs,i_sys) = x1(p,irhs,i_sys) - A(p,q,i_sys) * x1(q,irhs,i_sys)
                    end do
                    x1(p,irhs,i_sys) = x1(p,irhs,i_sys) / A(p,p,i_sys)
                enddo
            end do
        endif

    end subroutine gesv_gpu_batch_single
    attributes(global) subroutine gesv_gpu_batch_multi2(A, m, x1, nrhs1, x2, nrhs2, nsys)
        implicit none
        integer, value :: m, nrhs1, nrhs2, nsys
        real*8, device :: A (1:m,1:m     ,1:nsys)
        real*8, device :: x1(1:m,1:nrhs1 ,1:nsys)
        real*8, device :: x2(1:m,1:nrhs2 ,1:nsys)
        integer :: i_sys
        integer :: p, q, r, irhs

        real*8 :: factor(1:32)

        i_sys = (blockidx%x-1)*blockdim%x + (threadidx%x-1) + 1

        if(i_sys <= nsys) then
            ! Gaussian elimination
            do r = 1, m-1
                do p = r+1, m
                    factor(p) = A(p,r,i_sys) / A(r,r,i_sys)
                    do q = r, m
                        A(p,q,i_sys) = A(p,q,i_sys) - factor(p) * A(r,q,i_sys)
                    end do
                end do
                !x1
                do irhs = 1, nrhs1
                    do p = r+1, m
                        x1(p,irhs,i_sys) = x1(p,irhs,i_sys) - factor(p) * x1(r,irhs,i_sys) 
                    enddo
                end do
                !x2
                do irhs = 1, nrhs2
                    do p = r+1, m
                        x2(p,irhs,i_sys) = x2(p,irhs,i_sys) - factor(p) * x2(r,irhs,i_sys) 
                    enddo
                end do
            end do

            ! Back substitution
            !x1
            do irhs = 1, nrhs1
                x1(m,irhs,i_sys) = x1(m,irhs,i_sys) / A(m,m,i_sys)
                do p = m-1, 1, -1
                    x1(p,irhs,i_sys) = x1(p,irhs,i_sys)
                    do q = p+1, m
                        x1(p,irhs,i_sys) = x1(p,irhs,i_sys) - A(p,q,i_sys) * x1(q,irhs,i_sys)
                    end do
                    x1(p,irhs,i_sys) = x1(p,irhs,i_sys) / A(p,p,i_sys)
                enddo
            end do
            !x2
            do irhs = 1, nrhs2
                x2(m,irhs,i_sys) = x2(m,irhs,i_sys) / A(m,m,i_sys)
                do p = m-1, 1, -1
                    x2(p,irhs,i_sys) = x2(p,irhs,i_sys)
                    do q = p+1, m
                        x2(p,irhs,i_sys) = x2(p,irhs,i_sys) - A(p,q,i_sys) * x2(q,irhs,i_sys)
                    end do
                    x2(p,irhs,i_sys) = x2(p,irhs,i_sys) / A(p,p,i_sys)
                enddo
            end do
        endif

    end subroutine gesv_gpu_batch_multi2
    attributes(global) subroutine gesv_gpu_batch_multi3(A, m, x1, nrhs1, x2, nrhs2, x3, nrhs3, nsys)
        implicit none
        integer, value :: m, nrhs1, nrhs2, nrhs3, nsys
        real*8, device :: A (1:m,1:m     ,1:nsys)
        real*8, device :: x1(1:m,1:nrhs1 ,1:nsys)
        real*8, device :: x2(1:m,1:nrhs2 ,1:nsys)
        real*8, device :: x3(1:m,1:nrhs3 ,1:nsys)
        integer :: i_sys
        integer :: p, q, r, irhs

        real*8 :: factor(1:32)

        i_sys = (blockidx%x-1)*blockdim%x + (threadidx%x-1) + 1

        if(i_sys <= nsys) then
            ! Gaussian elimination
            do r = 1, m-1
                do p = r+1, m
                    factor(p) = A(p,r,i_sys) / A(r,r,i_sys)
                    do q = r, m
                        A(p,q,i_sys) = A(p,q,i_sys) - factor(p) * A(r,q,i_sys)
                    end do
                end do
                !x1
                do irhs = 1, nrhs1
                    do p = r+1, m
                        x1(p,irhs,i_sys) = x1(p,irhs,i_sys) - factor(p) * x1(r,irhs,i_sys) 
                    enddo
                end do
                !x2
                do irhs = 1, nrhs2
                    do p = r+1, m
                        x2(p,irhs,i_sys) = x2(p,irhs,i_sys) - factor(p) * x2(r,irhs,i_sys) 
                    enddo
                end do
                !x3
                do irhs = 1, nrhs3
                    do p = r+1, m
                        x3(p,irhs,i_sys) = x3(p,irhs,i_sys) - factor(p) * x3(r,irhs,i_sys) 
                    enddo
                end do
            end do

            ! Back substitution
            !x1
            do irhs = 1, nrhs1
                x1(m,irhs,i_sys) = x1(m,irhs,i_sys) / A(m,m,i_sys)
                do p = m-1, 1, -1
                    x1(p,irhs,i_sys) = x1(p,irhs,i_sys)
                    do q = p+1, m
                        x1(p,irhs,i_sys) = x1(p,irhs,i_sys) - A(p,q,i_sys) * x1(q,irhs,i_sys)
                    end do
                    x1(p,irhs,i_sys) = x1(p,irhs,i_sys) / A(p,p,i_sys)
                enddo
            end do
            !x2
            do irhs = 1, nrhs2
                x2(m,irhs,i_sys) = x2(m,irhs,i_sys) / A(m,m,i_sys)
                do p = m-1, 1, -1
                    x2(p,irhs,i_sys) = x2(p,irhs,i_sys)
                    do q = p+1, m
                        x2(p,irhs,i_sys) = x2(p,irhs,i_sys) - A(p,q,i_sys) * x2(q,irhs,i_sys)
                    end do
                    x2(p,irhs,i_sys) = x2(p,irhs,i_sys) / A(p,p,i_sys)
                enddo
            end do
            !x3
            do irhs = 1, nrhs3
                x3(m,irhs,i_sys) = x3(m,irhs,i_sys) / A(m,m,i_sys)
                do p = m-1, 1, -1
                    x3(p,irhs,i_sys) = x3(p,irhs,i_sys)
                    do q = p+1, m
                        x3(p,irhs,i_sys) = x3(p,irhs,i_sys) - A(p,q,i_sys) * x3(q,irhs,i_sys)
                    end do
                    x3(p,irhs,i_sys) = x3(p,irhs,i_sys) / A(p,p,i_sys)
                enddo
            end do
        endif

    end subroutine gesv_gpu_batch_multi3

    attributes(global) subroutine gemm_gpu_batch(A, m1, B, C, m2, alpha, beta, nsys)
        integer, value :: m1, m2, alpha, beta, nsys
        real*8, device :: A(1:m1,1:m1,1:nsys)
        real*8, device :: B(1:m1,1:m2,1:nsys)
        real*8, device :: C(1:m1,1:m2,1:nsys)
        integer :: i_sys
        integer :: i, j, k
        real*8  :: sss(1:32, 1:32)

        i_sys = (blockidx%x-1)*blockdim%x + (threadidx%x-1) + 1

        if(i_sys <= nsys) then
            sss(1:m1, 1:m2) = 0.d0
            do i = 1, m1
            do j = 1, m2
                do k = 1, m1
                    sss(i,j) = sss(i,j) + A(i,k, i_sys) * B(k,j, i_sys)
                end do
            end do
            end do

            do i = 1, m1
            do j = 1, m2
                C(i,j, i_sys) = dble(alpha)*C(i,j, i_sys) + dble(beta)*sss(i,j)
            end do
            end do
        endif
    end subroutine gemm_gpu_batch

    attributes(global) subroutine gemm_gpu_batch_delta(A, B, C, II, m, beta, nsys)
        integer, value :: m, beta, nsys
        real*8, device :: A(1:m,1:m,1:nsys)
        real*8, device :: B(1:m,1:m,1:nsys)
        real*8, device :: C(1:m,1:m,1:nsys)
        real*8, device :: II(1:32, 1:32)
        integer :: i_sys
        integer :: i, j, k
        

        i_sys = (blockidx%x-1)*blockdim%x + (threadidx%x-1) + 1

        if(i_sys <= nsys) then
            do i = 1, m
            do j = 1, m
                C(i,j, i_sys) = II(i,j)
            end do
            end do

            do i = 1, m
            do j = 1, m
                do k = 1, m
                    C(i,j, i_sys) = C(i,j, i_sys) + dble(beta)*A(i,k, i_sys) * B(k,j, i_sys)
                end do
            end do
            end do

        endif
    end subroutine gemm_gpu_batch_delta

    attributes(global) subroutine replace_gpu_batch(A, B, m1, m2, nsys, II, alpha, beta)
        integer, value :: m1, m2, nsys, alpha, beta
        real*8, device :: A(1:m1,1:m2,1:nsys)
        real*8, device :: B(1:m1,1:m2,1:nsys)
        real*8, device :: II(1:32, 1:32)
        integer :: i_sys
        integer :: i, j, k
        

        i_sys = (blockidx%x-1)*blockdim%x + (threadidx%x-1) + 1

        if(i_sys <= nsys) then
            B(1:m1,1:m2,i_sys) = dble(alpha)*A(1:m1,1:m2,i_sys) + dble(beta)*II(1:m1,1:m2)
        endif

    end subroutine replace_gpu_batch

    attributes(global) subroutine init_array(A,m1,m2,nsys,n,vv)
        integer, value :: m1,m2,nsys,n
        real*8 , value :: vv
        real*8, device :: A(1:m1,1:m2,1:nsys,1:n)
        integer :: i, j
        
        i = (blockidx%x-1)*blockdim%x + (threadidx%x-1) + 1
        j = (blockidx%y-1)*blockdim%y + (threadidx%y-1) + 1

        if(i<=nsys .and. j<=n) then
            A(1:m1,1:m2,i,j) = vv
        endif

    end subroutine init_array
endmodule