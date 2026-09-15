module communication_baseline_gpu
    use cudafor
    use mod_cudatools, only: gesv_gpu_batch_multi2, gemm_gpu_batch
    implicit none

contains

    ! Local block-Thomas phase used after the conventional all-to-all
    ! redistribution.  This is intentionally folder-local: the public solver
    ! library exposes the optimised PaScaL GPU path, not the old-layout
    ! conventional baseline required by this comparison.
    subroutine solve_local_baseline_gpu(n, nsys, m, a, b, c, d)
        integer, value :: n, nsys, m
        real(kind=8), device :: a(m,m,nsys,n), b(m,m,nsys,n), c(m,m,nsys,n)
        real(kind=8), device :: d(m,nsys,n)
        integer :: q
        type(dim3) :: threads, blocks

        threads = dim3(64, 1, 1)
        blocks = dim3(ceiling(dble(nsys)/dble(threads%x)), 1, 1)

        q = 1
        call gesv_gpu_batch_multi2<<<blocks,threads>>>(b(1,1,1,q), m, d(1,1,q), 1, &
                                                       c(1,1,1,q), m, nsys)
        do q = 2, n
            call gemm_gpu_batch<<<blocks,threads>>>(a(1,1,1,q), m, c(1,1,1,q-1), &
                                                    b(1,1,1,q), m, 1, -1, nsys)
            call gemm_gpu_batch<<<blocks,threads>>>(a(1,1,1,q), m, d(1,1,q-1), &
                                                    d(1,1,q), 1, 1, -1, nsys)
            call gesv_gpu_batch_multi2<<<blocks,threads>>>(b(1,1,1,q), m, d(1,1,q), 1, &
                                                           c(1,1,1,q), m, nsys)
        end do

        do q = n-1, 1, -1
            call gemm_gpu_batch<<<blocks,threads>>>(c(1,1,1,q), m, d(1,1,q+1), &
                                                    d(1,1,q), 1, 1, -1, nsys)
        end do
    end subroutine solve_local_baseline_gpu

end module communication_baseline_gpu
