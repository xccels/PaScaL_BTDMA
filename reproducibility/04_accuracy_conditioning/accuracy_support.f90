module accuracy_support
    use, intrinsic :: iso_fortran_env, only: real64, int64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none

    integer, parameter :: dp = real64
    integer, parameter :: n_global_design = 2048
    integer, parameter :: nsys_design = 128 * 128
    integer, parameter :: nsys_family_design = nsys_design / 2
    integer(int64), parameter :: base_seed = 271828183_int64
    real(dp), parameter :: residual_failure_tolerance = 1.0e-8_dp
    character(len=*), parameter :: generator_version = 'controlled-givens-v1'

contains

    pure real(dp) function condition_value(log10_k) result(kappa)
        integer, intent(in) :: log10_k
        kappa = 10.0_dp**log10_k
    end function condition_value

    pure subroutine theoretical_conditions(family, kappa, global_condition, block_condition)
        integer, intent(in) :: family
        real(dp), intent(in) :: kappa
        real(dp), intent(out) :: global_condition, block_condition

        select case (family)
        case (1)
            global_condition = kappa
            block_condition = 1.0_dp
        case (2)
            global_condition = 2.0_dp * kappa
            block_condition = kappa
        case default
            global_condition = -1.0_dp
            block_condition = -1.0_dp
        end select
    end subroutine theoretical_conditions

    pure function family_label(family) result(label)
        integer, intent(in) :: family
        character(len=16) :: label

        select case (family)
        case (1)
            label = 'global'
        case (2)
            label = 'diagonal-block'
        case default
            label = 'unknown'
        end select
    end function family_label

    subroutine make_input_id(m, log10_k, family, system_id, input_id)
        integer, intent(in) :: m, log10_k, family, system_id
        character(len=*), intent(out) :: input_id
        character(len=1) :: family_code

        if (family == 1) then
            family_code = 'G'
        else
            family_code = 'B'
        end if

        write(input_id, '(A,"-seed",I0,"-m",I0,"-N",I0,"-sys",I0,"-",A,"-K1e",I0)') &
            generator_version, base_seed, m, n_global_design, system_id, family_code, log10_k
    end subroutine make_input_id

    subroutine make_orthogonal(m, system_id, global_row, side, q)
        integer, intent(in) :: m, system_id, global_row, side
        real(dp), intent(out) :: q(m,m)

        integer :: i, p, r
        integer(int64) :: state
        real(dp) :: angle, cosine, sine
        real(dp) :: col_p(m), col_r(m)

        q = 0.0_dp
        do i = 1, m
            q(i,i) = 1.0_dp
        end do

        state = modulo(base_seed + 104729_int64 * int(system_id, int64) &
                     + 13007_int64 * int(global_row, int64) &
                     + 1009_int64 * int(side, int64), 2147483646_int64) + 1_int64

        do p = 1, m - 1
            do r = p + 1, m
                angle = 0.50_dp * (next_uniform(state) - 0.5_dp)
                cosine = cos(angle)
                sine = sin(angle)
                col_p = q(:,p)
                col_r = q(:,r)
                q(:,p) = cosine * col_p + sine * col_r
                q(:,r) = -sine * col_p + cosine * col_r
            end do
        end do
    end subroutine make_orthogonal

    real(dp) function next_uniform(state) result(value)
        integer(int64), intent(inout) :: state
        integer(int64), parameter :: modulus = 2147483647_int64
        integer(int64), parameter :: multiplier = 48271_int64

        state = modulo(multiplier * state, modulus)
        value = real(state, dp) / real(modulus, dp)
    end function next_uniform

    subroutine exact_vector(m, system_id, global_row, x)
        integer, intent(in) :: m, system_id, global_row
        real(dp), intent(out) :: x(m)
        integer :: component

        do component = 1, m
            x(component) = sin(0.013_dp * real(17 * global_row + 5 * system_id + component, dp)) &
                         + 0.25_dp * cos(0.007_dp * real(3 * global_row + system_id + 11 * component, dp))
        end do
    end subroutine exact_vector

    subroutine build_blocks(m, n_global, system_id, global_row, family, kappa, lower, diagonal, upper)
        integer, intent(in) :: m, n_global, system_id, global_row, family
        real(dp), intent(in) :: kappa
        real(dp), intent(out) :: lower(m,m), diagonal(m,m), upper(m,m)

        integer :: component
        real(dp) :: rho, c_n, diagonal_scalar, off_diagonal_scalar
        real(dp) :: spectrum(m)
        real(dp) :: q_left(m,m), q_right(m,m), q_previous(m,m), q_next(m,m)

        if (family == 1) then
            rho = kappa
            spectrum = 1.0_dp
        else if (family == 2) then
            rho = 2.0_dp
            do component = 1, m
                spectrum(component) = kappa**(-real(component - 1, dp) / real(m - 1, dp))
            end do
        else
            error stop 'build_blocks: family must be 1 (global) or 2 (diagonal-block)'
        end if

        c_n = cos(acos(-1.0_dp) / real(n_global + 1, dp))
        diagonal_scalar = 0.5_dp * (1.0_dp + 1.0_dp / rho)
        off_diagonal_scalar = (1.0_dp - 1.0_dp / rho) / (4.0_dp * c_n)

        call make_orthogonal(m, system_id, global_row, 1, q_left)
        call make_orthogonal(m, system_id, global_row, 2, q_right)
        call weighted_orthogonal_product(m, q_left, spectrum, q_right, diagonal)
        diagonal = diagonal_scalar * diagonal

        lower = 0.0_dp
        if (global_row > 1) then
            call make_orthogonal(m, system_id, global_row - 1, 2, q_previous)
            call weighted_orthogonal_product(m, q_left, spectrum, q_previous, lower)
            lower = -off_diagonal_scalar * lower
        end if

        upper = 0.0_dp
        if (global_row < n_global) then
            call make_orthogonal(m, system_id, global_row + 1, 2, q_next)
            call weighted_orthogonal_product(m, q_left, spectrum, q_next, upper)
            upper = -off_diagonal_scalar * upper
        end if
    end subroutine build_blocks

    pure subroutine weighted_orthogonal_product(m, q_left, spectrum, q_right, result_matrix)
        integer, intent(in) :: m
        real(dp), intent(in) :: q_left(m,m), spectrum(m), q_right(m,m)
        real(dp), intent(out) :: result_matrix(m,m)
        integer :: i, j, k

        result_matrix = 0.0_dp
        do j = 1, m
            do i = 1, m
                do k = 1, m
                    result_matrix(i,j) = result_matrix(i,j) &
                                       + q_left(k,i) * spectrum(k) * q_right(k,j)
                end do
            end do
        end do
    end subroutine weighted_orthogonal_product

    subroutine build_rhs(m, n_global, system_id, global_row, lower, diagonal, upper, rhs, x_exact)
        integer, intent(in) :: m, n_global, system_id, global_row
        real(dp), intent(in) :: lower(m,m), diagonal(m,m), upper(m,m)
        real(dp), intent(out) :: rhs(m), x_exact(m)
        real(dp) :: x_previous(m), x_next(m)

        call exact_vector(m, system_id, global_row, x_exact)
        x_previous = 0.0_dp
        x_next = 0.0_dp
        if (global_row > 1) call exact_vector(m, system_id, global_row - 1, x_previous)
        if (global_row < n_global) call exact_vector(m, system_id, global_row + 1, x_next)
        rhs = matmul(lower, x_previous) + matmul(diagonal, x_exact) + matmul(upper, x_next)
    end subroutine build_rhs

    subroutine generate_one_system(m, n_global, system_id, family, kappa, lower, diagonal, upper, rhs, x_exact)
        integer, intent(in) :: m, n_global, system_id, family
        real(dp), intent(in) :: kappa
        real(dp), intent(out) :: lower(m,m,n_global), diagonal(m,m,n_global), upper(m,m,n_global)
        real(dp), intent(out) :: rhs(m,n_global), x_exact(m,n_global)
        integer :: row

        do row = 1, n_global
            call build_blocks(m, n_global, system_id, row, family, kappa, &
                              lower(:,:,row), diagonal(:,:,row), upper(:,:,row))
            call build_rhs(m, n_global, system_id, row, lower(:,:,row), diagonal(:,:,row), upper(:,:,row), &
                           rhs(:,row), x_exact(:,row))
        end do
    end subroutine generate_one_system

    subroutine build_singular_probe_blocks(m, n_global, global_row, probe, lower, diagonal, upper)
        integer, intent(in) :: m, n_global, global_row, probe
        real(dp), intent(out) :: lower(m,m), diagonal(m,m), upper(m,m)
        integer :: component

        lower = 0.0_dp
        diagonal = 0.0_dp
        upper = 0.0_dp
        select case (probe)
        case (1)
            do component = 1, m
                diagonal(component,component) = 1.0_dp
            end do
            if (global_row == 1) then
                diagonal = 0.0_dp
                do component = 1, m
                    upper(component,component) = 1.0_dp
                end do
            else if (global_row == 2) then
                do component = 1, m
                    lower(component,component) = 1.0_dp
                end do
            end if
        case (2)
            do component = 1, m
                if (global_row == 1 .or. global_row == n_global) then
                    diagonal(component,component) = 1.0_dp
                else
                    diagonal(component,component) = 2.0_dp
                end if
                if (global_row > 1) lower(component,component) = -1.0_dp
                if (global_row < n_global) upper(component,component) = -1.0_dp
            end do
        case default
            error stop 'build_singular_probe_blocks: probe must be 1 or 2'
        end select
    end subroutine build_singular_probe_blocks

    pure function singular_probe_label(probe) result(label)
        integer, intent(in) :: probe
        character(len=64) :: label

        select case (probe)
        case (1)
            label = 'singular-diagonal-block-global-nonsingular'
        case (2)
            label = 'singular-global-invertible-diagonal-blocks'
        case default
            label = 'unknown'
        end select
    end function singular_probe_label

    subroutine sequential_block_thomas(m, n_global, lower, diagonal, upper, rhs, status)
        integer, intent(in) :: m, n_global
        real(dp), intent(inout) :: lower(m,m,n_global), diagonal(m,m,n_global), upper(m,m,n_global)
        real(dp), intent(inout) :: rhs(m,n_global)
        integer, intent(out) :: status

        integer :: row, info
        integer :: pivots(m)
        real(dp) :: coefficient(m,m), multiple_rhs(m,m+1)
        real(dp) :: lower_times_upper(m,m), lower_times_rhs(m)

        status = 0
        do row = 1, n_global
            if (row == 1) then
                coefficient = diagonal(:,:,row)
                multiple_rhs(:,1) = rhs(:,row)
                multiple_rhs(:,2:m+1) = upper(:,:,row)
            else
                lower_times_upper = matmul(lower(:,:,row), upper(:,:,row-1))
                lower_times_rhs = matmul(lower(:,:,row), rhs(:,row-1))
                coefficient = diagonal(:,:,row) - lower_times_upper
                multiple_rhs(:,1) = rhs(:,row) - lower_times_rhs
                multiple_rhs(:,2:m+1) = upper(:,:,row)
            end if

            call dgesv(m, m + 1, coefficient, m, pivots, multiple_rhs, m, info)
            if (info /= 0) then
                status = info
                return
            end if
            rhs(:,row) = multiple_rhs(:,1)
            upper(:,:,row) = multiple_rhs(:,2:m+1)
        end do

        do row = n_global - 1, 1, -1
            rhs(:,row) = rhs(:,row) - matmul(upper(:,:,row), rhs(:,row+1))
        end do
    end subroutine sequential_block_thomas

    subroutine system_metrics(m, n_global, lower, diagonal, upper, original_rhs, computed, exact, &
                              residual_norm, relative_error, has_nonfinite)
        integer, intent(in) :: m, n_global
        real(dp), intent(in) :: lower(m,m,n_global), diagonal(m,m,n_global), upper(m,m,n_global)
        real(dp), intent(in) :: original_rhs(m,n_global), computed(m,n_global), exact(m,n_global)
        real(dp), intent(out) :: residual_norm, relative_error
        logical, intent(out) :: has_nonfinite

        integer :: row
        real(dp) :: residual(m), x_previous(m), x_next(m)
        real(dp) :: residual_sum, rhs_sum, error_sum, exact_sum

        residual_sum = 0.0_dp
        rhs_sum = 0.0_dp
        error_sum = 0.0_dp
        exact_sum = 0.0_dp
        do row = 1, n_global
            x_previous = 0.0_dp
            x_next = 0.0_dp
            if (row > 1) x_previous = computed(:,row-1)
            if (row < n_global) x_next = computed(:,row+1)
            residual = matmul(lower(:,:,row), x_previous) &
                     + matmul(diagonal(:,:,row), computed(:,row)) &
                     + matmul(upper(:,:,row), x_next) - original_rhs(:,row)
            residual_sum = residual_sum + sum(residual * residual)
            rhs_sum = rhs_sum + sum(original_rhs(:,row) * original_rhs(:,row))
            error_sum = error_sum + sum((computed(:,row) - exact(:,row))**2)
            exact_sum = exact_sum + sum(exact(:,row) * exact(:,row))
        end do

        residual_norm = sqrt(residual_sum / rhs_sum)
        relative_error = sqrt(error_sum / exact_sum)
        has_nonfinite = .not. ieee_is_finite(residual_norm) .or. .not. ieee_is_finite(relative_error)
    end subroutine system_metrics

    pure logical function vector_has_nonfinite(values) result(nonfinite)
        real(dp), intent(in) :: values(:)
        integer :: i

        nonfinite = .false.
        do i = 1, size(values)
            if (.not. ieee_is_finite(values(i))) then
                nonfinite = .true.
                return
            end if
        end do
    end function vector_has_nonfinite

    subroutine write_accuracy_header(unit_number)
        integer, intent(in) :: unit_number
        write(unit_number, '(A)') &
            'generator,seed,m,N,design_n_sys_total,executed_n_sys_total,system_id,family,target_K,' // &
            'theoretical_global_condition,theoretical_max_diagonal_condition,solver,mpi_ranks,gpu_count,' // &
            'residual_norm,relative_solution_error,api_status,status_available,runtime_status,nonfinite,failed,input_id'
    end subroutine write_accuracy_header

    subroutine write_accuracy_record(unit_number, m, log10_k, family, executed_nsys, system_id, solver, &
                                     mpi_ranks, gpu_count, residual_norm, relative_error, api_status, &
                                     status_available, runtime_status, nonfinite, failed)
        integer, intent(in) :: unit_number, m, log10_k, family, executed_nsys, system_id
        integer, intent(in) :: mpi_ranks, gpu_count, api_status, runtime_status
        real(dp), intent(in) :: residual_norm, relative_error
        logical, intent(in) :: status_available, nonfinite, failed
        character(len=*), intent(in) :: solver
        real(dp) :: kappa, global_condition, block_condition
        character(len=16) :: family_name
        character(len=192) :: input_id

        kappa = condition_value(log10_k)
        call theoretical_conditions(family, kappa, global_condition, block_condition)
        family_name = family_label(family)
        call make_input_id(m, log10_k, family, system_id, input_id)
        write(unit_number, '(*(g0,:,","))') trim(generator_version), base_seed, m, n_global_design, &
            nsys_design, executed_nsys, system_id, trim(family_name), kappa, global_condition, block_condition, &
            trim(solver), mpi_ranks, gpu_count, residual_norm, relative_error, api_status, &
            trim(merge('yes', 'no ', status_available)), runtime_status, &
            trim(merge('yes', 'no ', nonfinite)), trim(merge('yes', 'no ', failed)), trim(input_id)
    end subroutine write_accuracy_record

end module accuracy_support
