#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <stdbool.h>
#include <omp.h>

int get_index(int N, const double *cs, const double *c14_depths, int depth_index)
{
    int low = 0, high = N - 1;
    while (low <= high)
    {
        int mid = (low + high) / 2;
        if (cs[mid] < c14_depths[depth_index])
        {
            low = mid + 1;
        }
        else
        {
            high = mid - 1;
        }
    }
    return (low > 0) ? low - 1 : 0;
}

gsl_matrix *create_identity_matrix(int n)
{
    gsl_matrix *I = gsl_matrix_alloc(n, n);
    gsl_matrix_set_zero(I);
    for (int i = 0; i < n; i++)
    {
        gsl_matrix_set(I, i, i, 1.0);
    }
    return I;
}

void invert_matrix(gsl_matrix *A, gsl_matrix *A_inv)
{
    int s;
    gsl_permutation *p = gsl_permutation_alloc(A->size1);
    gsl_matrix *A_copy = gsl_matrix_alloc(A->size1, A->size2);
    gsl_matrix_memcpy(A_copy, A);

    gsl_linalg_LU_decomp(A_copy, p, &s);
    gsl_linalg_LU_invert(A_copy, p, A_inv);

    gsl_permutation_free(p);
    gsl_matrix_free(A_copy);
}

double *matrix_vector_multiply(gsl_matrix *A, gsl_vector *B, int n)
{
    double *vec = calloc(n, sizeof(double));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            vec[i] += gsl_matrix_get(A, i, j) * gsl_vector_get(B, j);
        }
    }
    return vec;
}

double quadratic_form(gsl_vector *x, const gsl_matrix *A, int n)
{
    double result = 0.0;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            result += gsl_vector_get(x, i) * gsl_matrix_get(A, i, j) * gsl_vector_get(x, j);
        }
    }
    return result;
}

double find_min(double *arr, int size)
{
    double min = INT_MAX;
    for (int i = 0; i < size; ++i)
    {
        if (arr[i] < min)
            min = arr[i];
    }
    return min;
}

gsl_matrix *array_to_gsl_matrix(const double *arr, int N)
{
    gsl_matrix *mat = gsl_matrix_alloc(N, N);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            gsl_matrix_set(mat, i, j, arr[i * N + j]);
        }
    }

    return mat;
}

void interpolate_D18O(
    int num_D18O_depths, int num_D18O_reference_times,
    const double *D18O_times, const double *D18O_reference, const double *D18O_reference_times,
    double *D18O_reference_interp, double *D18O_interp_derivative, const double *D18O_depths)
{
    for (int i = 0; i < num_D18O_depths; i++)
    {
        int index = get_index(num_D18O_reference_times, D18O_reference_times, D18O_times, i);

        double t0 = D18O_reference_times[index];
        double t1 = D18O_reference_times[index + 1];
        double y0 = D18O_reference[index];
        double y1 = D18O_reference[index + 1];
        double t = D18O_times[i];

        double dt = t1 - t0;
        double dy = y1 - y0;

        double slope = dy / dt;
        D18O_interp_derivative[i] = slope;
        D18O_reference_interp[i] = y0 + slope * (t - t0);
        // #ifdef DEBUG
        //     printf("index: %d, i: %d, D18O_age: %f, D18O_depths: %f, reference_val: %f, reference_time: %f, slope: %f, y0:\n", index, i, D18O_times[i], D18O_depths[i], D18O_reference[index], D18O_reference_times[index], slope, y0);
        //     getchar();
        // #endif
    }
    // #ifdef DEBUG
    // for  (int i = 0; i < num_D18O_depths; i++) {
    //     printf("D18O value at %d: %f, D18O depth: %f, D18O age: %f\n", i, D18O_reference_interp[i], D18O_depths[i], D18O_times[i]);
    // }
    // getchar();

    // #endif
}

void *expected_ages(
    int N, double delta_c, const double *cs, double theta, int num_depths,
    const double *depths, const double *sed_rates, double *ages_out)
{
    double *cumulative_sum_vec = calloc(N + 1, sizeof(double));
    double cumulative_sum;
    for (int i = 1; i < N + 1; i++)
    {
        cumulative_sum_vec[i] += cumulative_sum_vec[i - 1] + sed_rates[i - 1] * delta_c;
    }
    for (int i = 0; i < num_depths; i++)
    {
        int index = get_index(N, cs, depths, i);
        cumulative_sum = cumulative_sum_vec[index];
        cumulative_sum += sed_rates[index] * (depths[i] - cs[index]);
        ages_out[i] = theta - cumulative_sum;
    }
    free(cumulative_sum_vec);
}

double energy_function(
    int N, double delta_c, const double *cs, double a, double b, double theta,
    int num_c14_depths, int num_D18O_depths, int num_D18O_reference_times, const double *c14_ages, const double *c14_depths,
    const double *c14_sigma, const double *D18O, const double *D18O_depths, const double *D18O_sigma,
    const double *D18O_reference, const double *D18O_reference_times, const double *sed_rates)
{

    //-log(p(log(sed_rates)|data))

    // Prior
    double prior = 0.0;
    for (int i = 0; i < N; i++)
    {
        prior += -a * log(sed_rates[i]) + b * sed_rates[i];
    }

    // Conditional from C14
    double *expected_c14_ages = malloc(num_c14_depths * sizeof(double));
    expected_ages(N, delta_c, cs, theta, num_c14_depths, c14_depths, sed_rates, expected_c14_ages);

    double c14_conditional = 0.0;
    for (int i = 0; i < num_c14_depths; i++)
    {
        double diff = c14_ages[i] - expected_c14_ages[i];
        c14_conditional += diff * diff / (2 * c14_sigma[i] * c14_sigma[i]);
    }

    // Conditional from D18O
    double *D18O_ages = malloc(num_D18O_depths * sizeof(double));
    expected_ages(N, delta_c, cs, theta, num_D18O_depths, D18O_depths, sed_rates, D18O_ages);

    double *D18O_reference_interp = malloc(num_D18O_depths * sizeof(double));
    double *D18O_interp_derivative = malloc(num_D18O_depths * sizeof(double));
    interpolate_D18O(
        num_D18O_depths, num_D18O_reference_times, D18O_ages, D18O_reference, D18O_reference_times, D18O_reference_interp, D18O_interp_derivative, D18O_depths);

    double D18O_conditional = 0.0;
    for (int i = 0; i < num_D18O_depths; i++)
    {
        double diff = D18O[i] - D18O_reference_interp[i];
        D18O_conditional += diff * diff / (2 * D18O_sigma[i] * D18O_sigma[i]);
    }

    free(expected_c14_ages);
    free(D18O_reference_interp);
    free(D18O_interp_derivative);
    free(D18O_ages);

    return prior + c14_conditional + D18O_conditional;
}

double *grad_energy_function(
    int N, double delta_c, const double *cs, double a, double b, double theta,
    int num_c14_depths, int num_D18O_depths, int num_D18O_reference_times, const double *c14_ages, const double *c14_depths,
    const double *c14_sigma, const double *D18O, const double *D18O_depths, const double *D18O_sigma,
    const double *D18O_reference, const double *D18O_reference_times, const double *sed_rates)
{
    // derivative of -log(p(log(sed_rates)|data)) with respect to log(sed_rates)
    double f_l;
    double c14_conditional_term;

    double D18O_conditional_term;
    double *D18O_reference_interp = malloc(num_D18O_depths * sizeof(double));
    double *D18O_interp_derivative = malloc(num_D18O_depths * sizeof(double));

    double *expected_age = malloc(num_c14_depths * sizeof(double));
    expected_ages(N, delta_c, cs, theta, num_c14_depths, c14_depths, sed_rates, expected_age);
    double *gradient = calloc(N, sizeof(double));

    double *D18O_ages = malloc(num_D18O_depths * sizeof(double));
    expected_ages(N, delta_c, cs, theta, num_D18O_depths, D18O_depths, sed_rates, D18O_ages);

    interpolate_D18O(
        num_D18O_depths, num_D18O_reference_times, D18O_ages, D18O_reference, D18O_reference_times, D18O_reference_interp, D18O_interp_derivative, D18O_depths);

    int *c14_indices = malloc(num_c14_depths * sizeof(int));
    double *inv_c14_var = malloc(num_c14_depths * sizeof(double));
    for (int i = 0; i < num_c14_depths; i++)
    {
        c14_indices[i] = get_index(N, cs, c14_depths, i);
        inv_c14_var[i] = 1.0 / (c14_sigma[i] * c14_sigma[i]);
    }

    int *D18O_indices = malloc(num_D18O_depths * sizeof(int));
    double *inv_D18O_var = malloc(num_D18O_depths * sizeof(double));
    for (int i = 0; i < num_D18O_depths; i++)
    {
        D18O_indices[i] = get_index(N, cs, D18O_depths, i);
        inv_D18O_var[i] = 1.0 / (D18O_sigma[i] * D18O_sigma[i]);
    }

    for (int l = 0; l < N; l++)
    {

        // Prior
        double grad = -a + b * sed_rates[l];

        // Conditional from C14 data
        for (int i = 0; i < num_c14_depths; i++)
        {
            int j = c14_indices[i];
            double f_l = (l < j) ? -delta_c : (l == j) ? -(c14_depths[i] - cs[j])
                                                       : 0.0;
            if (f_l != 0.0)
            {
                double diff = c14_ages[i] - expected_age[i];
                grad -= diff * f_l * sed_rates[l] * inv_c14_var[i];
            }
        }

        // Conditional from D18O
        for (int i = 0; i < num_D18O_depths; i++)
        {
            int j = D18O_indices[i];
            double f_l = (l < j) ? -delta_c : (l == j) ? -(D18O_depths[i] - cs[j])
                                                       : 0.0;
            if (f_l != 0.0)
            {
                double diff = D18O[i] - D18O_reference_interp[i];
                grad -= diff * D18O_interp_derivative[i] * f_l * sed_rates[l] * inv_D18O_var[i];
            }
        }
        gradient[l] = grad;
    }

    // #ifdef DEBUG
    //     // Gradient check: finite differences
    //     double eps = 1e-6;
    //     double *log_sed_rates_copy = malloc(N * sizeof(double));
    //     double *sed_rates_copy = malloc(N * sizeof(double));

    //     for (int i=0; i<N; i++){
    //         log_sed_rates_copy[i]=log(sed_rates[i]);
    //         sed_rates_copy[i]=sed_rates[i];
    //     }
    //     for (int l = 0; l < N; l++) {
    //         double orig = log_sed_rates_copy[l];
    //         sed_rates_copy[l] = exp(orig + eps);
    //         double f_plus = energy_function(N, delta_c, cs, a, b, theta, num_c14_depths, num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma,
    //                 D18O, D18O_depths, D18O_sigma, D18O_reference, D18O_reference_times, sed_rates_copy); // with updated sed_rates
    //         sed_rates_copy[l] = exp(orig - eps);
    //         double f_minus = energy_function(N, delta_c, cs, a, b, theta, num_c14_depths, num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma,
    //                 D18O, D18O_depths, D18O_sigma, D18O_reference, D18O_reference_times, sed_rates_copy);
    //         sed_rates_copy[l] = exp(orig);

    //         double fd_grad = (f_plus - f_minus) / (2 * eps);
    //         printf("Gradient check at %d: analytic = %f, finite-diff = %f, diff = %f\n",
    //             l, gradient[l], fd_grad, gradient[l] - fd_grad);
    //     }
    //     getchar();
    //     fflush(stdout);
    // #endif

    free(expected_age);
    free(D18O_ages);
    free(D18O_reference_interp);
    free(D18O_interp_derivative);
    free(c14_indices);
    free(inv_c14_var);
    free(D18O_indices);
    free(inv_D18O_var);
    return gradient;
}

void *hmc(
    int N, double H, double delta_c, const double *cs,
    double dt, double *M_vec, int L, int num_MH, int num_chains,
    double a, double b, double theta, int num_c14_depths, int num_D18O_depths, int num_D18O_reference_times, const double *c14_ages,
    const double *c14_depths, const double *c14_sigma, const double *D18O,
    const double *D18O_depths, const double *D18O_sigma, const double *D18O_reference, const double *D18O_reference_times,
    double **samples_out, double **energy_out)
{

    double *samples = malloc(num_chains * num_MH * N * sizeof(double));
    double *energy = malloc(num_chains * num_MH * sizeof(double));

#pragma omp parallel for
    for (int i = 0; i < num_chains; i++)
    {

        const gsl_rng_type *T;
        gsl_rng *r;
        gsl_rng_env_setup();
        T = gsl_rng_default;
        r = gsl_rng_alloc(T);
        gsl_rng_set(r, 41 + i);

        gsl_vector *momentum = gsl_vector_calloc(N);
        gsl_vector *momentum_init = gsl_vector_calloc(N);
        gsl_matrix *M = array_to_gsl_matrix(M_vec, N);
        gsl_matrix *M_inv = gsl_matrix_alloc(N, N);
        invert_matrix(M, M_inv);

        double *variables = malloc(N * sizeof(double));
        double *log_variables = malloc(N * sizeof(double));
        double *variables_init = malloc(N * sizeof(double));
        double mean_acceptance = 0.0;

        for (int j = 0; j < N; j++)
        {
            variables[j] = gsl_ran_gamma(r, a, 1 / b);
            log_variables[j] = log(variables[j]);
        }

        for (int j = 0; j < num_MH; j++)
        {
            memcpy(variables_init, variables, N * sizeof(double));
            for (int m = 0; m < N; m++)
            {
                gsl_vector_set(momentum, m, gsl_ran_gaussian(r, 1.0));
            }
            gsl_vector_memcpy(momentum_init, momentum);

            for (int k = 0; k < L; k++)
            {
                double *gradient = grad_energy_function(N, delta_c, cs, a, b, theta, num_c14_depths,
                                                        num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma, D18O, D18O_depths, D18O_sigma,
                                                        D18O_reference, D18O_reference_times, variables);
                for (int n = 0; n < N; n++)
                {
                    gsl_vector_set(momentum, n, gsl_vector_get(momentum, n) - (dt / 2) * gradient[n]);
                }
                free(gradient);
                double *some_product = matrix_vector_multiply(M_inv, momentum, N);
                // #ifdef DEBUG
                //     for (int i=0; i<N; i++){
                //         printf("Variables at %d: %f\n",
                //             i, variables[i]);
                //     }
                //     getchar();

                // #endif
                for (int n = 0; n < N; n++)
                {
                    log_variables[n] += dt * some_product[n];
                    variables[n] = exp(log_variables[n]);
                }
                free(some_product);

                gradient = grad_energy_function(N, delta_c, cs, a, b, theta, num_c14_depths,
                                                num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma, D18O, D18O_depths, D18O_sigma,
                                                D18O_reference, D18O_reference_times, variables);

                for (int n = 0; n < N; n++)
                {
                    gsl_vector_set(momentum, n, gsl_vector_get(momentum, n) - (dt / 2) * gradient[n]);
                }
                free(gradient);
            }
            double logp_new = energy_function(N, delta_c, cs, a, b, theta, num_c14_depths, num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma,
                                              D18O, D18O_depths, D18O_sigma, D18O_reference, D18O_reference_times, variables);
            double logp_old = energy_function(N, delta_c, cs, a, b, theta, num_c14_depths, num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma,
                                              D18O, D18O_depths, D18O_sigma, D18O_reference, D18O_reference_times, variables_init);

            double kinetic_new = 0.5 * quadratic_form(momentum, M_inv, N);
            double kinetic_old = 0.5 * quadratic_form(momentum_init, M_inv, N);

            double acceptance_prob = fmin(1, exp(logp_old + kinetic_old - logp_new - kinetic_new));
            mean_acceptance += acceptance_prob;
            double random_number = gsl_rng_uniform(r);

            bool keep_variables = random_number > acceptance_prob;
            for (int m = 0; m < N; m++)
            {
                if (keep_variables)
                {
                    variables[m] = variables_init[m];
                }
                log_variables[m] = log(variables[m]);
                samples[i * num_MH * N + j * N + m] = variables[m];
            }
            if (keep_variables)
            {
                logp_new = logp_old;
            }
            energy[i * num_MH + j] = logp_new;
        }
        mean_acceptance /= num_MH;
        printf("%f\n", mean_acceptance);
        free(variables);
        free(variables_init);
        free(log_variables);
        gsl_vector_free(momentum);
        gsl_vector_free(momentum_init);
        gsl_matrix_free(M);
        gsl_matrix_free(M_inv);
    }

    *samples_out = samples;
    *energy_out = energy;

    // return samples_out;
}
