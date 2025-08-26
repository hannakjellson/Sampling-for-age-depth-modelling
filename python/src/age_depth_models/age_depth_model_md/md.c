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

#define MAX_BIAS 10000

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
    }
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

double get_probability_estimate(double CV_point, double *bias_centers, double *bias_heights, double *bias_widths, int bias_count, double *weights, double gamma, double beta, double sum_weights)
{
    double probability_estimate = 0;
    for (int i = 0; i < bias_count; i++)
    {
        double dx = CV_point - bias_centers[i];
        probability_estimate += weights[i] * bias_heights[i] * exp(-dx * dx / (2 * bias_widths[i] * bias_widths[i]));
    }
    if (bias_count > 0)
    {
        probability_estimate /= sum_weights;
    }
    return probability_estimate;
}

double get_CV_point(double theta, double *variables, int problem_index, double delta_c)
{
    double sum = theta;
    for (int i = 0; i < problem_index; i++)
    {
        sum -= variables[i] * delta_c;
    }
    return sum;
}

double *get_probability_estimate_gradient(double theta, double *variables, double *bias_centers, double *bias_heights, double *bias_widths, int bias_count, double *weights, double gamma, double beta, double sum_weights, double N, int problem_index, double delta_c)
{
    double CV_point = get_CV_point(theta, variables, problem_index, delta_c);
    double *dV = calloc(N, sizeof(double));
    for (int j = 0; j < problem_index; j++)
    {
        for (int i = 0; i < bias_count; i++)
        {
            double dx = CV_point - bias_centers[i];
            dV[j] += weights[i] * bias_heights[i] * (-dx / (bias_widths[i] * bias_widths[i])) * exp(-dx * dx / (2 * bias_widths[i] * bias_widths[i]));
        }
        if (bias_count > 0)
        {
            dV[j] /= sum_weights;
            dV[j] *= -delta_c; // As were taking the derivative wrt sed rates and not the collective var.
        }
    }
    return dV;
}

double bias_potential(double CV_point, double *bias_centers, double *bias_heights, double *bias_widths, int bias_count, double *weights, double gamma, double beta, double sum_weights, double Z, double DeltaE)
{
    double probability_estimate = get_probability_estimate(CV_point, bias_centers, bias_heights, bias_widths, bias_count, weights, gamma, beta, sum_weights);
    double epsilon = exp(-beta * DeltaE / (1 - (1 / gamma)));
    // printf("%f\n", CV_point);
    // printf("%f\n", probability_estimate);
    // printf("%f\n", log(epsilon));
    // printf("%f\n", log(probability_estimate));
    // printf("%f\n", log(probability_estimate / Z + epsilon));
    double V = (1 - (1 / gamma)) * log(probability_estimate / Z + epsilon) / beta;
    return V;
}

double *grad_bias(double theta, double *variables, double *bias_centers, double *bias_heights, double *bias_widths, int bias_count, double *weights, double gamma, double beta, double sum_weights, double Z, double DeltaE, int N, int problem_index, double delta_c)
{
    // Need to edit this computation such that it takes the derivatiive of s wrt the variables into account.
    double CV_point = get_CV_point(theta, variables, problem_index, delta_c);
    double probability_estimate = get_probability_estimate(CV_point, bias_centers, bias_heights, bias_widths, bias_count, weights, gamma, beta, sum_weights);
    double *probability_estimate_gradient = get_probability_estimate_gradient(theta, variables, bias_centers, bias_heights, bias_widths, bias_count, weights, gamma, beta, sum_weights, N, problem_index, delta_c);
    double epsilon = exp(-beta * DeltaE / (1 - (1 / gamma)));
    double *dV = malloc(N * sizeof(double));
    for (int i = 0; i < N; i++)
    {
        dV[i] = (1 - (1 / gamma)) * probability_estimate_gradient[i] * variables[i] / (Z * beta * ((probability_estimate / Z) + epsilon)); // Times x since we take the gradient wrt log(sed_rates) and not sed_rates.
    }
    return dV;
}

void deposit_gaussian(double CV_point, double width, double *bias_centers, double *bias_heights, double *bias_widths, int *bias_count)
{
    if (*bias_count < MAX_BIAS)
    {
        bias_centers[*bias_count] = CV_point;
        bias_heights[*bias_count] = (1 / sqrt(2 * M_PI * width * width));
        bias_widths[*bias_count] = width;
        (*bias_count)++;
    }
}

double rand_normal()
{
    double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
    double u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);
}

void md(
    int N, double H, double delta_c, const double *cs,
    double dt, double *M_vec, int num_MD, int num_chains,
    double a, double b, double theta, int num_c14_depths, int num_D18O_depths, int num_D18O_reference_times, const double *c14_ages,
    const double *c14_depths, const double *c14_sigma, const double *D18O,
    const double *D18O_depths, const double *D18O_sigma, const double *D18O_reference, const double *D18O_reference_times,
    double bias_std, int problem_index, double gamma, double beta, double d, double DeltaE,
    double **samples_out, double **energy_out, double **bias_out)
{
    // printf("problem_index received in C: %d\n", problem_index);
    // fflush(stdout);

    double *samples = malloc(num_chains * num_MD * N * sizeof(double));
    double *energy = malloc(num_chains * num_MD * sizeof(double));
    double *bias = malloc(num_chains * num_MD * sizeof(double));

#pragma omp parallel for
    for (int i = 0; i < num_chains; i++)
    {

        const gsl_rng_type *T;
        gsl_rng *r;
        gsl_rng_env_setup();
        T = gsl_rng_default;
        r = gsl_rng_alloc(T);
        gsl_rng_set(r, 41 + i);

        gsl_matrix *M = array_to_gsl_matrix(M_vec, N);
        gsl_matrix *M_inv = gsl_matrix_alloc(N, N);
        invert_matrix(M, M_inv);

        double *variables = malloc(N * sizeof(double));
        double *log_variables = malloc(N * sizeof(double));

        double bias_centers[MAX_BIAS];
        double bias_heights[MAX_BIAS];
        double bias_widths[MAX_BIAS];
        int bias_count = 0;
        double weights[num_MD];
        double sum_weights = 0;
        double sum_squared_weights = 0;
        double bias_std_j;
        double N_eff;
        double Z = 1;
        double force;
        double noise;

        for (int j = 0; j < N; j++)
        {
            variables[j] = gsl_ran_gamma(r, a, 1 / b);
            log_variables[j] = log(variables[j]);
        }

        for (int j = 0; j < num_MD; j++)
        {
            double CV_point = get_CV_point(theta, variables, problem_index, delta_c);
            double bias_pot = bias_potential(CV_point, bias_centers, bias_heights, bias_widths, bias_count, weights, gamma, beta, sum_weights, Z, DeltaE);
            weights[j] = exp(beta * bias_pot);
            sum_weights += weights[j];
            sum_squared_weights += weights[j] * weights[j];
            N_eff = sum_weights * sum_weights / sum_squared_weights;
            bias_std_j = bias_std * pow(N_eff * (d + 2) / 4, -1 / (d + 4));
            deposit_gaussian(CV_point, bias_std, bias_centers, bias_heights, bias_widths, &bias_count);
            double *gradient = grad_energy_function(N, delta_c, cs, a, b, theta, num_c14_depths,
                                                    num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma, D18O, D18O_depths, D18O_sigma,
                                                    D18O_reference, D18O_reference_times, variables);
            double *bias_gradient = grad_bias(theta, variables, bias_centers, bias_heights, bias_widths, bias_count, weights, gamma, beta, sum_weights, Z, DeltaE, N, problem_index, delta_c);
            // printf("CV_point=%f bias_pot=%f weight=%f sum_weights=%f\n",
            //    CV_point, bias_pot, weights[j], sum_weights);
            // printf("log_variables[0]=%f variables[0]=%f\n", log_variables[1], variables[1]);

            for (int k = 0; k < N; k++)
            {
                noise = sqrt(dt) * rand_normal();
                force = -gradient[k] - bias_gradient[k];
                log_variables[k] += (dt / 2) * force + noise;
                variables[k] = exp(log_variables[k]);
                samples[i * num_MD * N + j * N + k] = variables[k];
                // printf("Step %d var[%d]=%.6e log=%.6e bias_grad=%.6e energy_grad=%.6e\n",
                //    j, k, variables[k], log_variables[k], bias_gradient[k], gradient[k]);
            }
            energy[i * num_MD + j] = energy_function(N, delta_c, cs, a, b, theta, num_c14_depths,
                                                     num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma, D18O, D18O_depths, D18O_sigma,
                                                     D18O_reference, D18O_reference_times, variables);
            bias[i * num_MD + j] = bias_pot;
            free(gradient);
            free(bias_gradient);
        }

        free(variables);
        free(log_variables);
        gsl_matrix_free(M);
        gsl_matrix_free(M_inv);
        gsl_rng_free(r);
    }

    *samples_out = samples;
    *energy_out = energy;
    *bias_out = bias; // read this in the plotting script to reweight the samples. Reducing the step size a lot made the algorithm not diverge, dont know if this is good though. Might be other issues hidden by this!
}
