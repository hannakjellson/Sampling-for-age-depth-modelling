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

int binary_search(const double *arr, int n, double target)
{
    int left = 0, right = n;

    while (left < right)
    {
        int mid = (left + right) / 2;
        if (arr[mid] < target)
            left = mid + 1;
        else
            right = mid;
    }

    return left;
}

void interpolate_D18O(
    int num_D18O_depths, int num_D18O_reference_times,
    const double *D18O_times, const double *D18O_reference, const double *D18O_reference_times,
    double D18O_reference_interp[num_D18O_depths], double D18O_interp_derivative[num_D18O_depths], const double *D18O_depths)
{
    for (int i = 0; i < num_D18O_depths; i++)
    {
        int index = binary_search(D18O_reference_times, num_D18O_reference_times, D18O_times[i]) - 1;

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

void expected_ages(
    int N, double delta_c, const double *cs, double theta, int num_depths,
    const double *depths, const double *sed_rates, int indices[num_depths], double ages_out[num_depths])
{
    double cumulative_sum_vec[N + 1];
    cumulative_sum_vec[0] = 0;
    double cumulative_sum;
    for (int i = 1; i < N + 1; i++)
    {
        cumulative_sum_vec[i] = cumulative_sum_vec[i - 1] + sed_rates[i - 1] * delta_c;
    }
    for (int i = 0; i < num_depths; i++)
    {
        int index = indices[i];
        cumulative_sum = cumulative_sum_vec[index];
        cumulative_sum += sed_rates[index] * (depths[i] - cs[index]);
        ages_out[i] = theta - cumulative_sum;
    }
}

double energy_function(
    int N, double delta_c, const double *cs, double a, double b, double theta,
    int num_c14_depths, int num_D18O_depths, int num_D18O_reference_times, const double *c14_ages, const double *c14_depths,
    const double *c14_sigma, int c14_depth_indices[num_c14_depths], double inv_c14_var[num_c14_depths], double c14_expected_ages[num_c14_depths], const double *D18O, const double *D18O_depths, const double *D18O_sigma,
    int D18O_depth_indices[num_D18O_depths], double inv_D18O_var[num_D18O_depths], const double *D18O_reference, const double *D18O_reference_times, const double *sed_rates)
{

    //-log(p(log(sed_rates)|data))

    // Prior
    double prior = 0.0;
    for (int i = 0; i < N; i++)
    {
        prior += -a * log(sed_rates[i]) + b * sed_rates[i];
    }

    // Conditional from C14
    double expected_c14_ages[num_c14_depths];
    expected_ages(N, delta_c, cs, theta, num_c14_depths, c14_depths, sed_rates, c14_depth_indices, expected_c14_ages);

    double c14_conditional = 0.0;
    for (int i = 0; i < num_c14_depths; i++)
    {
        double diff = c14_ages[i] - expected_c14_ages[i];
        c14_conditional += inv_c14_var[i] * diff * diff / (2);
    }

    // Conditional from D18O
    double D18O_ages[num_D18O_depths];
    expected_ages(N, delta_c, cs, theta, num_D18O_depths, D18O_depths, sed_rates, D18O_depth_indices, D18O_ages);

    double D18O_reference_interp[num_D18O_depths];
    double D18O_interp_derivative[num_D18O_depths];
    interpolate_D18O(
        num_D18O_depths, num_D18O_reference_times, D18O_ages, D18O_reference, D18O_reference_times, D18O_reference_interp, D18O_interp_derivative, D18O_depths);

    double D18O_conditional = 0.0;
    for (int i = 0; i < num_D18O_depths; i++)
    {
        double diff = D18O[i] - D18O_reference_interp[i];
        D18O_conditional += inv_D18O_var[i] * diff * diff / 2;
    }
    return prior + c14_conditional + D18O_conditional;
}

void grad_energy_function(
    int N, double delta_c, const double *cs, double a, double b, double theta,
    int num_c14_depths, int num_D18O_depths, int num_D18O_reference_times, const double *c14_ages, const double *c14_depths,
    const double *c14_sigma, int c14_indices[num_c14_depths], double inv_c14_var[num_c14_depths], double expected_c14_ages[num_c14_depths], const double *D18O, const double *D18O_depths, const double *D18O_sigma,
    int D18O_indices[num_D18O_depths], double inv_D18O_var[num_D18O_depths], double expected_D18O_ages[num_D18O_depths], const double *D18O_reference, const double *D18O_reference_times, const double *sed_rates, double gradient[N])
{
    // derivative of -log(p(log(sed_rates)|data)) with respect to log(sed_rates)
    double f_l;
    double c14_conditional_term;

    double D18O_conditional_term;
    double D18O_reference_interp[num_D18O_depths];
    double D18O_interp_derivative[num_D18O_depths];

    interpolate_D18O(
        num_D18O_depths, num_D18O_reference_times, expected_D18O_ages, D18O_reference, D18O_reference_times, D18O_reference_interp, D18O_interp_derivative, D18O_depths);

    for (int l = 0; l < N; l++)
    {
        // Prior
        double grad = -a + b * sed_rates[l];

        // Conditional from C14 data
        for (int i = 0; i < num_c14_depths; i++)
        {
            int j = c14_indices[i];
            f_l = (l < j) ? -delta_c : (l == j) ? -(c14_depths[i] - cs[j])
                                                : 0.0;
            if (f_l != 0.0)
            {
                double diff = c14_ages[i] - expected_c14_ages[i];
                grad -= diff * f_l * sed_rates[l] * inv_c14_var[i];
            }
        }

        // Conditional from D18O
        for (int i = 0; i < num_D18O_depths; i++)
        {
            int j = D18O_indices[i];
            f_l = (l < j) ? -delta_c : (l == j) ? -(D18O_depths[i] - cs[j])
                                                : 0.0;
            if (f_l != 0.0)
            {
                double diff = D18O[i] - D18O_reference_interp[i];
                grad -= diff * D18O_interp_derivative[i] * f_l * sed_rates[l] * inv_D18O_var[i];
            }
        }
        gradient[l] = grad;
    }
}

double get_CV_point(int N, double theta, double variables[N], int problem_index, double delta_c)
{
    double sum = theta;
    for (int i = 0; i < problem_index; i++)
    {
        sum -= variables[i] * delta_c;
    }
    return sum;
}

double get_probability_estimate(double CV_point, double *bias_centers, double *bias_heights, double *bias_widths, int bias_count, double *kernel_weights, double gamma, double beta, double sum_weights)
{
    double probability_estimate = 0.0;
    for (int i = 0; i < bias_count; i++)
    {
        double dx = CV_point - bias_centers[i];
        probability_estimate += kernel_weights[i] * bias_heights[i] * exp(-dx * dx / (2 * bias_widths[i] * bias_widths[i]));
    }
    if (bias_count > 0 && sum_weights > 0)
    {
        probability_estimate /= sum_weights;
    }
    return probability_estimate;
}

double get_probability_estimate_gradient(double CV_point, double *bias_centers, double *bias_heights, double *bias_widths, int bias_count, double *kernel_weights, double gamma, double beta, double sum_weights)
{
    double dp = 0.0;
    for (int i = 0; i < bias_count; i++)
    {
        double dx = CV_point - bias_centers[i];
        dp += kernel_weights[i] * bias_heights[i] * (-dx / (bias_widths[i] * bias_widths[i])) * exp(-dx * dx / (2 * bias_widths[i] * bias_widths[i]));
    }
    if (bias_count > 0 && sum_weights > 0)
    {
        dp /= sum_weights;
    }
    return dp;
}

double compute_Zn(double *bias_centers, double *bias_heights, double *bias_widths, int bias_count, double *kernel_weights, double gamma, double beta, double sum_weights)
{
    double Zn = 0.0;
    for (int i = 0; i < bias_count; i++)
    {
        double s = bias_centers[i];
        double p = get_probability_estimate(s, bias_centers, bias_heights, bias_widths, bias_count, kernel_weights, gamma, beta, sum_weights);
        Zn += p;
    }
    if (bias_count > 0)
        Zn /= bias_count;
    return Zn;
}

double bias_potential(double CV_point, double *bias_centers, double *bias_heights, double *bias_widths, int bias_count, double *kernel_weights, double gamma, double beta, double sum_weights, double Z, double DeltaE)
{
    double probability_estimate = get_probability_estimate(CV_point, bias_centers, bias_heights, bias_widths, bias_count, kernel_weights, gamma, beta, sum_weights);
    double epsilon = exp(-beta * DeltaE / (1.0 - (1.0 / gamma)));
    double V = (1.0 - (1.0 / gamma)) * log((probability_estimate / Z) + epsilon) / beta;
    return V;
}

void grad_bias(int N, double variables[N], double CV_point, double theta, int problem_index, double delta_c, double *bias_centers, double *bias_heights, double *bias_widths, int bias_count, double *kernel_weights, double gamma, double beta, double sum_weights, double Z, double DeltaE, double gradient[N])
{
    double probability_estimate = get_probability_estimate(CV_point, bias_centers, bias_heights, bias_widths, bias_count, kernel_weights, gamma, beta, sum_weights);
    double probability_estimate_grad = get_probability_estimate_gradient(CV_point, bias_centers, bias_heights, bias_widths, bias_count, kernel_weights, gamma, beta, sum_weights);
    double epsilon = exp(-beta * DeltaE / (1.0 - (1.0 / gamma)));
    double dV = (1.0 - (1.0 / gamma)) * probability_estimate_grad / (Z * beta * ((probability_estimate / Z) + epsilon));
    for (int i = 0; i < N; i++)
    {
        gradient[i] = (i < problem_index) ? dV * variables[i] * (-delta_c) : 0;
    }
}

void new_kernel(double *bias_centers, double *bias_heights, double *bias_widths, double *kernel_weights, double *sum_squared_weights, int *bias_count, int index, bool right)
{
    double new_bias_height = right ? bias_heights[index + 1] + bias_heights[index] : bias_heights[index - 1] + bias_heights[index];
    double new_bias_center = right ? (1.0 / new_bias_height) * (bias_heights[index] * bias_centers[index] + bias_heights[index + 1] * bias_centers[index + 1]) : (1.0 / new_bias_height) * (bias_heights[index] * bias_centers[index] + bias_heights[index - 1] * bias_centers[index - 1]);
    double new_bias_width = right ? sqrt((1.0 / new_bias_height) * (bias_heights[index] * (pow(bias_widths[index], 2) + pow(bias_centers[index], 2)) + bias_heights[index + 1] * (pow(bias_widths[index + 1], 2) + pow(bias_centers[index + 1], 2))) - (new_bias_center * new_bias_center)) : sqrt((1.0 / new_bias_height) * (bias_heights[index] * (pow(bias_widths[index], 2) + pow(bias_centers[index], 2)) + bias_heights[index - 1] * (pow(bias_widths[index - 1], 2) + pow(bias_centers[index - 1], 2))) - (new_bias_center * new_bias_center));

    int new_index = right ? index : index - 1;
    bias_heights[new_index] = new_bias_height;
    bias_centers[new_index] = new_bias_center;
    bias_widths[new_index] = new_bias_width;

    (*sum_squared_weights) -= kernel_weights[new_index] * kernel_weights[new_index];
    (*sum_squared_weights) -= kernel_weights[new_index + 1] * kernel_weights[new_index + 1];
    (*sum_squared_weights) += (kernel_weights[new_index] + kernel_weights[new_index + 1]) * (kernel_weights[new_index] + kernel_weights[new_index + 1]);
    kernel_weights[new_index] = kernel_weights[new_index] + kernel_weights[new_index + 1];

    memmove(&bias_heights[new_index + 1], &bias_heights[new_index + 2], (*bias_count - new_index - 2) * sizeof(double));
    memmove(&bias_centers[new_index + 1], &bias_centers[new_index + 2], (*bias_count - new_index - 2) * sizeof(double));
    memmove(&bias_widths[new_index + 1], &bias_widths[new_index + 2], (*bias_count - new_index - 2) * sizeof(double));
    memmove(&kernel_weights[new_index + 1], &kernel_weights[new_index + 2], (*bias_count - new_index - 2) * sizeof(double));

    (*bias_count)--;
}

void merge_kernels(int index, double *bias_centers, double *bias_heights, double *bias_widths, double *kernel_weights, double *sum_squared_weights, int *bias_count, double distance_threshold)
{
    double distance = 0;
    while (*bias_count > 1)
    {
        double distance_right = index + 1 < *bias_count ? bias_centers[index + 1] - bias_centers[index] : DBL_MAX;
        double distance_left = index - 1 >= 0 ? bias_centers[index] - bias_centers[index - 1] : DBL_MAX; // Both of them wont be DBL_MAX as bias_count > 2
        bool right = distance_right < distance_left ? true : false;
        distance = right ? distance_right : distance_left;
        if (distance < distance_threshold)
        {
            new_kernel(bias_centers, bias_heights, bias_widths, kernel_weights, sum_squared_weights, bias_count, index, right);
            index = right ? index : index - 1;
        }
        else
            break;
    }
}

void deposit_gaussian(double CV_point, double width, double *bias_centers, double *bias_heights, double *bias_widths, double *kernel_weights, double current_weight, double *sum_squared_weights, int *bias_count, int iterations, double distance_threshold)
{
    if (*bias_count < MAX_BIAS)
    {
        int index = binary_search(bias_centers, *bias_count, CV_point);
        memmove(&bias_centers[index + 1], &bias_centers[index], (*bias_count - index) * sizeof(double));
        memmove(&bias_heights[index + 1], &bias_heights[index], (*bias_count - index) * sizeof(double));
        memmove(&bias_widths[index + 1], &bias_widths[index], (*bias_count - index) * sizeof(double));
        memmove(&kernel_weights[index + 1], &kernel_weights[index], (*bias_count - index) * sizeof(double));

        bias_centers[index] = CV_point;
        bias_heights[index] = (1.0 / sqrt(2.0 * M_PI * width * width));
        bias_widths[index] = width;
        kernel_weights[index] = current_weight;
        (*bias_count)++;
        merge_kernels(index, bias_centers, bias_heights, bias_widths, kernel_weights, sum_squared_weights, bias_count, distance_threshold);
    }
}

void hmc(
    int N, double H, double delta_c, const double *cs,
    double dt, int num_dt, int num_HMC, int num_chains, int num_samples, int problem_index, double bias_sigma,
    double a, double b, double theta, int num_c14_depths, int num_D18O_depths, int num_D18O_reference_times, const double *c14_ages,
    const double *c14_depths, const double *c14_sigma, const double *D18O,
    const double *D18O_depths, const double *D18O_sigma, const double *D18O_reference, const double *D18O_reference_times, double gamma, double beta, double DeltaE, double distance_threshold,
    double *samples_out, double *energy_out, double *bias_out)
{

    int c14_depth_indices[num_c14_depths];
    double inv_c14_var[num_c14_depths];
    for (int i = 0; i < num_c14_depths; i++)
    {
        c14_depth_indices[i] = binary_search(cs, N + 1, c14_depths[i]) - 1;
        inv_c14_var[i] = 1.0 / (c14_sigma[i] * c14_sigma[i]);
    }

    int D18O_depth_indices[num_D18O_depths];
    double inv_D18O_var[num_D18O_depths];
    for (int i = 0; i < num_D18O_depths; i++)
    {
        D18O_depth_indices[i] = binary_search(cs, N + 1, D18O_depths[i]) - 1;
        inv_D18O_var[i] = 1.0 / (D18O_sigma[i] * D18O_sigma[i]);
    }

#pragma omp parallel for
    for (int i = 0; i < num_chains; i++)
    {
        double Z = 1.0;
        const gsl_rng_type *T;
        gsl_rng *r;

        gsl_rng_env_setup();
        T = gsl_rng_default;
        r = gsl_rng_alloc(T);
        gsl_rng_set(r, 41 + i);

        double bias_centers[MAX_BIAS];
        double bias_widths[MAX_BIAS];
        double bias_heights[MAX_BIAS];
        int bias_count = 0;

        double *weights = malloc(num_samples * sizeof(double));
        double *kernel_weights = malloc(num_samples * sizeof(double));
        double sum_weights = 0;
        double sum_squared_weights = 0;

        double N_eff;

        double momentum[N];
        double momentum_init[N];

        double variables[N];
        double log_variables[N];
        double variables_init[N];
        double gradient[N];
        double bias_gradient[N];
        double c14_expected_ages[num_c14_depths];
        double D18O_expected_ages[num_D18O_depths];
        double mean_acceptance = 0.0;
        double logp_new;
        double logp_old;
        double kinetic_new;
        double kinetic_old;
        double CV_point;
        double CV_point_init;

        double new_variables[N];
        for (int l = 0; l < N; l++)
        {
            variables[l] = gsl_ran_gamma(r, a, 1 / b);
            log_variables[l] = log(variables[l]);
        }

        double min_energy = energy_function(N, delta_c, cs, a, b, theta, num_c14_depths, num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma, c14_depth_indices, inv_c14_var, c14_expected_ages, D18O, D18O_depths, D18O_sigma, D18O_depth_indices, inv_D18O_var, D18O_reference, D18O_reference_times, variables);
        for (int j = 0; j < 100; j++)
        {
            for (int l = 0; l < N; l++)
            {
                new_variables[l] = gsl_ran_gamma(r, a, 1 / b);
            }

            double energy = energy_function(N, delta_c, cs, a, b, theta, num_c14_depths, num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma, c14_depth_indices, inv_c14_var, c14_expected_ages, D18O, D18O_depths, D18O_sigma, D18O_depth_indices, inv_D18O_var, D18O_reference, D18O_reference_times, new_variables);
            if (energy < min_energy)
            {
                min_energy = energy;
                for (int l = 0; l < N; l++)
                {
                    variables[l] = new_variables[l];
                    log_variables[l] = log(variables[l]);
                }
            }
        }

        for (int l = 0; l < num_samples; l++)
        {
            if (isnan(variables[0]))
            {
                printf("Error: NaN detected in chain: %d!\n", i);
                exit(EXIT_FAILURE);
            }
            CV_point = get_CV_point(N, theta, variables, problem_index, delta_c);
            double potential = bias_potential(CV_point, bias_centers, bias_heights, bias_widths, bias_count, kernel_weights, gamma, beta, sum_weights, Z, DeltaE);
            weights[l] = exp(beta * potential);
            sum_weights += weights[l];
            sum_squared_weights += weights[l] * weights[l];
            N_eff = (sum_squared_weights > 0) ? sum_weights * sum_weights / sum_squared_weights : 1;
            double bias_std_j = bias_sigma * pow(N_eff * (N + 2) / 4.0, -1.0 / (N + 4.0));

            for (int j = 0; j < N; j++)
            {
                samples_out[N * num_samples * i + N * l + j] = variables[j];
            }
            energy_out[i * num_samples + l] = energy_function(N, delta_c, cs, a, b, theta, num_c14_depths, num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma, c14_depth_indices, inv_c14_var, c14_expected_ages, D18O, D18O_depths, D18O_sigma, D18O_depth_indices, inv_D18O_var, D18O_reference, D18O_reference_times, variables);
            bias_out[i * num_samples + l] = potential;
            deposit_gaussian(CV_point, bias_std_j, bias_centers, bias_heights, bias_widths, kernel_weights, weights[l], &sum_squared_weights, &bias_count, l, distance_threshold);
            Z = compute_Zn(bias_centers, bias_heights, bias_widths, bias_count, kernel_weights, gamma, beta, sum_weights);
            if (l % 1000 == 0)
            {
                printf("chain: %d, iteration: %d\n", i, l);
            }

            for (int j = 0; j < num_HMC; j++)
            {
                memcpy(variables_init, variables, N * sizeof(double));
                for (int m = 0; m < N; m++)
                {
                    momentum[m] = gsl_ran_gaussian(r, 1.0);
                }
                memcpy(momentum_init, momentum, N * sizeof(double));

                expected_ages(N, delta_c, cs, theta, num_c14_depths, c14_depths, variables, c14_depth_indices, c14_expected_ages);
                expected_ages(N, delta_c, cs, theta, num_D18O_depths, D18O_depths, variables, D18O_depth_indices, D18O_expected_ages);

                logp_old = energy_function(N, delta_c, cs, a, b, theta, num_c14_depths, num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma,
                                           c14_depth_indices, inv_c14_var, c14_expected_ages, D18O, D18O_depths, D18O_sigma, D18O_depth_indices, inv_D18O_var, D18O_reference, D18O_reference_times, variables) +
                           bias_potential(CV_point, bias_centers, bias_heights, bias_widths, bias_count, kernel_weights, gamma, beta, sum_weights, Z, DeltaE);
                // Compute gradient at old state
                grad_energy_function(N, delta_c, cs, a, b, theta, num_c14_depths,
                                     num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma, c14_depth_indices, inv_c14_var, c14_expected_ages, D18O, D18O_depths, D18O_sigma,
                                     D18O_depth_indices, inv_D18O_var, D18O_expected_ages, D18O_reference, D18O_reference_times, variables, gradient);
                CV_point = get_CV_point(N, theta, variables, problem_index, delta_c);
                grad_bias(N, variables, CV_point, theta, problem_index, delta_c, bias_centers, bias_heights, bias_widths, bias_count, kernel_weights, gamma, beta, sum_weights, Z, DeltaE, bias_gradient);

                // Initial half step for momentum
                for (int n = 0; n < N; n++)
                {
                    momentum[n] -= (dt / 2) * (gradient[n] + bias_gradient[n]);
                }

                // Leapfrog integration
                for (int k = 0; k < num_dt; k++)
                {
                    for (int n = 0; n < N; n++)
                    {
                        log_variables[n] += dt * momentum[n];
                        variables[n] = exp(log_variables[n]);
                    }

                    expected_ages(N, delta_c, cs, theta, num_c14_depths, c14_depths, variables, c14_depth_indices, c14_expected_ages);
                    expected_ages(N, delta_c, cs, theta, num_D18O_depths, D18O_depths, variables, D18O_depth_indices, D18O_expected_ages);

                    grad_energy_function(N, delta_c, cs, a, b, theta, num_c14_depths,
                                         num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma, c14_depth_indices, inv_c14_var, c14_expected_ages, D18O, D18O_depths, D18O_sigma,
                                         D18O_depth_indices, inv_D18O_var, D18O_expected_ages, D18O_reference, D18O_reference_times, variables, gradient);
                    CV_point = get_CV_point(N, theta, variables, problem_index, delta_c);
                    grad_bias(N, variables, CV_point, theta, problem_index, delta_c, bias_centers, bias_heights, bias_widths, bias_count, kernel_weights, gamma, beta, sum_weights, Z, DeltaE, bias_gradient);

                    if (k != num_dt - 1)
                    {
                        for (int n = 0; n < N; n++)
                        {
                            momentum[n] -= dt * (gradient[n] + bias_gradient[n]);
                        }
                    }
                }
                for (int n = 0; n < N; n++)
                {
                    momentum[n] -= 0.5 * dt * (gradient[n] + bias_gradient[n]);
                }
                logp_new = energy_function(N, delta_c, cs, a, b, theta, num_c14_depths, num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma,
                                           c14_depth_indices, inv_c14_var, c14_expected_ages, D18O, D18O_depths, D18O_sigma, D18O_depth_indices, inv_D18O_var, D18O_reference, D18O_reference_times, variables) +
                           bias_potential(CV_point, bias_centers, bias_heights, bias_widths, bias_count, kernel_weights, gamma, beta, sum_weights, Z, DeltaE);

                kinetic_new = 0;
                kinetic_old = 0;
                for (int i = 0; i < N; i++)
                {
                    kinetic_new += 0.5 * momentum[i] * momentum[i];
                    kinetic_old += 0.5 * momentum_init[i] * momentum_init[i];
                }

                double dE = kinetic_new + logp_new - kinetic_old - logp_old;
                double acceptance_prob = fmin(1, exp(-dE));
                mean_acceptance += acceptance_prob;
                double random_number = gsl_rng_uniform(r);

                bool keep_variables = random_number >= acceptance_prob;
                for (int m = 0; m < N; m++)
                {
                    if (keep_variables)
                    {
                        variables[m] = variables_init[m];
                    }
                    log_variables[m] = log(variables[m]);
                }
                if (keep_variables)
                {
                    logp_new = logp_old;
                }
            }
        }
        mean_acceptance /= (num_samples * num_HMC);
        printf("%f\n", mean_acceptance);
    }
}
