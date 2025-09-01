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

void interpolate_D18O(
    int num_D18O_depths, int num_D18O_reference_times,
    const double *D18O_times, const double *D18O_reference, const double *D18O_reference_times,
    double D18O_reference_interp[num_D18O_depths], double D18O_interp_derivative[num_D18O_depths], const double *D18O_depths)
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
            double f_l = (l < j) ? -delta_c : (l == j) ? -(c14_depths[i] - cs[j])
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

double bias_potential(double CV_point, int num_lambda, double gaussian_centers[num_lambda], double sigma, double sigma_2, double delta_F[num_lambda])
{
    double sum_for_V = 0;

    for (int i = 0; i < num_lambda; i++)
    {
        // if (fabs(CV_point - gaussian_centers[i]) > 3 * sigma)
        // {
        //     continue;
        // }
        sum_for_V += exp(-(pow(CV_point - gaussian_centers[i], 2) / (2 * sigma_2)) + delta_F[i]);
    }
    double V = -log(sum_for_V / num_lambda);
    return V;
}

void grad_bias(int N, double delta_c, int problem_index, double CV_point, double variables[N], int num_lambda, double gaussian_centers[num_lambda], double sigma, double sigma_2, double delta_F[num_lambda], double bias_gradient[N])
{
    // Derivative of the bias with respect to log(sedimentation_rates)
    double sum_for_dV = 0;
    double sum_for_V = 0;

    for (int i = 0; i < num_lambda; i++)
    {
        // if (fabs(CV_point - gaussian_centers[i]) > 3 * sigma)
        // {
        //     continue;
        // }
        double gaussian_diff = (CV_point - gaussian_centers[i]);
        double gaussian_diff_2 = gaussian_diff * gaussian_diff;
        sum_for_dV += (gaussian_diff / sigma_2) * exp(-(gaussian_diff_2 / (2 * sigma_2)) + delta_F[i]);
        sum_for_V += exp(-(gaussian_diff_2 / (2 * sigma_2)) + delta_F[i]);
    }
    if (sum_for_V == 0)
    {
        fprintf(stderr, "WARNING: sum_for_V = %f\n. Try increasing sigma.", sum_for_V);
    }

    for (int i = 0; i < N; i++)
    {
        bias_gradient[i] = (i < problem_index) ? -variables[i] * delta_c * sum_for_dV / sum_for_V : 0;
    }
}

static inline double logaddexp(double a, double b)
{
    if (a == -INFINITY)
        return b;
    if (b == -INFINITY)
        return a;

    double m = (a > b) ? a : b;
    return m + log(exp(a - m) + exp(b - m));
}

void update_delta_F(double CV_point, int lambda_index, int num_lambda, double sigma_2, double gaussian_centers[num_lambda], double delta_F_nominator_sum[num_lambda], double delta_F_denominator_sum, double delta_F[num_lambda], double potential)
{
    double log_term = (-pow(CV_point - gaussian_centers[lambda_index], 2) / (2 * sigma_2)) + potential;
    delta_F_nominator_sum[lambda_index] = logaddexp(delta_F_nominator_sum[lambda_index], log_term);

    delta_F[lambda_index] = -(delta_F_nominator_sum[lambda_index] - delta_F_denominator_sum);
}

void *hmc(
    int N, double H, double delta_c, const double *cs,
    double dt, int num_dt, int num_HMC, int num_chains, int num_samples, int num_lambda, int problem_index, double bias_sigma,
    double a, double b, double theta, int num_c14_depths, int num_D18O_depths, int num_D18O_reference_times, const double *c14_ages,
    const double *c14_depths, const double *c14_sigma, const double *D18O,
    const double *D18O_depths, const double *D18O_sigma, const double *D18O_reference, const double *D18O_reference_times,
    double *samples_out, double *energy_out, double *bias_out)
{

    double bias_sigma_2 = bias_sigma * bias_sigma;
    double gaussian_centers[num_lambda];
    for (int i = 0; i < num_lambda; i++)
    {
        gaussian_centers[i] = 1300 + 200 * ((double)i / (num_lambda - 1));
    }

    int c14_depth_indices[num_c14_depths];
    double inv_c14_var[num_c14_depths];
    for (int i = 0; i < num_c14_depths; i++)
    {
        c14_depth_indices[i] = get_index(N, cs, c14_depths, i);
        inv_c14_var[i] = 1.0 / (c14_sigma[i] * c14_sigma[i]);
    }

    int D18O_depth_indices[num_D18O_depths];
    double inv_D18O_var[num_D18O_depths];
    for (int i = 0; i < num_D18O_depths; i++)
    {
        D18O_depth_indices[i] = get_index(N, cs, D18O_depths, i);
        inv_D18O_var[i] = 1.0 / (D18O_sigma[i] * D18O_sigma[i]);
    }

#pragma omp parallel for
    for (int i = 0; i < num_chains; i++)
    {
        const gsl_rng_type *T;
        gsl_rng *r;

        gsl_rng_env_setup();
        T = gsl_rng_default;
        r = gsl_rng_alloc(T);
        gsl_rng_set(r, 41 + i);

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

        for (int j = 0; j < N; j++)
        {
            variables[j] = gsl_ran_gamma(r, a, 1 / b);
            log_variables[j] = log(variables[j]);
        }

        CV_point = get_CV_point(N, theta, variables, problem_index, delta_c);
        double delta_F[num_lambda], delta_F_nominator_sum[num_lambda], delta_F_denominator_sum;
        delta_F_denominator_sum = 1.0;
        for (int i = 0; i < num_lambda; i++)
        {
            delta_F_nominator_sum[i] = exp(-pow(CV_point - gaussian_centers[i], 2) / (2 * bias_sigma_2));
            delta_F[i] = -log(delta_F_nominator_sum[i] / delta_F_denominator_sum);
            if (delta_F[i] >= 15)
            {
                delta_F[i] = 15;
            }
        }

        for (int l = 0; l < num_samples; l++)
        {
            if (l % 10000 == 0)
            {
                printf("%d\n", l / 10000);
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

                // Compute gradient at old state
                grad_energy_function(N, delta_c, cs, a, b, theta, num_c14_depths,
                                     num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma, c14_depth_indices, inv_c14_var, c14_expected_ages, D18O, D18O_depths, D18O_sigma,
                                     D18O_depth_indices, inv_D18O_var, D18O_expected_ages, D18O_reference, D18O_reference_times, variables, gradient);
                CV_point = get_CV_point(N, theta, variables, problem_index, delta_c);
                grad_bias(N, delta_c, problem_index, CV_point, variables, num_lambda, gaussian_centers, bias_sigma, bias_sigma_2, delta_F, bias_gradient);

                logp_old = energy_function(N, delta_c, cs, a, b, theta, num_c14_depths, num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma,
                                           c14_depth_indices, inv_c14_var, c14_expected_ages, D18O, D18O_depths, D18O_sigma, D18O_depth_indices, inv_D18O_var, D18O_reference, D18O_reference_times, variables) +
                           bias_potential(CV_point, num_lambda, gaussian_centers, bias_sigma, bias_sigma_2, delta_F);

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
                    grad_bias(N, delta_c, problem_index, CV_point, variables, num_lambda, gaussian_centers, bias_sigma, bias_sigma_2, delta_F, bias_gradient);

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
                           bias_potential(CV_point, num_lambda, gaussian_centers, bias_sigma, bias_sigma_2, delta_F);

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

                bool keep_variables = random_number > acceptance_prob;
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
            for (int m = 0; m < N; m++)
            {
                samples_out[i * num_samples * N + l * N + m] = variables[m];
            }
            energy_out[i * num_samples + l] = logp_new;
            double bias_value = bias_potential(CV_point, num_lambda, gaussian_centers, bias_sigma, bias_sigma_2, delta_F);
            bias_out[i * num_samples + l] = bias_value;

            delta_F_denominator_sum = logaddexp(delta_F_denominator_sum, bias_value);
            for (int i = 0; i < num_lambda; i++)
            {
                update_delta_F(CV_point, i, num_lambda, bias_sigma_2, gaussian_centers, delta_F_nominator_sum, delta_F_denominator_sum, delta_F, bias_value);
            }
        }

        mean_acceptance /= (num_samples * num_HMC);
        printf("%f\n", mean_acceptance);
    }
}
