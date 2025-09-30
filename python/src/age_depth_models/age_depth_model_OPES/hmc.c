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
#include "energy_functions.h"

#define min(a, b) (((a) <= (b)) ? (a) : (b))
#define max(a, b) (((a) >= (b)) ? (a) : (b))
#define MAX_BIAS 10000

void hmc(
    int N, int num_dt, int num_HMC, int num_chains, int num_samples, int num_lambda, int num_temps,
    int num_c14_depths, int num_D18O_depths, int num_D18O_reference_times, double H, double dt, double delta_c,
    double bias_sigma, double a, double b, double theta, double dE, double startbias, double endbias,
    double startbias_temp, double endbias_temp, double bias_distance_count, double gamma, double distance_threshold, double cap_energy_scale,
    double cap_width, const double *cs, const double *sp, const double *sp_energies, const double *c14_ages, const double *c14_depths,
    const double *c14_sigma, const double *D18O, const double *D18O_depths, const double *D18O_sigma,
    const double *D18O_reference, const double *D18O_reference_times, double *samples_out, double *energy_out, double *bias_out)
{

    // None of them seem to get a bias:(((, the bias is zero, so there is a bug.
    bool umbrella_bias = !(num_lambda == -1);
    bool temp_bias = !(num_temps == -1);
    bool unified = (umbrella_bias || temp_bias);
    bool rethinking = !isnan(distance_threshold);
    bool cap = !(isnan(cap_energy_scale));

    double problem_index;
    double bias_sigma_2;
    double *gaussian_centers = NULL;
    if (umbrella_bias || rethinking)
    {
        num_temps = temp_bias ? num_temps : 1;
        problem_index = 45;
        bias_sigma_2 = bias_sigma * bias_sigma;
        gaussian_centers = malloc(num_lambda * sizeof(double));
        for (int i = 0; i < num_lambda; i++)
        {
            gaussian_centers[i] = startbias + ((endbias - startbias) * ((double)i / (num_lambda - 1)));
        }
    }

    double *betas = NULL;
    double beta0;
    if (temp_bias)
    {
        num_lambda = umbrella_bias ? num_lambda : 1;
        double temp_center;
        double factor = 0;
        betas = malloc(num_temps * sizeof(double));
        for (int i = 0; i < num_temps; i++)
        {
            factor = (num_temps - 1 > 0) ? (double)i / (num_temps - 1) : 0.0;
            temp_center = startbias_temp + ((endbias_temp - startbias_temp) * factor);
            betas[i] = 1 / temp_center;
            beta0 += betas[i];
        }
        beta0 /= num_temps;
    }

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
        double Z = 1.0; // For rethinking
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
        double c14_expected_ages[num_c14_depths];
        double D18O_expected_ages[num_D18O_depths];
        double mean_acceptance = 0.0;
        double logp_new;
        double logp_old;
        double kinetic_new;
        double kinetic_old;
        double energy_old;
        double energy_new;

        // Bias variables
        double bias_gradient[N];
        double bias_old;
        double bias_new;
        double CV_point;
        double CV_point_init;

        // Unified variables
        double energy;
        int CV_point_index;
        int start_index;
        int end_index;
        double CV_point_plus_dist_sigma;
        double CV_point_minus_dist_sigma;
        double delta_F[num_lambda * num_temps];
        double delta_F_nominator_sum[num_lambda * num_temps];
        double delta_F_denominator_sum;

        // Rethinking variables
        double bias_centers[MAX_BIAS];
        double bias_widths[MAX_BIAS];
        double bias_heights[MAX_BIAS];
        int bias_count = 0;

        double *weights = malloc((num_samples + 1) * sizeof(double));
        double *kernel_weights = malloc((num_samples + 1) * sizeof(double));
        double sum_weights = 0;
        double sum_squared_weights = 0;

        double N_eff;
        double bias_std_j;

        // Cap variables
        double energy_old_orig;
        double energy_new_orig;
        double cap_energy_top;
        double cap_energy_bottom;
        double cap_energy;

        if (cap)
        {
            cap_energy_top = sp_energies[i] + (cap_width / 2);
            cap_energy_bottom = sp_energies[i] - (cap_width / 2);
        }

        for (int j = 0; j < N; j++)
        {
            variables[j] = sp[i * N + j];
            log_variables[j] = log(variables[j]);
        }

        if (umbrella_bias || rethinking)
        {
            CV_point = get_CV_point(N, theta, variables, problem_index, delta_c);
            if (umbrella_bias)
            {
                CV_point_plus_dist_sigma = CV_point + bias_distance_count * bias_sigma;
                CV_point_minus_dist_sigma = CV_point - bias_distance_count * bias_sigma;
                start_index = max(0, binary_search(gaussian_centers, num_lambda, CV_point_minus_dist_sigma));
                end_index = min(num_lambda, binary_search(gaussian_centers, num_lambda, CV_point_plus_dist_sigma));
                if (start_index == num_lambda)
                    start_index = num_lambda - 1;
                if (end_index == 0)
                    end_index = 1;
            }
        }

        expected_ages(N, delta_c, cs, theta, num_c14_depths, c14_depths,
                      variables, c14_depth_indices, c14_expected_ages);
        expected_ages(N, delta_c, cs, theta, num_D18O_depths, D18O_depths,
                      variables, D18O_depth_indices, D18O_expected_ages);
        if (temp_bias)
        {
            energy = energy_function(N, delta_c, cs, a, b, theta, num_c14_depths, num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma,
                                     c14_depth_indices, inv_c14_var, c14_expected_ages, D18O, D18O_depths, D18O_sigma, D18O_depth_indices, inv_D18O_var, D18O_expected_ages, D18O_reference, D18O_reference_times, variables);

            if (cap)
            {
                double cap_energy = energy > cap_energy_top ? cap_energy_top : cap_energy_bottom;
                if (energy > cap_energy_top || energy < cap_energy_bottom)
                {
                    energy = cap_energy + ((energy - cap_energy) * ((1 - cap_energy_scale) / (1 + ((energy - cap_energy) * (energy - cap_energy))) + cap_energy_scale));
                }
            }

            if (!umbrella_bias)
            {
                start_index = 0;
                end_index = 1;
            }
        }

        if (unified)
        {
            delta_F_denominator_sum = 1.0;
            double umbrella_factor;
            double temp_factor;
            for (int lambda_index = 0; lambda_index < num_lambda; lambda_index++)
            {
                umbrella_factor = umbrella_bias ? exp(-pow(CV_point - gaussian_centers[lambda_index], 2) / (2 * bias_sigma_2)) : 1;
                for (int temp_index = 0; temp_index < num_temps; temp_index++)
                {
                    temp_factor = temp_bias ? exp(-(betas[temp_index] - beta0) * energy) : 1;
                    delta_F_nominator_sum[lambda_index * num_temps + temp_index] = umbrella_factor * temp_factor;
                    delta_F[lambda_index * num_temps + temp_index] = -log(delta_F_nominator_sum[lambda_index * num_temps + temp_index] / delta_F_denominator_sum);
                    if (delta_F[lambda_index * num_temps + temp_index] >= dE)
                    {
                        delta_F[lambda_index * num_temps + temp_index] = dE;
                    }
                }
            }
        }

        if (rethinking)
        {
            weights[0] = exp(-dE);
            sum_weights += weights[0];
            sum_squared_weights += weights[0] * weights[0];
            N_eff = (sum_squared_weights > 0) ? sum_weights * sum_weights / sum_squared_weights : 1;
            bias_std_j = bias_sigma * pow(N_eff * (N + 2) / 4.0, -1.0 / (N + 4.0));
            deposit_gaussian(CV_point, bias_std_j, bias_centers, bias_heights, bias_widths, kernel_weights, weights[0], &sum_squared_weights, &bias_count, distance_threshold, MAX_BIAS);
            Z = compute_Zn(bias_centers, bias_heights, bias_widths, bias_count, kernel_weights, gamma, sum_weights);
        }

        for (int l = 0; l < num_samples; l++)
        {
            if (isnan(variables[0]))
            {
                printf("Variables are nan, aborting");
                exit(EXIT_FAILURE);
            }
            if (l % 100 == 0)
            {
                printf("Sample %d\n", l);
            }
            for (int j = 0; j < num_HMC; j++)
            {
                memcpy(variables_init, variables, N * sizeof(double));
                for (int m = 0; m < N; m++)
                {
                    momentum[m] = gsl_ran_gaussian(r, 1.0);
                }
                memcpy(momentum_init, momentum, N * sizeof(double));

                energy_old = energy_function(N, delta_c, cs, a, b, theta, num_c14_depths, num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma,
                                             c14_depth_indices, inv_c14_var, c14_expected_ages, D18O, D18O_depths, D18O_sigma, D18O_depth_indices, inv_D18O_var, D18O_expected_ages, D18O_reference, D18O_reference_times, variables);
                if (cap)
                {
                    energy_old_orig = energy_old;
                    cap_energy = energy_old > cap_energy_top ? cap_energy_top : cap_energy_bottom;
                    if (energy_old > cap_energy_top || energy_old < cap_energy_bottom)
                    {
                        energy_old = cap_energy + ((energy_old - cap_energy) * ((1 - cap_energy_scale) / (1 + ((energy_old - cap_energy) * (energy_old - cap_energy))) + cap_energy_scale));
                    }
                }

                CV_point_init = CV_point;
                if (unified)
                {
                    bias_old = bias_potential(CV_point, num_lambda, num_temps, gaussian_centers, betas, beta0, energy_old, bias_sigma, bias_sigma_2, delta_F, start_index, end_index, umbrella_bias, temp_bias);
                }
                else if (rethinking)
                {
                    bias_old = bias_potential_r(CV_point, bias_centers, bias_heights, bias_widths, bias_count, kernel_weights, gamma, sum_weights, Z, dE);
                }
                else
                    bias_old = 0;
                logp_old = energy_old + bias_old;

                // printf("logp %f, chain %d\n", logp_old, i);
                // Compute gradient at old state
                grad_energy_function(N, delta_c, cs, a, b, theta, num_c14_depths,
                                     num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma, c14_depth_indices, inv_c14_var, c14_expected_ages, D18O, D18O_depths, D18O_sigma,
                                     D18O_depth_indices, inv_D18O_var, D18O_expected_ages, D18O_reference, D18O_reference_times, variables, gradient);
                if (cap)
                {
                    double cap_energy = energy_old > cap_energy_top ? cap_energy_top : cap_energy_bottom;
                    if (energy_old > cap_energy_top || energy_old < cap_energy_bottom)
                    {
                        double diff_square = (energy_old - cap_energy) * (energy_old - cap_energy);
                        for (int i = 0; i < N; i++)
                        {
                            gradient[i] *= (((1 - cap_energy_scale) / (1 + diff_square)) + cap_energy_scale - (2 * (1 - cap_energy_scale) * diff_square / ((1 + diff_square) * (1 + diff_square))));
                        }
                    }
                }

                if (unified)
                {
                    grad_bias(N, delta_c, problem_index, CV_point, variables, num_lambda, num_temps, gaussian_centers, betas, beta0, energy_old, gradient, bias_sigma, bias_sigma_2, delta_F, start_index, end_index, umbrella_bias, temp_bias, bias_gradient);
                }

                if (rethinking)
                    grad_bias_r(N, delta_c, problem_index, CV_point, variables, bias_centers, bias_heights, bias_widths, bias_count, kernel_weights, gamma, sum_weights, Z, dE, gradient);
                // Initial half step for momentum
                for (int n = 0; n < N; n++)
                {
                    momentum[n] -= (unified || rethinking) ? (dt / 2) * (gradient[n] + bias_gradient[n]) : (dt / 2) * gradient[n];
                }

                // Leapfrog integration
                for (int k = 0; k < num_dt; k++)
                {
                    for (int n = 0; n < N; n++)
                    {
                        log_variables[n] += dt * momentum[n];
                        variables[n] = exp(log_variables[n]);
                    }

                    if (umbrella_bias || rethinking)
                    {
                        CV_point = get_CV_point(N, theta, variables, problem_index, delta_c);
                        if (umbrella_bias)
                        {
                            CV_point_plus_dist_sigma = CV_point + bias_distance_count * bias_sigma;
                            CV_point_minus_dist_sigma = CV_point - bias_distance_count * bias_sigma;
                            start_index = max(0, binary_search(gaussian_centers, num_lambda, CV_point_minus_dist_sigma));
                            end_index = min(num_lambda, binary_search(gaussian_centers, num_lambda, CV_point_plus_dist_sigma));
                            if (start_index == num_lambda)
                                start_index = num_lambda - 1;
                            if (end_index == 0)
                                end_index = 1;
                        }
                    }

                    expected_ages(N, delta_c, cs, theta, num_c14_depths, c14_depths, variables, c14_depth_indices, c14_expected_ages);
                    expected_ages(N, delta_c, cs, theta, num_D18O_depths, D18O_depths, variables, D18O_depth_indices, D18O_expected_ages);

                    if (temp_bias || cap)
                    {
                        energy = energy_function(N, delta_c, cs, a, b, theta, num_c14_depths, num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma,
                                                 c14_depth_indices, inv_c14_var, c14_expected_ages, D18O, D18O_depths, D18O_sigma, D18O_depth_indices, inv_D18O_var, D18O_expected_ages, D18O_reference, D18O_reference_times, variables);
                        if (cap)
                        {
                            cap_energy = energy > cap_energy_top ? cap_energy_top : cap_energy_bottom;
                            if (energy > cap_energy_top || energy < cap_energy_bottom)
                            {
                                energy = cap_energy + ((energy - cap_energy) * ((1 - cap_energy_scale) / (1 + ((energy - cap_energy) * (energy - cap_energy))) + cap_energy_scale));
                            }
                        }
                    }

                    grad_energy_function(N, delta_c, cs, a, b, theta, num_c14_depths,
                                         num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma, c14_depth_indices, inv_c14_var, c14_expected_ages, D18O, D18O_depths, D18O_sigma,
                                         D18O_depth_indices, inv_D18O_var, D18O_expected_ages, D18O_reference, D18O_reference_times, variables, gradient);

                    if (cap)
                    {
                        double cap_energy = energy > cap_energy_top ? cap_energy_top : cap_energy_bottom;
                        if (energy > cap_energy_top || energy < cap_energy_bottom)
                        {
                            double diff_square = (energy - cap_energy) * (energy - cap_energy);
                            for (int i = 0; i < N; i++)
                            {
                                gradient[i] *= (((1 - cap_energy_scale) / (1 + diff_square)) + cap_energy_scale - (2 * (1 - cap_energy_scale) * diff_square / ((1 + diff_square) * (1 + diff_square))));
                            }
                        }
                    }
                    if (unified)
                    {
                        grad_bias(N, delta_c, problem_index, CV_point, variables, num_lambda, num_temps, gaussian_centers, betas, beta0, energy, gradient, bias_sigma, bias_sigma_2, delta_F, start_index, end_index, umbrella_bias, temp_bias, bias_gradient);
                    }

                    if (rethinking)
                        grad_bias_r(N, delta_c, problem_index, CV_point, variables, bias_centers, bias_heights, bias_widths, bias_count, kernel_weights, gamma, sum_weights, Z, dE, gradient);

                    if (k != num_dt - 1)
                    {
                        for (int n = 0; n < N; n++)
                        {
                            momentum[n] -= (unified || rethinking) ? dt * (gradient[n] + bias_gradient[n]) : dt * gradient[n];
                        }
                    }
                }
                for (int n = 0; n < N; n++)
                {
                    momentum[n] -= (unified || rethinking) ? 0.5 * dt * (gradient[n] + bias_gradient[n]) : 0.5 * dt * gradient[n];
                }
                energy_new = energy_function(N, delta_c, cs, a, b, theta, num_c14_depths, num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma,
                                             c14_depth_indices, inv_c14_var, c14_expected_ages, D18O, D18O_depths, D18O_sigma, D18O_depth_indices, inv_D18O_var, D18O_expected_ages, D18O_reference, D18O_reference_times, variables);
                if (cap)
                {
                    energy_new_orig = energy_new;
                    cap_energy = energy_new > cap_energy_top ? cap_energy_top : cap_energy_bottom;
                    if (energy_new > cap_energy_top || energy_new < cap_energy_bottom)
                    {
                        energy_new = cap_energy + ((energy_new - cap_energy) * ((1 - cap_energy_scale) / (1 + ((energy_new - cap_energy) * (energy_new - cap_energy))) + cap_energy_scale));
                    }
                }
                if (unified)
                {
                    bias_new = bias_potential(CV_point, num_lambda, num_temps, gaussian_centers, betas, beta0, energy_new, bias_sigma, bias_sigma_2, delta_F, start_index, end_index, umbrella_bias, temp_bias);
                }

                else if (rethinking)
                {
                    bias_new = bias_potential_r(CV_point, bias_centers, bias_heights, bias_widths, bias_count, kernel_weights, gamma, sum_weights, Z, dE);
                    // printf("before %f\n", bias_new);
                }
                else
                    bias_new = 0;
                logp_new = energy_new + bias_new;

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
                    energy_new = energy_old;
                    expected_ages(N, delta_c, cs, theta, num_c14_depths, c14_depths, variables, c14_depth_indices, c14_expected_ages);
                    expected_ages(N, delta_c, cs, theta, num_D18O_depths, D18O_depths, variables, D18O_depth_indices, D18O_expected_ages);
                    if (unified || rethinking)
                    {
                        bias_new = bias_old;
                        if (umbrella_bias || rethinking)
                        {
                            CV_point = CV_point_init;
                            if (umbrella_bias)
                            {
                                CV_point_plus_dist_sigma = CV_point + bias_distance_count * bias_sigma;
                                CV_point_minus_dist_sigma = CV_point - bias_distance_count * bias_sigma;
                                start_index = max(0, binary_search(gaussian_centers, num_lambda, CV_point_minus_dist_sigma));
                                end_index = min(num_lambda, binary_search(gaussian_centers, num_lambda, CV_point_plus_dist_sigma));
                                if (start_index == num_lambda)
                                    start_index = num_lambda - 1;
                                if (end_index == 0)
                                    end_index = 1;
                            }
                        }
                    }
                    if (cap)
                        energy_new_orig = energy_old_orig;
                }
            }
            for (int m = 0; m < N; m++)
            {
                samples_out[i * num_samples * N + l * N + m] = variables[m];
            }
            energy_out[i * num_samples + l] = energy_new;

            if (unified || rethinking)
            {
                bias_out[i * num_samples + l] = (cap) ? bias_new + energy_new - energy_new_orig : bias_new;
            }
            else if (cap)
            {
                bias_out[i * num_samples + l] = energy_new - energy_new_orig;
            }

            if (unified)
            {
                delta_F_denominator_sum += exp(bias_new);
                update_delta_F(CV_point, num_lambda, num_temps, bias_sigma_2, dE, gaussian_centers, betas, beta0, energy_new, delta_F_nominator_sum, delta_F_denominator_sum, delta_F, bias_new, umbrella_bias, temp_bias);
            }

            if (rethinking)
            {
                weights[l + 1] = exp(bias_new);
                sum_weights += weights[l + 1];
                sum_squared_weights += weights[l + 1] * weights[l + 1];
                N_eff = (sum_squared_weights > 0) ? sum_weights * sum_weights / sum_squared_weights : 1;
                bias_std_j = bias_sigma * pow(N_eff * (N + 2) / 4.0, -1.0 / (N + 4.0));
                deposit_gaussian(CV_point, bias_std_j, bias_centers, bias_heights, bias_widths, kernel_weights, weights[l + 1], &sum_squared_weights, &bias_count, distance_threshold, MAX_BIAS);
                // printf("center %f\n", bias_centers[1]);
                Z = compute_Zn(bias_centers, bias_heights, bias_widths, bias_count, kernel_weights, gamma, sum_weights);
            }
        }

        mean_acceptance /= (num_samples * num_HMC);
        printf("%f\n", mean_acceptance);
    }
}
