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
#include <libgen.h>
#include <limits.h>
#include "energy_functions.h"

#define min(a, b) (((a) <= (b)) ? (a) : (b))
#define max(a, b) (((a) >= (b)) ? (a) : (b))
#define MAX_BIAS 10000
#define MAX_NBR_PC 2

void hmc(
    int N, int num_dt, int num_HMC, int num_chains, int num_samples, int num_lambda, int num_temps, int num_pcs,
    int num_c14_depths, int num_D18O_depths, int num_D18O_reference_times, int seed, double H, double dt, double delta_c,
    double bias_sigma, double a, double b, double theta, double beta, double dE, double startbias, double endbias,
    double startbias_temp, double endbias_temp, double bias_distance_count, double cap_energy_scale, double energy_exp, int dfs, double *c_delta_F,
    double cap_width, double *betas, const double *cs, const double *pcs, const double *sp, const double *sp_mean, const double *sp_energies, const double *c14_ages, const double *c14_depths,
    const double *c14_sigma, const double *D18O, const double *D18O_depths, const double *D18O_sigma,
    const double *D18O_reference, const double *D18O_reference_times, double *samples_out, double *energy_out, double *bias_out, const char *config_str, const char *config_find_min_str, const char *data_name, bool shared_bias)
{
    bool umbrella_bias = !(num_lambda == -1);
    bool temp_bias = !(num_temps == -1);
    bool OPES = (umbrella_bias || temp_bias);
    bool cap = !(isnan(cap_energy_scale));
    int num_lambda_2;
    bool cdf = (c_delta_F == NULL) ? false : true;

    double bias_sigma_2;
    double *gaussian_centers = NULL;
    if (umbrella_bias)
    {
        num_temps = temp_bias ? num_temps : 1;
        bias_sigma_2 = bias_sigma * bias_sigma;
        gaussian_centers = malloc(num_lambda * sizeof(double));
        for (int i = 0; i < num_lambda; i++)
        {
            gaussian_centers[i] = startbias + ((endbias - startbias) * ((double)i / (num_lambda - 1)));
        }
    }

    double beta0 = 1;
    if (temp_bias)
    {
        num_lambda = umbrella_bias ? num_lambda : 1;
        num_pcs = umbrella_bias ? num_pcs : 1;
    }

    if (!OPES)
    {
        num_lambda = 1;
        num_pcs = 1;
        num_temps = 1;
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

    double *delta_F_denominator_sum = malloc(sizeof(double));
    double *max_delta_F_denominator_sum_term = malloc(sizeof(double));
    int total = (int)pow(num_lambda, num_pcs) * num_temps;
    double max_delta_F_nominator_sum_term[total];
    double delta_F_nominator_sum[total];
    double delta_F[total];
    FILE *deltaF_out;

    if (OPES && shared_bias)
    {
        *delta_F_denominator_sum = 0;
        *max_delta_F_denominator_sum_term = 0;

        for (int i = 0; i < num_lambda; i++)
        {
            for (int j = 0; j < num_temps; j++)
            {
                max_delta_F_nominator_sum_term[i * num_temps + j] = 0;
                delta_F_nominator_sum[i * num_temps + j] = 0;
                delta_F[i * num_temps + j] = cdf ? c_delta_F[i * num_temps + j] : (betas[j] - beta0) * energy_exp;
            }
        }
    }

    omp_lock_t deltaF_lock;
    omp_init_lock(&deltaF_lock);

    if (OPES && shared_bias)
    {
        char fname[PATH_MAX];
        char resolved_path[PATH_MAX];

        // Get the directory of this source file at compile time
        char *src_dir = strdup(__FILE__); // duplicate __FILE__ string
        char *dir = dirname(src_dir);     // get the directory part

        // Build the relative path
        snprintf(fname, sizeof(fname), "%s/../../../../output/%s/%s/%s/deltaF.bin",
                 dir, data_name, config_find_min_str, config_str);

        // --- Open file for writing ---
        remove(fname);
        deltaF_out = fopen(fname, "wb");
        if (!deltaF_out)
        {
            fprintf(stderr, "Error: could not open %s for writing\n", fname);
            exit(EXIT_FAILURE);
        }
    }

#pragma omp parallel num_threads(num_chains)
    {
        int i = omp_get_thread_num();
        const gsl_rng_type *T;
        gsl_rng *r;

        gsl_rng_env_setup();
        T = gsl_rng_default;
        r = gsl_rng_alloc(T);
        gsl_rng_set(r, seed + i);

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
        double CV_point[num_pcs];
        double CV_point_init[num_pcs];

        // Unified variables
        double energy;
        int start_index[MAX_NBR_PC];
        int end_index[MAX_NBR_PC];
        double CV_point_plus_dist_sigma;
        double CV_point_minus_dist_sigma;

        double *delta_F_denominator_sum_local = malloc(sizeof(double));
        double *max_delta_F_denominator_sum_term_local = malloc(sizeof(double));
        double *max_delta_F_nominator_sum_term_local;
        double *delta_F_nominator_sum_local;
        double *delta_F_local;
        FILE *deltaF_out_local;

        if (OPES && shared_bias)
        {
            delta_F_denominator_sum_local = delta_F_denominator_sum;
            max_delta_F_denominator_sum_term_local = max_delta_F_denominator_sum_term;
            delta_F_local = delta_F;
            delta_F_nominator_sum_local = delta_F_nominator_sum;
            max_delta_F_nominator_sum_term_local = max_delta_F_nominator_sum_term;
            deltaF_out_local = deltaF_out;
        }
        else if (OPES)
        {
            delta_F_local = malloc(total * sizeof(double));
            delta_F_nominator_sum_local = malloc(total * sizeof(double));
            max_delta_F_nominator_sum_term_local = malloc(total * sizeof(double));
            *delta_F_denominator_sum_local = 0;
            *max_delta_F_denominator_sum_term_local = 0;
            for (int i = 0; i < total; i++)
            {
                max_delta_F_nominator_sum_term_local[i] = 0;
                delta_F_nominator_sum_local[i] = 0;
                delta_F_local[i] = cdf ? c_delta_F[i] : (betas[i] - beta0) * energy_exp;
            }
        }

        if (OPES)
        {
            if (!shared_bias)
            {
                char fname[PATH_MAX];
                char resolved_path[PATH_MAX];

                // Get the directory of this source file at compile time
                char *src_dir = strdup(__FILE__); // duplicate __FILE__ string
                char *dir = dirname(src_dir);     // get the directory part

                // Build the relative path
                snprintf(fname, sizeof(fname), "%s/../../../../output/%s/%s/%s/deltaF_chain%d.bin",
                         dir, data_name, config_find_min_str, config_str, i);

                // --- Open file for writing ---
                remove(fname);
                deltaF_out_local = fopen(fname, "wb");
                if (!deltaF_out_local)
                {
                    fprintf(stderr, "Error: could not open %s for writing\n", fname);
                    exit(EXIT_FAILURE);
                }
            }
            num_lambda_2 = (num_pcs == 2) ? num_lambda : 1;
        }

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

        if (umbrella_bias)
        {
            for (int j = 0; j < num_pcs; j++)
            {
                CV_point[j] = get_CV_point(N, variables, pcs + j * N, sp_mean);
                CV_point_plus_dist_sigma = CV_point[j] + bias_distance_count * bias_sigma;
                CV_point_minus_dist_sigma = CV_point[j] - bias_distance_count * bias_sigma;
                start_index[j] = max(0, binary_search(gaussian_centers, num_lambda, CV_point_minus_dist_sigma));
                end_index[j] = min(num_lambda, binary_search(gaussian_centers, num_lambda, CV_point_plus_dist_sigma));
                if (start_index[j] == num_lambda)
                    start_index[j] = num_lambda - 1;
                if (end_index[j] == 0)
                    end_index[j] = 1;
            }
            for (int j = num_pcs; j < MAX_NBR_PC; j++)
            {
                start_index[j] = 0;
                end_index[j] = 1;
            }
        }

        expected_ages(N, delta_c, cs, theta, num_c14_depths, c14_depths,
                      variables, c14_depth_indices, c14_expected_ages);
        expected_ages(N, delta_c, cs, theta, num_D18O_depths, D18O_depths,
                      variables, D18O_depth_indices, D18O_expected_ages);
        if (temp_bias)
        {
            energy = energy_function(N, delta_c, cs, a, b, theta, beta, num_c14_depths, num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma,
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
                for (int j = 0; j < MAX_NBR_PC; j++)
                {
                    start_index[j] = 0;
                    end_index[j] = 1;
                }
            }
        }

        //         if (OPES)
        //         {
        //             for (int m = 0; m < num_chains; m++) // Ugly but seems to yield deterministic results.
        //             {
        // #pragma omp barrier
        //                 if (i == m && ((shared_bias && i == 0) || !shared_bias))
        //                 {
        //                     update_delta_F(num_pcs, CV_point, num_lambda, num_temps, bias_sigma_2, dE, gaussian_centers, betas, beta0, energy, delta_F_nominator_sum_local, delta_F_denominator_sum_local, max_delta_F_nominator_sum_term_local, max_delta_F_denominator_sum_term_local, delta_F_local, 0.0, umbrella_bias, temp_bias);
        //                 }

        // #pragma omp barrier
        //                 if (i == m && shared_bias && i != 0)
        //                 {
        //                     omp_set_lock(&deltaF_lock);
        //                     delta_F_denominator_sum_local += 1;
        //                     printf("dF %f\n", delta_F[1]);
        //                     update_delta_F(num_pcs, CV_point, num_lambda, num_temps, bias_sigma_2, dE, gaussian_centers, betas, beta0, energy, delta_F_nominator_sum_local, delta_F_denominator_sum_local, max_delta_F_nominator_sum_term_local, max_delta_F_denominator_sum_term_local, delta_F_local, 0.0, umbrella_bias, temp_bias);
        //                     omp_unset_lock(&deltaF_lock);
        //                 }
        // #pragma omp barrier
        //             }
        //         } // maybe destroyed something here as well?

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

                energy_old = energy_function(N, delta_c, cs, a, b, theta, beta, num_c14_depths, num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma,
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

                if (OPES)
                {
                    if (!temp_bias)
                        memcpy(CV_point_init, CV_point, num_pcs * sizeof(double));
                    bias_old = bias_potential(num_pcs, CV_point, num_lambda, num_temps, gaussian_centers, betas, beta0, energy_old, bias_sigma, bias_sigma_2, delta_F_local, start_index, end_index, umbrella_bias, temp_bias);
                }
                else
                    bias_old = 0;
                logp_old = energy_old + bias_old;
                // printf("logp %f, chain %d\n", logp_old, i);
                // Compute gradient at old state
                grad_energy_function(N, delta_c, cs, a, b, theta, beta, num_c14_depths,
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

                if (OPES)
                {
                    grad_bias(N, delta_c, num_pcs, pcs, CV_point, variables, num_lambda, num_temps, gaussian_centers, betas, beta0, energy_old, gradient, bias_sigma, bias_sigma_2, delta_F_local, start_index, end_index, umbrella_bias, temp_bias, bias_gradient);
                }

                for (int n = 0; n < N; n++)
                {
                    momentum[n] -= (OPES) ? (dt / 2) * (gradient[n] + bias_gradient[n]) : (dt / 2) * gradient[n];
                }

                // Leapfrog integration
                for (int k = 0; k < num_dt; k++)
                {
                    for (int n = 0; n < N; n++)
                    {
                        log_variables[n] += dt * momentum[n];
                        variables[n] = exp(log_variables[n]);
                    }

                    if (umbrella_bias)
                    {
                        for (int l = 0; l < num_pcs; l++)
                        {
                            CV_point[l] = get_CV_point(N, variables, pcs + l * N, sp_mean);
                            if (umbrella_bias)
                            {
                                CV_point_plus_dist_sigma = CV_point[l] + bias_distance_count * bias_sigma;
                                CV_point_minus_dist_sigma = CV_point[l] - bias_distance_count * bias_sigma;
                                start_index[l] = max(0, binary_search(gaussian_centers, num_lambda, CV_point_minus_dist_sigma));
                                end_index[l] = min(num_lambda, binary_search(gaussian_centers, num_lambda, CV_point_plus_dist_sigma));
                                if (start_index[l] == num_lambda)
                                    start_index[l] = num_lambda - 1;
                                if (end_index[l] == 0)
                                    end_index[l] = 1;
                            }
                        }
                    }

                    expected_ages(N, delta_c, cs, theta, num_c14_depths, c14_depths, variables, c14_depth_indices, c14_expected_ages);
                    expected_ages(N, delta_c, cs, theta, num_D18O_depths, D18O_depths, variables, D18O_depth_indices, D18O_expected_ages);

                    if (temp_bias || cap)
                    {
                        energy = energy_function(N, delta_c, cs, a, b, theta, beta, num_c14_depths, num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma,
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

                    grad_energy_function(N, delta_c, cs, a, b, theta, beta, num_c14_depths,
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
                    if (OPES)
                    {
                        grad_bias(N, delta_c, num_pcs, pcs, CV_point, variables, num_lambda, num_temps, gaussian_centers, betas, beta0, energy, gradient, bias_sigma, bias_sigma_2, delta_F_local, start_index, end_index, umbrella_bias, temp_bias, bias_gradient);
                    }

                    if (k != num_dt - 1)
                    {
                        for (int n = 0; n < N; n++)
                        {
                            momentum[n] -= (OPES) ? dt * (gradient[n] + bias_gradient[n]) : dt * gradient[n];
                        }
                    }
                }
                for (int n = 0; n < N; n++)
                {
                    momentum[n] -= (OPES) ? 0.5 * dt * (gradient[n] + bias_gradient[n]) : 0.5 * dt * gradient[n];
                }
                energy_new = energy_function(N, delta_c, cs, a, b, theta, beta, num_c14_depths, num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma,
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
                if (OPES)
                {
                    bias_new = bias_potential(num_pcs, CV_point, num_lambda, num_temps, gaussian_centers, betas, beta0, energy_new, bias_sigma, bias_sigma_2, delta_F_local, start_index, end_index, umbrella_bias, temp_bias);
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

                double hmc_dE = kinetic_new + logp_new - kinetic_old - logp_old;
                double acceptance_prob = fmin(1, exp(-hmc_dE));
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
                    if (OPES)
                    {
                        bias_new = bias_old;
                        if (umbrella_bias)
                        {
                            for (int l = 0; l < num_pcs; l++)
                            {
                                CV_point[l] = CV_point_init[l];
                                if (umbrella_bias)
                                {
                                    CV_point_plus_dist_sigma = CV_point[l] + bias_distance_count * bias_sigma;
                                    CV_point_minus_dist_sigma = CV_point[l] - bias_distance_count * bias_sigma;
                                    start_index[l] = max(0, binary_search(gaussian_centers, num_lambda, CV_point_minus_dist_sigma));
                                    end_index[l] = min(num_lambda, binary_search(gaussian_centers, num_lambda, CV_point_plus_dist_sigma));
                                    if (start_index[l] == num_lambda)
                                        start_index[l] = num_lambda - 1;
                                    if (end_index[l] == 0)
                                        end_index[l] = 1;
                                }
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

            if (OPES)
            {
                bias_out[i * num_samples + l] = (cap) ? bias_new + energy_new - energy_new_orig : bias_new;
            }
            else if (cap)
            {
                bias_out[i * num_samples + l] = energy_new - energy_new_orig;
            }

            if (OPES)
            {
                for (int m = 0; m < num_chains; m++)
                {
#pragma omp barrier
                    if (m == i)
                    {
                        omp_set_lock(&deltaF_lock);
                        if (bias_new > *max_delta_F_denominator_sum_term_local)
                        {
                            *delta_F_denominator_sum_local *= exp(*max_delta_F_denominator_sum_term_local - bias_new);
                            *max_delta_F_denominator_sum_term_local = bias_new;
                        }
                        *delta_F_denominator_sum_local += exp(bias_new - *max_delta_F_denominator_sum_term_local);
                        update_delta_F(num_pcs, CV_point, num_lambda, num_temps, bias_sigma_2, dE, gaussian_centers, betas, beta0, energy_new, delta_F_nominator_sum_local, delta_F_denominator_sum_local, max_delta_F_nominator_sum_term_local, max_delta_F_denominator_sum_term_local, delta_F_local, bias_new, umbrella_bias, temp_bias, l, dfs, c_delta_F, cdf);
                        if (l % 100 == 0 && l > 0 && false)
                        {
                            for (int i = 0; i < total; i++)
                            {
                                delta_F_local[i] = *max_delta_F_denominator_sum_term_local + log(*delta_F_denominator_sum_local) - max_delta_F_nominator_sum_term_local[i] - log(delta_F_nominator_sum_local[i]);
                                // delta_F_local[i] = -log(delta_F_nominator_sum_local[i] / delta_F_denominator_sum_local); //-log(delta_F_nominator[i] / len_energies_df);
                                delta_F_nominator_sum_local[i] = 0;
                                max_delta_F_nominator_sum_term_local[i] = 0;
                            }
                            *delta_F_denominator_sum_local = 0;
                            *max_delta_F_denominator_sum_term_local = 0;
                        }
                        omp_unset_lock(&deltaF_lock);
                    }
#pragma omp barrier
                }
                if (l % 1 == 0 && (!shared_bias || (shared_bias && i == 0)))
                {
                    size_t written = fwrite(delta_F_local, sizeof(double), num_lambda * num_lambda_2 * num_temps, deltaF_out_local);
                    if (written != (size_t)(num_lambda * num_lambda_2 * num_temps))
                    {
                        fprintf(stderr, "Error writing data");
                    }
                }
#pragma omp barrier
            }
        }

        mean_acceptance /= (num_samples * num_HMC);
        printf("%f\n", mean_acceptance);
        if (OPES && !shared_bias)
        {
            fclose(deltaF_out_local);
        }
    }
    if (OPES && shared_bias)
    {
        fclose(deltaF_out);
    }
    omp_destroy_lock(&deltaF_lock);
}
