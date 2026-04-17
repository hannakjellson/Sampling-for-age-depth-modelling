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
#include "configs.h"

void opes(
    OPESConfig *oc, Data *d, double *samples_out, double *energy_out, double *d18o_energy_out, double *bias_out, double *df_out)
{
    int i;
    int c14_depth_indices[d->nc14];

    for (i = 0; i < d->nc14; i++)
    {
        c14_depth_indices[i] = binary_search(d->cs, d->N + 1, d->c14d[i]) - 1;
    }

    int D18O_depth_indices[d->nd18o];
    for (i = 0; i < d->nd18o; i++)
    {
        D18O_depth_indices[i] = binary_search(d->cs, d->N + 1, d->d18od[i]) - 1;
    }

    double *delta_F_denominator_sum = malloc(sizeof(double));
    double delta_F_nominator_sum[oc->nt];
    double *max_dfd = malloc(sizeof(double));
    double max_dfn[oc->nt];
    double delta_F[oc->nt];
    int stop_throw_idx = INT_MAX;

    omp_lock_t deltaF_lock;
    omp_init_lock(&deltaF_lock);

    if (oc->sb)
    {
        *delta_F_denominator_sum = oc->dfd;
        *max_dfd = 0;
        for (i = 0; i < oc->nt; i++)
        {
            max_dfn[i] = log(oc->dfn[i] / oc->dfd);
            delta_F_nominator_sum[i] = oc->dfd;
            delta_F[i] = oc->df[i];
        }
    }

#pragma omp parallel num_threads(oc->hmcc->nch)
    {
        int i = omp_get_thread_num();
        int j, k, l, m;
        const gsl_rng_type *T;
        gsl_rng *r;

        gsl_rng_env_setup();
        T = gsl_rng_default;
        r = gsl_rng_alloc(T);
        gsl_rng_set(r, oc->hmcc->sd + i);

        double momentum[d->N];
        double momentum_init[d->N];

        double variables[d->N];
        double sed_rates[d->N];
        double variables_init[d->N];
        double gradient[d->N];
        double c14_expected_ages[d->nc14];
        double D18O_expected_ages[d->nd18o];
        double logp_new;
        double logp_old;
        double kinetic_new;
        double kinetic_old;
        double energy_old;
        double energy_new;

        // Bias variables
        double bias_gradient[d->N];
        double d18o_gradient[d->N];
        double d18o_energy;
        double d18o_energy_old;
        double bias_old;
        double bias_new;
        double *delta_F_denominator_sum_local;
        double *delta_F_nominator_sum_local;
        double *max_dfd_local;
        double *max_dfn_local;
        double *delta_F_local;

        // HMC variables
        double hmc_dE;
        double acceptance_prob;
        double mean_acceptance = 0.0;
        double random_number;
        bool keep_variables;

        if (oc->sb)
        {
            delta_F_denominator_sum_local = delta_F_denominator_sum;
            delta_F_nominator_sum_local = delta_F_nominator_sum;
            delta_F_local = delta_F;
            max_dfd_local = max_dfd;
            max_dfn_local = max_dfn;
        }
        else
        {
            delta_F_denominator_sum_local = malloc(sizeof(double));
            max_dfd_local = malloc(sizeof(double));
            delta_F_nominator_sum_local = malloc(oc->nt * sizeof(double));
            max_dfn_local = malloc(oc->nt * sizeof(double));
            delta_F_local = malloc(oc->nt * sizeof(double));

            *delta_F_denominator_sum_local = oc->dfd;
            *max_dfd_local = 0;
            for (j = 0; j < oc->nt; j++)
            {
                max_dfn_local[j] = log(oc->dfn[j] / oc->dfd);
                delta_F_nominator_sum_local[j] = oc->dfd;
                delta_F_local[j] = oc->df[j];
            }
        }

        for (j = 0; j < d->N; j++)
        {
            sed_rates[j] = oc->hmcc->sp[i * d->N + j];
            variables[j] = (log(sed_rates[j]) - d->pm) / d->ps;
        }

        expected_ages(d->N, d->dc, d->cs, d->th, d->nc14, d->c14d,
                      sed_rates, c14_depth_indices, c14_expected_ages);
        expected_ages(d->N, d->dc, d->cs, d->th, d->nd18o, d->d18od,
                      sed_rates, D18O_depth_indices, D18O_expected_ages);

        for (j = 0; j < oc->hmcc->ns; j++)
        {
            if (isnan(variables[0]))
            {
                printf("Variables are nan, aborting");
                exit(EXIT_FAILURE);
            }
            if (j % 100 == 0 && i == 0)
            {
                printf("Sample %d\n", j);
            }
            for (k = 0; k < oc->nhmc; k++)
            {
                memcpy(variables_init, variables, d->N * sizeof(double));
                for (l = 0; l < d->N; l++)
                {
                    momentum[l] = gsl_ran_gaussian(r, 1.0);
                }
                memcpy(momentum_init, momentum, d->N * sizeof(double));

                energy_old = energy_function(d, c14_expected_ages, D18O_expected_ages, variables, &d18o_energy);
                d18o_energy_old = d18o_energy;
                bias_old = bias_potential(oc->nt, oc->bs, d18o_energy, delta_F_local);
                logp_old = energy_old + bias_old;

                // Compute gradient at old state
                grad_energy_function(d, c14_depth_indices, c14_expected_ages, D18O_depth_indices, D18O_expected_ages, variables, gradient, d18o_gradient);
                grad_bias(d->N, oc->nt, oc->bs, variables, d18o_energy, d18o_gradient, delta_F_local, bias_gradient);
                for (l = 0; l < d->N; l++)
                {
                    momentum[l] -= (oc->hmcc->dt / 2) * (gradient[l] + bias_gradient[l]);
                }

                // Leapfrog integration
                for (l = 0; l < oc->hmcc->ndt; l++)
                {
                    for (m = 0; m < d->N; m++)
                    {
                        variables[m] += oc->hmcc->dt * momentum[m];
                        sed_rates[m] = exp(variables[m] * d->ps + d->pm);
                    }

                    expected_ages(d->N, d->dc, d->cs, d->th, d->nc14, d->c14d, sed_rates, c14_depth_indices, c14_expected_ages);
                    expected_ages(d->N, d->dc, d->cs, d->th, d->nd18o, d->d18od, sed_rates, D18O_depth_indices, D18O_expected_ages);

                    grad_energy_function(d, c14_depth_indices, c14_expected_ages, D18O_depth_indices, D18O_expected_ages, variables, gradient, d18o_gradient);
                    grad_bias(d->N, oc->nt, oc->bs, variables, d18o_energy, d18o_gradient, delta_F_local, bias_gradient);

                    if (l != oc->hmcc->ndt - 1)
                    {
                        for (m = 0; m < d->N; m++)
                        {
                            momentum[m] -= oc->hmcc->dt * (gradient[m] + bias_gradient[m]);
                        }
                    }
                }
                for (l = 0; l < d->N; l++)
                {
                    momentum[l] -= 0.5 * oc->hmcc->dt * (gradient[l] + bias_gradient[l]);
                }

                energy_new = energy_function(d, c14_expected_ages, D18O_expected_ages, variables, &d18o_energy);
                bias_new = bias_potential(oc->nt, oc->bs, d18o_energy, delta_F_local);
                logp_new = energy_new + bias_new;

                kinetic_new = 0;
                kinetic_old = 0;

                for (l = 0; l < d->N; l++)
                {
                    kinetic_new += 0.5 * momentum[l] * momentum[l];
                    kinetic_old += 0.5 * momentum_init[l] * momentum_init[l];
                }

                hmc_dE = kinetic_new + logp_new - kinetic_old - logp_old;
                acceptance_prob = fmin(1, exp(-hmc_dE));
                mean_acceptance += acceptance_prob;
                random_number = gsl_rng_uniform(r);

                keep_variables = random_number > acceptance_prob;
                for (l = 0; l < d->N; l++)
                {
                    if (keep_variables)
                    {
                        variables[l] = variables_init[l];
                    }
                    sed_rates[l] = exp(variables[l] * d->ps + d->pm);
                }
                if (keep_variables)
                {
                    energy_new = energy_old;
                    d18o_energy = d18o_energy_old;
                    expected_ages(d->N, d->dc, d->cs, d->th, d->nc14, d->c14d, sed_rates, c14_depth_indices, c14_expected_ages);
                    expected_ages(d->N, d->dc, d->cs, d->th, d->nd18o, d->d18od, sed_rates, D18O_depth_indices, D18O_expected_ages);
                    bias_new = bias_old;
                }
            }

            for (k = 0; k < d->N; k++)
            {
                samples_out[i * oc->hmcc->ns * d->N + j * d->N + k] = sed_rates[k];
            }
            energy_out[i * oc->hmcc->ns + j] = energy_new;
            d18o_energy_out[i * oc->hmcc->ns + j] = d18o_energy;
            bias_out[i * oc->hmcc->ns + j] = bias_new;

            if (j > 100)
            {
                for (k = 0; k < oc->hmcc->nch; k++)
                {
#pragma omp barrier
                    if (k == i)
                    {
                        omp_set_lock(&deltaF_lock);
                        if (bias_new > *max_dfd_local)
                        {
                            *delta_F_denominator_sum_local *= exp(*max_dfd_local - bias_new);
                            *max_dfd_local = bias_new;
                        }
                        *delta_F_denominator_sum_local += exp(bias_new - *max_dfd_local);
                        if (j > 100 + oc->w && j < stop_throw_idx)
                        {
                            *delta_F_denominator_sum_local -= exp(bias_out[i * oc->hmcc->ns + j - oc->w] - *max_dfd_local);
                        }
                        update_delta_F(oc->nt, oc->bs, delta_F_nominator_sum_local, delta_F_denominator_sum_local, delta_F_local, bias_out, d18o_energy_out, df_out, i, j, oc->hmcc->ns, *max_dfd_local, max_dfn_local, stop_throw_idx, oc->w);
                        if (i == 0 && j % 100 == 0)
                        {
                            printf("df: %f\n", df_out[i * oc->hmcc->ns * oc->nt + j * oc->nt + oc->nt - 1]);
                        }
                        if (j > 100 + oc->w && stop_throw_idx == INT_MAX && i == oc->hmcc->nch - 1 && df_out[i * oc->hmcc->ns * oc->nt + j * oc->nt + oc->nt - 1] > df_out[i * oc->hmcc->ns * oc->nt + (j - oc->w) * oc->nt + oc->nt - 1])
                        {
                            stop_throw_idx = 2 * j;
                            printf("sti: %d\n", stop_throw_idx);
                        }
                        omp_unset_lock(&deltaF_lock);
                    }
#pragma omp barrier
                }
            }
            else
            {
                for (k = 0; k < oc->nt; k++)
                    df_out[i * oc->hmcc->ns * oc->nt + j * oc->nt + k] = delta_F_local[k];
            }
        }

        if (!oc->sb)
        {
            free(delta_F_nominator_sum_local);
            free(delta_F_local);
            free(delta_F_denominator_sum_local);
            free(max_dfd_local);
            free(max_dfn_local);
        }
        mean_acceptance /= (oc->hmcc->ns * oc->nhmc);
        printf("%f\n", mean_acceptance);
    }
    free(delta_F_denominator_sum);
    free(max_dfd);
    omp_destroy_lock(&deltaF_lock);
}
