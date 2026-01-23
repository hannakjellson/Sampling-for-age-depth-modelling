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

void pt(
    PTConfig *pc, Data *d, double *samples_out, double *energy_out, double *d18o_energy_out)
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

    double curr_samples[pc->nt * d->N];
    double curr_energies[pc->nt];

#pragma omp parallel num_threads(pc->nt)
    {
        int i = omp_get_thread_num();
        int j, k, l, m;
        const gsl_rng_type *T;
        gsl_rng *r;

        gsl_rng_env_setup();
        T = gsl_rng_default;
        r = gsl_rng_alloc(T);
        gsl_rng_set(r, pc->hmcc->sd + i);

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

        // HMC variables
        double hmc_dE;
        double acceptance_prob;
        double mean_acceptance = 0.0;
        double random_number;
        bool keep_variables;
        bool even;
        int A;
        double rej = 0;
        double alpha;
        double *d18o_energy = malloc(sizeof(double));

        double temp_sample;

        for (j = 0; j < d->N; j++)
        {
            sed_rates[j] = pc->hmcc->sp[i * d->N + j];
            variables[j] = (log(sed_rates[j]) - d->pm) / d->ps;
        }

        expected_ages(d->N, d->dc, d->cs, d->th, d->nc14, d->c14d,
                      sed_rates, c14_depth_indices, c14_expected_ages);
        expected_ages(d->N, d->dc, d->cs, d->th, d->nd18o, d->d18od,
                      sed_rates, D18O_depth_indices, D18O_expected_ages);

        for (j = 0; j < pc->hmcc->ns; j++)
        {
            even = (j % 2 == 0);
            if (isnan(variables[0]))
            {
                printf("Variables are nan, aborting");
                exit(EXIT_FAILURE);
            }
            if (j % 100 == 0 && i == 0)
            {
                printf("Sample %d\n", j);
            }
            for (k = 0; k < pc->nhmc; k++)
            {
                memcpy(variables_init, variables, d->N * sizeof(double));
                for (l = 0; l < d->N; l++)
                {
                    momentum[l] = gsl_ran_gaussian(r, 1.0);
                }
                memcpy(momentum_init, momentum, d->N * sizeof(double));

                energy_old = energy_function(d, c14_expected_ages, D18O_expected_ages, variables, d18o_energy, pc->bs[i]);

                // Compute gradient at old state
                grad_energy_function(d, c14_depth_indices, c14_expected_ages, D18O_depth_indices, D18O_expected_ages, variables, gradient, pc->bs[i]);

                for (l = 0; l < d->N; l++)
                {
                    momentum[l] -= (pc->hmcc->dt / 2) * gradient[l];
                }

                // Leapfrog integration
                for (l = 0; l < pc->hmcc->ndt; l++)
                {
                    for (m = 0; m < d->N; m++)
                    {
                        variables[m] += pc->hmcc->dt * momentum[m];
                        sed_rates[m] = exp(variables[m] * d->ps + d->pm);
                    }

                    expected_ages(d->N, d->dc, d->cs, d->th, d->nc14, d->c14d, sed_rates, c14_depth_indices, c14_expected_ages);
                    expected_ages(d->N, d->dc, d->cs, d->th, d->nd18o, d->d18od, sed_rates, D18O_depth_indices, D18O_expected_ages);

                    grad_energy_function(d, c14_depth_indices, c14_expected_ages, D18O_depth_indices, D18O_expected_ages, variables, gradient, pc->bs[i]);

                    if (l != pc->hmcc->ndt - 1)
                    {
                        for (m = 0; m < d->N; m++)
                        {
                            momentum[m] -= pc->hmcc->dt * gradient[m];
                        }
                    }
                }
                for (l = 0; l < d->N; l++)
                {
                    momentum[l] -= 0.5 * pc->hmcc->dt * gradient[l];
                }

                energy_new = energy_function(d, c14_expected_ages, D18O_expected_ages, variables, d18o_energy, pc->bs[i]);

                kinetic_new = 0;
                kinetic_old = 0;

                for (l = 0; l < d->N; l++)
                {
                    kinetic_new += 0.5 * momentum[l] * momentum[l];
                    kinetic_old += 0.5 * momentum_init[l] * momentum_init[l];
                }

                hmc_dE = kinetic_new + energy_new - kinetic_old - energy_old;
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
                    expected_ages(d->N, d->dc, d->cs, d->th, d->nc14, d->c14d, sed_rates, c14_depth_indices, c14_expected_ages);
                    expected_ages(d->N, d->dc, d->cs, d->th, d->nd18o, d->d18od, sed_rates, D18O_depth_indices, D18O_expected_ages);
                }
            }

            for (k = 0; k < d->N; k++)
            {
                curr_samples[i * d->N + k] = sed_rates[k];
                samples_out[i * pc->hmcc->ns * d->N + j * d->N + k] = sed_rates[k];
            }
            curr_energies[i] = *d18o_energy;
            energy_out[i * pc->hmcc->ns + j] = energy_new;
            d18o_energy_out[i * pc->hmcc->ns + j] = *d18o_energy;

#pragma omp barrier
            if (i != pc->nt - 1)
            {
                if (((even && i % 2 == 0) || (!even && !(i % 2 == 0))))
                {
                    alpha = exp(fmin(0, (pc->bs[i + 1] - pc->bs[i]) * (curr_energies[(i + 1)] - curr_energies[i])));
                    rej += (1 - alpha);
                    A = gsl_ran_bernoulli(r, alpha);
                    if (A == 1)
                    {
                        for (l = 0; l < d->N; l++)
                        {
                            temp_sample = curr_samples[i * d->N + l];
                            curr_samples[i * d->N + l] = curr_samples[(i + 1) * d->N + l];
                            curr_samples[(i + 1) * d->N + l] = temp_sample;
                        }
                    }
                }
            }
#pragma omp barrier
        }

        for (l = 0; l < d->N; l++)
        {
            sed_rates[l] = curr_samples[i * d->N + l];
            variables[l] = (log(sed_rates[l]) - d->pm) / d->ps;
        }

        mean_acceptance /= (pc->hmcc->ns * pc->nhmc);
        rej = rej / pc->hmcc->ns;
        printf("mean accept: %d, %f\n", i, mean_acceptance);
        printf("reject: %d, %f\n", i, rej);
        free(d18o_energy);
    }
}
