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

void hmc(
    HMCConfig *hmcc, Data *d, double *samples_out, double *energy_out)
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

#pragma omp parallel num_threads(hmcc->nch)
    {
        int i = omp_get_thread_num();
        int j, k, l, m;
        const gsl_rng_type *T;
        gsl_rng *r;

        gsl_rng_env_setup();
        T = gsl_rng_default;
        r = gsl_rng_alloc(T);
        gsl_rng_set(r, hmcc->sd + i);

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

        for (j = 0; j < d->N; j++)
        {
            sed_rates[j] = hmcc->sp[i * d->N + j];
            variables[j] = (log(sed_rates[j]) - d->pm) / d->ps;
        }

        expected_ages(d->N, d->dc, d->cs, d->th, d->nc14, d->c14d,
                      sed_rates, c14_depth_indices, c14_expected_ages);
        expected_ages(d->N, d->dc, d->cs, d->th, d->nd18o, d->d18od,
                      sed_rates, D18O_depth_indices, D18O_expected_ages);

        for (j = 0; j < hmcc->ns; j++)
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
            memcpy(variables_init, variables, d->N * sizeof(double));
            for (l = 0; l < d->N; l++)
            {
                momentum[l] = gsl_ran_gaussian(r, 1.0);
            }
            memcpy(momentum_init, momentum, d->N * sizeof(double));

            energy_old = energy_function(d, c14_expected_ages, D18O_expected_ages, variables);

            // Compute gradient at old state
            grad_energy_function(d, c14_depth_indices, c14_expected_ages, D18O_depth_indices, D18O_expected_ages, variables, gradient);
            for (l = 0; l < d->N; l++)
            {
                momentum[l] -= (hmcc->dt / 2) * (gradient[l]);
            }

            // Leapfrog integration
            for (l = 0; l < hmcc->ndt; l++)
            {
                for (m = 0; m < d->N; m++)
                {
                    variables[m] += hmcc->dt * momentum[m];
                    sed_rates[m] = exp(variables[m] * d->ps + d->pm);
                }

                expected_ages(d->N, d->dc, d->cs, d->th, d->nc14, d->c14d, sed_rates, c14_depth_indices, c14_expected_ages);
                expected_ages(d->N, d->dc, d->cs, d->th, d->nd18o, d->d18od, sed_rates, D18O_depth_indices, D18O_expected_ages);

                grad_energy_function(d, c14_depth_indices, c14_expected_ages, D18O_depth_indices, D18O_expected_ages, variables, gradient);

                if (l != hmcc->ndt - 1)
                {
                    for (m = 0; m < d->N; m++)
                    {
                        momentum[m] -= hmcc->dt * (gradient[m]);
                    }
                }
            }
            for (l = 0; l < d->N; l++)
            {
                momentum[l] -= 0.5 * hmcc->dt * (gradient[l]);
            }

            energy_new = energy_function(d, c14_expected_ages, D18O_expected_ages, variables);

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

            for (k = 0; k < d->N; k++)
            {
                samples_out[i * hmcc->ns * d->N + j * d->N + k] = sed_rates[k];
            }
            energy_out[i * hmcc->ns + j] = energy_new;
        }

        mean_acceptance /= (hmcc->ns);
        printf("%f\n", mean_acceptance);
    }
}
