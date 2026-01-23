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
#include "configs.h"

double norm(int num_elems, double vec[num_elems])
{
    double squared_sum = 0.0;
    for (int i = 0; i < num_elems; i++)
    {
        squared_sum += vec[i] * vec[i];
    }
    return sqrt(squared_sum);
}

typedef struct
{
    double energy;
    double d18o_energy;
    double *sample;
} EnergyIndex;

int cmp_energyindex(const void *a, const void *b)
{
    double diff = ((EnergyIndex *)a)->energy - ((EnergyIndex *)b)->energy;
    return (diff > 0) - (diff < 0); // returns -1,0,1
}

void adams(ADAMConfig *ac, Data *d, double *energy_out, double *samples_out)
{

    int i, l;
    EnergyIndex temp[ac->nsp];

    gsl_rng **rngs = malloc(ac->nsp * sizeof(gsl_rng *));
    const gsl_rng_type *T;
    gsl_rng_env_setup();
    T = gsl_rng_default;

    for (i = 0; i < ac->nsp; i++)
    {
        rngs[i] = gsl_rng_alloc(T);
        gsl_rng_set(rngs[i], ac->sd + i); // deterministic per-index seed
    }

    int num_d18O_points_stoch = 50;

#pragma omp parallel for
    for (i = 0; i < ac->nsp; i++)
    {
        int j, k;
        double c14_expected_ages[d->nc14];
        double D18O_expected_ages[d->nd18o];
        int c14_depth_indices[d->nc14];
        int D18O_depth_indices[d->nd18o];

        int *indices_for_stoch_d18O = malloc(d->nd18o * sizeof(int));
        int *random_indices_for_stoch_d18O = malloc(num_d18O_points_stoch * sizeof(int));

        for (j = 0; j < d->nd18o; j++)
        {
            indices_for_stoch_d18O[j] = j;
        }

        for (j = 0; j < d->nc14; j++)
        {
            c14_depth_indices[j] = binary_search(d->cs, d->N + 1, d->c14d[j]) - 1;
        }

        for (j = 0; j < d->nd18o; j++)
        {
            D18O_depth_indices[j] = binary_search(d->cs, d->N + 1, d->d18od[j]) - 1;
        }
        double variables[d->N], sed_rates[d->N], gradient[d->N];
        gsl_rng *r = rngs[i];
        temp[i].energy = DBL_MAX;
        temp[i].sample = malloc(d->N * sizeof(double));
        for (j = 0; j < d->N; j++)
        {
            sed_rates[j] = gsl_ran_lognormal(r, d->pm, d->ps);
            variables[j] = (log(sed_rates[j]) - d->pm) / d->ps;
        }

        double *velocity = calloc(d->N, sizeof(double));
        double velocity_factor = 0.9;
        double energy;
        double d18o_energy; // Not used here.

        for (j = 0; j < ac->mi; j++)
        {
            gsl_ran_choose(r, random_indices_for_stoch_d18O, num_d18O_points_stoch, indices_for_stoch_d18O, d->nd18o, sizeof(int));
            expected_ages(d->N, d->dc, d->cs, d->th, d->nc14, d->c14d, sed_rates, c14_depth_indices, c14_expected_ages);
            expected_ages(d->N, d->dc, d->cs, d->th, d->nd18o, d->d18od, sed_rates, D18O_depth_indices, D18O_expected_ages);
            energy = energy_function(d, c14_expected_ages, D18O_expected_ages, variables, &d18o_energy, 1);
            if (j % 10000 == 0)
            {
                printf("%d, %d, %f\n", i, j, energy);
            }
            if (energy < temp[i].energy)
            {
                temp[i].energy = energy;
                temp[i].d18o_energy = d18o_cond(d, D18O_expected_ages);
                for (int n = 0; n < d->N; n++)
                {
                    temp[i].sample[n] = sed_rates[n];
                }
            }

            stoch_grad_energy_function(d, num_d18O_points_stoch, random_indices_for_stoch_d18O, c14_depth_indices, c14_expected_ages, D18O_depth_indices, D18O_expected_ages, variables, gradient);

            if (norm(d->N, gradient) < ac->gl)
            {
                printf("Broke on grad");
                break;
            }

            double noise;
            for (k = 0; k < d->N; k++)
            {
                noise = gsl_ran_gaussian(r, ac->dt);
                velocity[k] = velocity_factor * velocity[k] - gradient[k] * ac->dt + noise;
                variables[k] += velocity[k];
                sed_rates[k] = exp(variables[k] * d->ps + d->pm);
            }
        }
    }
    for (i = 0; i < ac->nsp; i++)
    {
        gsl_rng_free(rngs[i]);
    }
    free(rngs);
    for (i = 0; i < ac->nsp; i++)
    {
        energy_out[i] = temp[i].energy;
        for (l = 0; l < d->N; l++)
        {
            samples_out[i * d->N + l] = temp[i].sample[l];
        }
    }
}
