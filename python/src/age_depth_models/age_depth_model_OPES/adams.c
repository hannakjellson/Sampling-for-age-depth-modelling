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
    double *sample;
} EnergyIndex;

int cmp_energyindex(const void *a, const void *b)
{
    double diff = ((EnergyIndex *)a)->energy - ((EnergyIndex *)b)->energy;
    return (diff > 0) - (diff < 0); // returns -1,0,1
}

void adams(int N, double delta_c, const double *cs, double a, double b, double theta,
           int num_c14_depths, int num_D18O_depths, int num_D18O_reference_times, const double *c14_ages, const double *c14_depths,
           const double *c14_sigma, const double *D18O, const double *D18O_depths, const double *D18O_sigma,
           const double *D18O_reference, const double *D18O_reference_times, int num_local_sp, int max_iter, double stepsize, double *Eout, double grad_lim, double *samples_out, int num_chains)
{

    EnergyIndex temp[num_local_sp];

    gsl_rng **rngs = malloc(num_local_sp * sizeof(gsl_rng *));
    const gsl_rng_type *T;
    gsl_rng_env_setup();
    T = gsl_rng_default;

    for (int i = 0; i < num_local_sp; i++)
    {
        rngs[i] = gsl_rng_alloc(T);
        gsl_rng_set(rngs[i], 41 + i); // deterministic per-index seed
    }

    int num_d18O_points_stoch = 50;

#pragma omp parallel for
    for (int i = 0; i < num_local_sp; i++)
    {
        double c14_expected_ages[num_c14_depths];
        double D18O_expected_ages[num_D18O_depths];
        int c14_depth_indices[num_c14_depths];
        double inv_c14_var[num_c14_depths];

        int *indices_for_stoch_d18O = malloc(num_D18O_depths * sizeof(int));
        for (int j = 0; j < num_D18O_depths; j++)
        {
            indices_for_stoch_d18O[j] = j;
        }
        int *random_indices_for_stoch_d18O = malloc(num_d18O_points_stoch * sizeof(int));

        for (int j = 0; j < num_c14_depths; j++)
        {
            c14_depth_indices[j] = binary_search(cs, N + 1, c14_depths[j]) - 1;
            inv_c14_var[j] = 1.0 / (c14_sigma[j] * c14_sigma[j]);
        }

        int D18O_depth_indices[num_D18O_depths];
        double inv_D18O_var[num_D18O_depths];
        for (int j = 0; j < num_D18O_depths; j++)
        {
            D18O_depth_indices[j] = binary_search(cs, N + 1, D18O_depths[j]) - 1;
            inv_D18O_var[j] = 1.0 / (D18O_sigma[j] * D18O_sigma[j]);
        }
        double variables[N], log_variables[N], gradient[N];
        gsl_rng *r = rngs[i];
        temp[i].energy = DBL_MAX;
        temp[i].sample = malloc(N * sizeof(double));
        for (int j = 0; j < N; j++)
        {
            variables[j] = gsl_ran_gamma(r, a, 1 / b);
            log_variables[j] = log(variables[j]);
        }

        double *velocity = calloc(N, sizeof(double));
        double velocity_factor = 0.9;
        double cap = 50000;

        for (int j = 0; j < max_iter; j++)
        {
            gsl_ran_choose(r, random_indices_for_stoch_d18O, num_d18O_points_stoch, indices_for_stoch_d18O, num_D18O_depths, sizeof(int));
            expected_ages(N, delta_c, cs, theta, num_c14_depths, c14_depths,
                          variables, c14_depth_indices, c14_expected_ages);
            expected_ages(N, delta_c, cs, theta, num_D18O_depths, D18O_depths,
                          variables, D18O_depth_indices, D18O_expected_ages);
            double energy = energy_function(N, delta_c, cs, a, b, theta, num_c14_depths, num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma, c14_depth_indices, inv_c14_var, c14_expected_ages, D18O, D18O_depths, D18O_sigma, D18O_depth_indices, inv_D18O_var, D18O_expected_ages, D18O_reference, D18O_reference_times, variables);

            if (energy < temp[i].energy)
            {
                temp[i].energy = energy;
                for (int n = 0; n < N; n++)
                {
                    temp[i].sample[n] = variables[n];
                }
            }
            cap = temp[i].energy + 15;

            stoch_grad_energy_function(N, num_d18O_points_stoch, random_indices_for_stoch_d18O, delta_c, cs, a, b, theta, num_c14_depths,
                                       num_D18O_depths, num_D18O_reference_times, c14_ages, c14_depths, c14_sigma, c14_depth_indices, inv_c14_var, c14_expected_ages, D18O, D18O_depths, D18O_sigma,
                                       D18O_depth_indices, inv_D18O_var, D18O_expected_ages, D18O_reference, D18O_reference_times, variables, gradient);

            if (norm(N, gradient) < grad_lim)
            {
                printf("Broke on grad");
                break;
            }

            for (int k = 0; k < N; k++)
            {
                double noise = gsl_ran_gaussian(r, stepsize);
                velocity[k] = velocity_factor * velocity[k] - gradient[k] * stepsize + noise;
                log_variables[k] += velocity[k];
                variables[k] = exp(log_variables[k]);
            }
        }
    }
    for (int i = 0; i < num_local_sp; i++)
    {
        gsl_rng_free(rngs[i]);
    }
    free(rngs);
    for (int i = 0; i < num_local_sp; i++)
    {
        Eout[i] = temp[i].energy; // store energies in Eout[0..3]
        for (int n = 0; n < N; n++)
        {
            samples_out[i * N + n] = temp[i].sample[n];
        }
    }
}