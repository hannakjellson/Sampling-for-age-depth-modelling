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
#include "configs.h"

int binary_search(const double *arr, int n, double target)
{
    int left = 0, right = n;

    while (left < right)
    {
        int mid = left + (right - left) / 2;
        if (arr[mid] < target)
            left = mid + 1;
        else
            right = mid;
    }

    return left;
}

void interpolate_D18O(
    Data *d, double *D18O_times, double *D18O_reference_interp, double *D18O_interp_derivative)
{
    int index, i;
    double t0, t1, y0, y1, t, dt, dy, slope;
    for (i = 0; i < d->nd18o; i++)
    {
        index = binary_search(d->d18ort, d->nd18or, D18O_times[i]) - 1;

        t0 = d->d18ort[index];
        t1 = d->d18ort[index + 1];
        y0 = d->d18or[index];
        y1 = d->d18or[index + 1];
        t = D18O_times[i];

        dt = t1 - t0;
        dy = y1 - y0;

        slope = dy / dt;
        D18O_interp_derivative[i] = slope;
        D18O_reference_interp[i] = y0 + slope * (t - t0);
    }
}

void expected_ages(
    int N, double dc, double *cs, double th, int nd,
    double *deps, double *sed_rates, int *indices, double *ages_out)
{
    double cumulative_sum_vec[N + 1];
    cumulative_sum_vec[0] = 0;
    double cumulative_sum;
    int index, i;

    for (i = 1; i < N + 1; i++)
    {
        cumulative_sum_vec[i] = cumulative_sum_vec[i - 1] + sed_rates[i - 1] * dc;
    }

    for (i = 0; i < nd; i++)
    {
        index = indices[i];
        cumulative_sum = cumulative_sum_vec[index];
        cumulative_sum += sed_rates[index] * (deps[i] - cs[index]);
        ages_out[i] = th - cumulative_sum;
    }
}

double fprior(int N, double *nl_sed_rates)
{
    int i;
    double prior = 0.0;
    for (i = 0; i < N; i++)
    {
        prior += log(sqrt(2 * M_PI)) + nl_sed_rates[i] * nl_sed_rates[i] / 2;
    }

    return prior;
}

double c14_cond(Data *d, double *c14_expected_ages)
{
    int i;
    double c14_conditional = 0.0;
    double diff;
    for (i = 0; i < d->nc14; i++)
    {
        diff = d->c14[i] - c14_expected_ages[i];
        c14_conditional += d->ic14v[i] * diff * diff / (2);
    }

    return c14_conditional;
}

double d18o_cond(Data *d, double *expected_D18O_ages)
{
    int i;
    double D18O_reference_interp[d->nd18o];
    double D18O_interp_derivative[d->nd18o];
    interpolate_D18O(d, expected_D18O_ages, D18O_reference_interp, D18O_interp_derivative);

    double D18O_conditional = 0.0;
    for (i = 0; i < d->nd18o; i++)
    {
        double diff = d->d18o[i] - D18O_reference_interp[i];
        D18O_conditional += d->id18ov[i] * diff * diff / 2;
    }

    return D18O_conditional;
}

double energy_function(
    Data *d, double *c14_expected_ages, double *expected_D18O_ages, double *nl_sed_rates, double *d18o_energy, double beta)
{
    // Prior
    double prior = 0.0;
    double c14_conditional = 0.0;
    double D18O_conditional = 0.0;

    prior = fprior(d->N, nl_sed_rates);
    c14_conditional = c14_cond(d, c14_expected_ages);
    D18O_conditional = d18o_cond(d, expected_D18O_ages);
    *d18o_energy = c14_conditional + D18O_conditional;

    return prior + beta * (c14_conditional + D18O_conditional);
}

void grad_energy_function(
    Data *d, int *c14_indices, double *expected_c14_ages, int *D18O_indices, double *expected_D18O_ages, double *norm_log_sed_rates, double *gradient, double beta)
{
    // derivative of -log(p(z|data)) with respect to z
    int i, l, j;
    double f_l, diff, grad;
    double c14_conditional_term;

    double D18O_conditional_term;
    double D18O_reference_interp[d->nd18o];
    double D18O_interp_derivative[d->nd18o];
    double sed_rates[d->N];

    interpolate_D18O(
        d, expected_D18O_ages, D18O_reference_interp, D18O_interp_derivative);

    for (l = 0; l < d->N; l++)
    {
        // Prior
        grad = norm_log_sed_rates[l];
        sed_rates[l] = exp(norm_log_sed_rates[l] * d->ps + d->pm);

        // Conditional from C14 data
        for (i = 0; i < d->nc14; i++)
        {
            j = c14_indices[i];

            f_l = (l < j) ? -d->dc : (l == j) ? -(d->c14d[i] - d->cs[j])
                                              : 0.0;
            if (f_l != 0.0)
            {
                double diff = d->c14[i] - expected_c14_ages[i];
                grad -= beta * diff * f_l * d->ps * sed_rates[l] * d->ic14v[i];
            }
        }

        // Conditional from D18O
        for (i = 0; i < d->nd18o; i++)
        {
            j = D18O_indices[i];
            f_l = (l < j) ? -d->dc : (l == j) ? -(d->d18od[i] - d->cs[j])
                                              : 0.0;
            if (f_l != 0.0)
            {
                double diff = d->d18o[i] - D18O_reference_interp[i];
                double val = diff * D18O_interp_derivative[i] * f_l * d->ps * sed_rates[l] * d->id18ov[i];
                grad -= beta * val;
            }
        }
        gradient[l] = grad;
    }
}

void stoch_grad_energy_function(
    Data *d, int num_D18O_indices_stoch, int *D18O_indices_stoch, int *c14_indices, double *c14_expected_ages, int *D18O_indices, double *D18O_expected_ages, double *nl_sed_rates, double *gradient)
{
    // derivative of -log(p(log(sed_rates)|data)) with respect to log(sed_rates)
    int l, i, j, k;
    double f_l;
    double c14_conditional_term;

    double D18O_conditional_term;
    double D18O_reference_interp[d->nd18o];
    double D18O_interp_derivative[d->nd18o];
    double sed_rates[d->N];
    double grad;
    double ips = 1 / (d->ps * d->ps);

    interpolate_D18O(
        d, D18O_expected_ages, D18O_reference_interp, D18O_interp_derivative);

    for (l = 0; l < d->N; l++)
    {
        // Prior
        grad = (nl_sed_rates[l] - d->pm) * ips;
        sed_rates[l] = exp(nl_sed_rates[l] * d->ps + d->pm);

        // Conditional from C14 data
        for (i = 0; i < d->nc14; i++)
        {
            j = c14_indices[i];

            f_l = (l < j) ? -d->dc : (l == j) ? -(d->c14d[i] - d->cs[j])
                                              : 0.0;
            if (f_l != 0.0)
            {
                double diff = d->c14[i] - c14_expected_ages[i];
                grad -= diff * f_l * d->ps * sed_rates[l] * d->ic14v[i];
            }
        }

        // Conditional from D18O
        for (i = 0; i < num_D18O_indices_stoch; i++)
        {
            k = D18O_indices_stoch[i];
            j = D18O_indices[k];
            f_l = (l < j) ? -d->dc : (l == j) ? -(d->d18od[k] - d->cs[j])
                                              : 0.0;
            if (f_l != 0.0)
            {
                double diff = d->d18o[k] - D18O_reference_interp[k];
                grad -= (d->nd18o / num_D18O_indices_stoch) * diff * D18O_interp_derivative[k] * f_l * d->ps * sed_rates[l] * d->id18ov[k];
            }
        }
        gradient[l] = grad;
    }
}