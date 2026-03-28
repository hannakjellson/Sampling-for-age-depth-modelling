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

void normalize_sed_rates(const double *sed_rates, int N, double pm, const double *pLi, double *variables)
{
    int i, j;
    for (i = 0; i < N; i++)
    {
        variables[i] = 0;
        for (j = 0; j <= i; j++)
        {
            variables[i] += pLi[N * i + j] * (log(sed_rates[j]) - pm);
        }
    }
}

void unnormalize_sed_rates(double *sed_rates, int N, double pm, const double *pL, const double *variables)
{
    int i, j;
    double temp;
    for (i = 0; i < N; i++)
    {
        temp = 0;
        for (j = 0; j <= i; j++)
        {
            temp += pL[N * i + j] * variables[j];
        }
        sed_rates[i] = exp(temp + pm);
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
    Data *d, double *c14_expected_ages, double *expected_D18O_ages, double *nl_sed_rates, double *d18o_energy)
{
    // Prior
    double prior = 0.0;
    double c14_conditional = 0.0;
    double D18O_conditional = 0.0;

    prior = fprior(d->N, nl_sed_rates);
    c14_conditional = c14_cond(d, c14_expected_ages);
    D18O_conditional = d18o_cond(d, expected_D18O_ages);
    *d18o_energy = D18O_conditional;

    return prior + c14_conditional + D18O_conditional;
}

void grad_energy_function(
    Data *d, int *c14_indices, double *expected_c14_ages, int *D18O_indices, double *expected_D18O_ages, double *norm_log_sed_rates, double *sed_rates, double *gradient, double *d18o_gradient)
{
    // derivative of -log(p(z|data)) with respect to z
    int i, l, j, k;
    double f_l, diff, grad, d18o_grad;
    double c14_conditional_term;

    double D18O_conditional_term;
    double D18O_reference_interp[d->nd18o];
    double D18O_interp_derivative[d->nd18o];
    double val;

    interpolate_D18O(
        d, expected_D18O_ages, D18O_reference_interp, D18O_interp_derivative);

    for (l = 0; l < d->N; l++)
    {
        d18o_grad = 0;

        // Prior
        grad = norm_log_sed_rates[l];

        // Conditional from C14 data
        for (i = 0; i < d->nc14; i++)
        {
            j = c14_indices[i];
            diff = d->c14[i] - expected_c14_ages[i];
            val = 0;
            for (k = 0; k < j; k++)
            {
                val += d->pL[k * d->N + l] * sed_rates[k] * d->dc;
            }
            val += d->pL[j * d->N + l] * sed_rates[j] * (d->c14d[i] - d->cs[j]);
            grad += val * diff * d->ic14v[i];
        }

        // Conditional from D18O
        for (i = 0; i < d->nd18o; i++)
        {
            j = D18O_indices[i];
            diff = d->d18o[i] - D18O_reference_interp[i];
            val = 0;
            for (k = 0; k < j; k++)
            {
                val += d->pL[k * d->N + l] * sed_rates[k] * d->dc;
            }
            val += d->pL[j * d->N + l] * sed_rates[j] * (d->d18od[i] - d->cs[j]);
            d18o_grad += val * diff * D18O_interp_derivative[i] * d->id18ov[i];
        }
        grad += d18o_grad;
        gradient[l] = grad;
        d18o_gradient[l] = d18o_grad;
    }
}

double bias_potential(int num_temps, double *betas, double d18o_energy, double *delta_F)
{
    int k;
    double sum_for_V = 0;
    double temp_term;
    double max_temp_term = 0;

    for (k = 0; k < num_temps; k++)
    {
        temp_term = -(betas[k] - 1) * d18o_energy + delta_F[k];
        if (temp_term > max_temp_term)
        {
            sum_for_V *= exp(max_temp_term - temp_term);
            max_temp_term = temp_term;
            sum_for_V += 1;
        }
        else
        {
            sum_for_V += exp(temp_term - max_temp_term);
        }
    }
    double V = -max_temp_term - log(sum_for_V) + log(num_temps);
    return V;
}

void grad_bias(int N, int num_temps, double *betas, double *variables, double d18o_energy, double *d18o_gradient, double *delta_F, double *bias_gradient)
{
    // Derivative of the bias with respect to log(sedimentation_rates)
    int i, l;
    double sum_for_dV[N];
    double sum_for_V = 0;
    double exp_temp;
    double temp_factor;
    double temp_term;
    double max_temp_term = 0;

    for (l = 0; l < N; l++)
    {
        sum_for_dV[l] = 0;
    }

    for (i = 0; i < num_temps; i++)
    {
        temp_factor = (betas[i] - 1);
        temp_term = -temp_factor * d18o_energy + delta_F[i];
        if (temp_term > max_temp_term)
        {
            exp_temp = exp(max_temp_term - temp_term);
            sum_for_V *= exp_temp;
            sum_for_V += 1;
            for (l = 0; l < N; l++)
            {
                sum_for_dV[l] *= exp_temp;
                sum_for_dV[l] += temp_factor * d18o_gradient[l];
            }
            max_temp_term = temp_term;
        }
        else
        {
            exp_temp = exp(temp_term - max_temp_term);
            sum_for_V += exp_temp;
            for (l = 0; l < N; l++)
            {
                sum_for_dV[l] += temp_factor * d18o_gradient[l] * exp_temp;
            }
        }
    }

    if (sum_for_V == 0)
    {
        fprintf(stderr, "WARNING: sum_for_V = %f\n", sum_for_V);
        exit(0);
    }

    for (i = 0; i < N; i++)
    {
        bias_gradient[i] = sum_for_dV[i] / sum_for_V;
    }
}

void update_delta_F(int num_temps, double *betas, double energy, double *delta_F_nominator_sum, double *delta_F_denominator_sum, double *delta_F, double bias, double *df_out, int idx)
{
    int i;
    double temp_term;
    for (i = 0; i < num_temps; i++)
    {
        temp_term = (betas[i] - 1) * energy;
        delta_F_nominator_sum[i] += exp(-temp_term + bias);
        delta_F[i] = log(*delta_F_denominator_sum) - log(delta_F_nominator_sum[i]);
        df_out[idx + i] = delta_F[i];
    }
}

void stoch_grad_energy_function(
    Data *d, int num_D18O_indices_stoch, int *D18O_indices_stoch, int *c14_indices, double *c14_expected_ages, int *D18O_indices, double *D18O_expected_ages, double *nl_sed_rates, double *sed_rates, double *gradient)
{
    // derivative of -log(p(log(sed_rates)|data)) with respect to log(sed_rates)
    int l, i, j, k, m;
    double diff, val;
    double c14_conditional_term;

    double D18O_conditional_term;
    double D18O_reference_interp[d->nd18o];
    double D18O_interp_derivative[d->nd18o];
    double grad;

    interpolate_D18O(
        d, D18O_expected_ages, D18O_reference_interp, D18O_interp_derivative);

    for (l = 0; l < d->N; l++)
    {
        // Prior
        grad = nl_sed_rates[l];

        // Conditional from C14 data
        for (i = 0; i < d->nc14; i++)
        {
            j = c14_indices[i];
            diff = d->c14[i] - c14_expected_ages[i];
            val = 0;
            for (k = 0; k < j; k++)
            {
                val += d->pL[k * d->N + l] * sed_rates[k] * d->dc;
            }
            val += d->pL[j * d->N + l] * sed_rates[j] * (d->c14d[i] - d->cs[j]);
            grad += val * diff * d->ic14v[i];
        }

        // Conditional from D18O
        for (i = 0; i < num_D18O_indices_stoch; i++)
        {
            m = D18O_indices_stoch[i];
            j = D18O_indices[m];
            diff = d->d18o[m] - D18O_reference_interp[m];
            val = 0;
            for (k = 0; k < j; k++)
            {
                val += d->pL[k * d->N + l] * sed_rates[k] * d->dc;
            }
            val += d->pL[j * d->N + l] * sed_rates[j] * (d->d18od[m] - d->cs[j]);
            grad += val * diff * D18O_interp_derivative[m] * d->id18ov[m];
        }

        gradient[l] = grad;
    }
}