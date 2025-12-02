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

#define MAX_NBR_PC 2

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
    int num_D18O_depths, int num_D18O_reference_times,
    const double *D18O_times, const double *D18O_reference, const double *D18O_reference_times,
    double D18O_reference_interp[num_D18O_depths], double D18O_interp_derivative[num_D18O_depths], const double *D18O_depths)
{
    for (int i = 0; i < num_D18O_depths; i++)
    {
        int index = binary_search(D18O_reference_times, num_D18O_reference_times, D18O_times[i]) - 1;

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
    int N, double delta_c, const double *cs, double a, double b, double theta, double beta,
    int num_c14_depths, int num_D18O_depths, int num_D18O_reference_times, const double *c14_ages, const double *c14_depths,
    const double *c14_sigma, int c14_depth_indices[num_c14_depths], double inv_c14_var[num_c14_depths], double c14_expected_ages[num_c14_depths], const double *D18O, const double *D18O_depths, const double *D18O_sigma,
    int D18O_depth_indices[num_D18O_depths], double inv_D18O_var[num_D18O_depths], double expected_D18O_ages[num_D18O_depths], const double *D18O_reference, const double *D18O_reference_times, const double *sed_rates)
{

    //-log(p(log(sed_rates)|data))

    // Prior
    double prior = 0.0;
    for (int i = 0; i < N; i++)
    {
        prior += -a * log(sed_rates[i]) + b * sed_rates[i];
    }

    double c14_conditional = 0.0;
    for (int i = 0; i < num_c14_depths; i++)
    {
        double diff = c14_ages[i] - c14_expected_ages[i];
        c14_conditional += inv_c14_var[i] * diff * diff / (2);
    }

    double D18O_reference_interp[num_D18O_depths];
    double D18O_interp_derivative[num_D18O_depths];
    interpolate_D18O(
        num_D18O_depths, num_D18O_reference_times, expected_D18O_ages, D18O_reference, D18O_reference_times, D18O_reference_interp, D18O_interp_derivative, D18O_depths);

    double D18O_conditional = 0.0;
    for (int i = 0; i < num_D18O_depths; i++)
    {
        double diff = D18O[i] - D18O_reference_interp[i];
        D18O_conditional += inv_D18O_var[i] * diff * diff / 2;
    }
    return (prior + c14_conditional + D18O_conditional) * beta;
}

void grad_energy_function(
    int N, double delta_c, const double *cs, double a, double b, double theta, double beta,
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

            f_l = (l < j) ? -delta_c : (l == j) ? -(c14_depths[i] - cs[j])
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
        gradient[l] = grad * beta;
        // printf("gradient in function %f\n", grad);
    }
}

double get_CV_point(int N, double variables[N], double pc[N], double sp_mean[N])
{

    double sum = 0;
    for (int i = 0; i < N; i++)
    {
        sum += (variables[i] - sp_mean[i]) * pc[i];
    }
    return sum;
}

double bias_potential(int num_pcs, double CV_point[num_pcs], int num_lambda, int num_temps, double gaussian_centers[num_lambda], double betas[num_temps], double beta0, double energy, double sigma, double sigma_2, double delta_F[(int)pow(num_lambda, num_pcs) * num_temps], int start_index[MAX_NBR_PC], int end_index[MAX_NBR_PC], bool umbrella, bool temp)
{
    double sum_for_V = 0;
    double umbrella_term_i;
    double umbrella_term_j;
    double temp_term;
    double max_term = -INFINITY;
    double total_term;
    int num_lambda2 = (num_pcs == 2) ? num_lambda : 1;

    for (int i = start_index[0]; i < end_index[0]; i++)
    {
        umbrella_term_i = umbrella ? (pow(CV_point[0] - gaussian_centers[i], 2) / (2 * sigma_2)) : 0;
        for (int j = start_index[1]; j < end_index[1]; j++)
        {
            umbrella_term_j = (umbrella && num_pcs == 2) ? (pow(CV_point[1] - gaussian_centers[j], 2) / (2 * sigma_2)) : 0;
            // printf("j %f\n", umbrella_term_j);
            for (int k = 0; k < num_temps; k++)
            {
                temp_term = temp ? (betas[k] - beta0) * energy : 0;
                // printf("temp %f\n", temp_term);
                total_term = -(umbrella_term_i + umbrella_term_j + temp_term) + delta_F[i * num_lambda2 * num_temps + j * num_temps + k];
                if (total_term > max_term)
                {
                    sum_for_V *= exp(max_term - total_term);
                    max_term = total_term;
                }
                sum_for_V += exp(total_term - max_term);
            }
        }
    }
    double V = -log(sum_for_V) - max_term + log(pow(num_lambda, num_pcs) * num_temps);
    return V;
}

void grad_bias(int N, double delta_c, int num_pcs, double pcs[num_pcs * N], double CV_point[num_pcs], double variables[N], int num_lambda, int num_temps, double gaussian_centers[num_lambda], double betas[num_temps], double beta0, double energy, double gradient[N], double sigma, double sigma_2, double delta_F[(int)pow(num_lambda, num_pcs) * num_temps], int start_index[MAX_NBR_PC], int end_index[MAX_NBR_PC], bool umbrella, bool temp, double bias_gradient[N])
{
    // Derivative of the bias with respect to log(sedimentation_rates)
    double sum_for_dV[N];
    double sum_for_V = 0;
    double gaussian_diff_i;
    double gaussian_diff_2_i;
    double gaussian_diff_j;
    double gaussian_diff_2_j;
    double dsdx[MAX_NBR_PC * N];
    double exp_term = 0;
    double umbrella_factor_i;
    double umbrella_term_i;
    double umbrella_factor_j;
    double umbrella_term_j;
    double temp_factor;
    double temp_term;
    int num_lambda2 = (num_pcs == 2) ? num_lambda : 1;
    double pre_max_term;
    double max_term = -INFINITY;
    double total_term;
    bool new_max;

    for (int j = 0; j < N; j++)
    {
        if (umbrella)
        {
            for (int i = 0; i < num_pcs; i++)
            {
                dsdx[i * N + j] = variables[j] * pcs[i * N + j];
            }
        }
        sum_for_dV[j] = 0;
    }

    for (int i = start_index[0]; i < end_index[0]; i++)
    {
        gaussian_diff_i = umbrella ? (CV_point[0] - gaussian_centers[i]) : 0;
        gaussian_diff_2_i = umbrella ? gaussian_diff_i * gaussian_diff_i : 0;
        umbrella_term_i = umbrella ? gaussian_diff_2_i / (2 * sigma_2) : 0;
        umbrella_factor_i = umbrella ? gaussian_diff_i / sigma_2 : 0;
        for (int j = start_index[1]; j < end_index[1]; j++)
        {
            gaussian_diff_j = (umbrella && num_pcs == 2) ? (CV_point[1] - gaussian_centers[j]) : 0;
            gaussian_diff_2_j = (umbrella && num_pcs == 2) ? gaussian_diff_j * gaussian_diff_j : 0;
            umbrella_term_j = (umbrella && num_pcs == 2) ? gaussian_diff_2_j / (2 * sigma_2) : 0;
            umbrella_factor_j = (umbrella && num_pcs == 2) ? gaussian_diff_j / sigma_2 : 0;
            for (int k = 0; k < num_temps; k++)
            {
                temp_term = temp ? (betas[k] - beta0) * energy : 0;
                temp_factor = temp ? (betas[k] - beta0) : 0;
                total_term = -(umbrella_term_i + umbrella_term_j + temp_term) + delta_F[i * num_lambda2 * num_temps + j * num_temps + k];
                if (total_term > max_term)
                {
                    pre_max_term = max_term;
                    max_term = total_term;
                    new_max = true;
                }
                else
                    new_max = false;
                exp_term = exp(total_term - max_term);
                for (int l = 0; l < N; l++)
                {
                    if (new_max)
                    {
                        sum_for_dV[l] *= exp(pre_max_term - max_term);
                    }
                    sum_for_dV[l] += exp_term * (temp_factor * gradient[l] + (umbrella_factor_i * dsdx[l]) + (umbrella_factor_j * dsdx[N + l]));
                }
                if (new_max)
                {
                    sum_for_V *= exp(pre_max_term - max_term);
                }
                sum_for_V += exp_term;
            }
        }
    }
    if (sum_for_V == 0)
    {
        fprintf(stderr, "WARNING: sum_for_V = %f\n", sum_for_V);
        exit(0);
    }

    for (int i = 0; i < N; i++)
    {
        bias_gradient[i] = sum_for_dV[i] / sum_for_V;
    }
}

void update_delta_F(int num_pcs, double CV_point[num_pcs], int num_lambda, int num_temps, double sigma_2, double dE, double gaussian_centers[num_lambda], double betas[num_temps], double beta0, double energy, double delta_F_nominator_sum[(int)pow(num_lambda, num_pcs) * num_temps], double delta_F_denominator_sum, double max_delta_F_nominator_sum_term[(int)pow(num_lambda, num_pcs) * num_temps], double max_delta_F_denominator_sum_term, double delta_F[(int)pow(num_lambda, num_pcs) * num_temps], double potential, bool umbrella, bool temp, int sample, int dfs)
{
    double gaussian_diff_i;
    double gaussian_diff_2_i;
    double gaussian_diff_j;
    double gaussian_diff_2_j;
    double umbrella_term_i;
    double umbrella_term;
    double temp_term;
    int num_lambda2 = num_pcs == 2 ? num_lambda : 1;
    for (int lambda_index = 0; lambda_index < num_lambda; lambda_index++)
    {
        gaussian_diff_i = umbrella ? (CV_point[0] - gaussian_centers[lambda_index]) : 0;
        gaussian_diff_2_i = umbrella ? gaussian_diff_i * gaussian_diff_i : 0;
        umbrella_term_i = umbrella ? gaussian_diff_2_i / (2 * sigma_2) : 0;
        for (int lambda_index_2 = 0; lambda_index_2 < num_lambda2; lambda_index_2++)
        {
            gaussian_diff_j = (umbrella && num_pcs == 2) ? (CV_point[1] - gaussian_centers[lambda_index_2]) : 0;
            gaussian_diff_2_j = (umbrella && num_pcs == 2) ? gaussian_diff_j * gaussian_diff_j : 0;
            umbrella_term = (umbrella && num_pcs == 2) ? umbrella_term_i + gaussian_diff_2_j / (2 * sigma_2) : umbrella_term_i;
            for (int beta_index = 0; beta_index < num_temps; beta_index++)
            {
                temp_term = temp ? (betas[beta_index] - beta0) * energy : 0;
                double new_max = -(umbrella_term + temp_term) + potential;
                if (new_max > max_delta_F_nominator_sum_term[lambda_index * num_lambda2 * num_temps + lambda_index_2 * num_temps + beta_index])
                {
                    double max_diff = max_delta_F_nominator_sum_term[lambda_index * num_lambda2 * num_temps + lambda_index_2 * num_temps + beta_index] - new_max;
                    delta_F_nominator_sum[lambda_index * num_lambda2 * num_temps + lambda_index_2 * num_temps + beta_index] *= exp(max_diff);
                    max_delta_F_nominator_sum_term[lambda_index * num_lambda2 * num_temps + lambda_index_2 * num_temps + beta_index] = new_max;
                }
                delta_F_nominator_sum[lambda_index * num_lambda2 * num_temps + lambda_index_2 * num_temps + beta_index] += exp(-(umbrella_term + temp_term) + potential - max_delta_F_nominator_sum_term[lambda_index * num_lambda2 * num_temps + lambda_index_2 * num_temps + beta_index]);
                // if (sample > dfs)
                // {
                delta_F[lambda_index * num_lambda2 * num_temps + lambda_index_2 * num_temps + beta_index] = max_delta_F_denominator_sum_term + log(delta_F_denominator_sum) - max_delta_F_nominator_sum_term[lambda_index * num_lambda2 * num_temps + lambda_index_2 * num_temps + beta_index] - log(delta_F_nominator_sum[lambda_index * num_lambda2 * num_temps + lambda_index_2 * num_temps + beta_index]);

                if (delta_F[lambda_index * num_lambda2 * num_temps + lambda_index_2 * num_temps + beta_index] >= dE)
                {
                    delta_F[lambda_index * num_lambda2 * num_temps + lambda_index_2 * num_temps + beta_index] = dE;
                }
                // }
            }
        }
    }
}

void stoch_grad_energy_function(
    int N, int num_D18O_indices_stoch, int *D18O_indices_stoch, double delta_c, const double *cs, double a, double b, double theta, double beta,
    int num_c14_depths, int num_D18O_depths, int num_D18O_reference_times, const double *c14_ages, const double *c14_depths,
    const double *c14_sigma, int c14_indices[num_c14_depths], double inv_c14_var[num_c14_depths], double c14_expected_ages[num_c14_depths], const double *D18O, const double *D18O_depths, const double *D18O_sigma,
    int D18O_indices[num_D18O_depths], double inv_D18O_var[num_D18O_depths], double D18O_expected_ages[num_D18O_depths], const double *D18O_reference, const double *D18O_reference_times, const double *sed_rates, double gradient[N])
{
    // derivative of -log(p(log(sed_rates)|data)) with respect to log(sed_rates)
    double f_l;
    double c14_conditional_term;

    double D18O_conditional_term;
    double D18O_reference_interp[num_D18O_depths];
    double D18O_interp_derivative[num_D18O_depths];

    interpolate_D18O(
        num_D18O_depths, num_D18O_reference_times, D18O_expected_ages, D18O_reference, D18O_reference_times, D18O_reference_interp, D18O_interp_derivative, D18O_depths);

    for (int l = 0; l < N; l++)
    {
        // Prior
        double grad = -a + b * sed_rates[l];

        // Conditional from C14 data
        for (int i = 0; i < num_c14_depths; i++)
        {
            int j = c14_indices[i];

            f_l = (l < j) ? -delta_c : (l == j) ? -(c14_depths[i] - cs[j])
                                                : 0.0;
            if (f_l != 0.0)
            {
                double diff = c14_ages[i] - c14_expected_ages[i];
                grad -= diff * f_l * sed_rates[l] * inv_c14_var[i];
            }
        }

        // Conditional from D18O
        for (int i = 0; i < num_D18O_indices_stoch; i++)
        {
            int ind = D18O_indices_stoch[i];
            int j = D18O_indices[ind];
            double f_l = (l < j) ? -delta_c : (l == j) ? -(D18O_depths[ind] - cs[j])
                                                       : 0.0;
            if (f_l != 0.0)
            {
                double diff = D18O[ind] - D18O_reference_interp[ind];
                grad -= (num_D18O_depths / num_D18O_indices_stoch) * diff * D18O_interp_derivative[ind] * f_l * sed_rates[l] * inv_D18O_var[ind];
            }
        }
        gradient[l] = grad * beta;
    }
}