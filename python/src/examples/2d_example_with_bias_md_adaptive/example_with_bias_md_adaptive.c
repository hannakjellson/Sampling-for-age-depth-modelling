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

#define MAX_BIAS 10000
#define distance_threshold 1
#define THETA (-0.6 * M_PI / 4.0)
#define COS_THETA 0.8910065241883679
#define SIN_THETA -0.45399049973954675

// Still need to fix Z_n and include it in the update!

double energy_function(double *x)

{
    // double rotated_X = COS_THETA * x[0] - SIN_THETA * x[1];
    // double rotated_Y = SIN_THETA * x[0] + COS_THETA * x[1];
    // return pow(rotated_X, 4) + pow(rotated_Y, 4) - 2 * pow(rotated_X, 2) - 4 * pow(rotated_Y, 2) + rotated_X * rotated_Y + 0.3 * rotated_X + 0.1 * rotated_Y;
    return 1.34549 * x[0] * x[0] * x[0] * x[0] + 1.90211 * x[0] * x[0] * x[0] * x[1] + 3.92705 * x[0] * x[0] * x[1] * x[1] - 6.44246 * x[0] * x[0] - 1.90211 * x[0] * x[1] * x[1] * x[1] + 5.58721 * x[0] * x[1] + 1.33481 * x[0] + 1.34549 * x[1] * x[1] * x[1] * x[1] - 5.55754 * x[1] * x[1] + 0.904586 * x[1] + 18.5598;
}

void grad_energy(double x[2], double gradient[2])
{
    // double xr = COS_THETA * x[0] - SIN_THETA * x[1];
    // double yr = SIN_THETA * x[0] + COS_THETA * x[1];
    // double dxrdx = COS_THETA;
    // double dyrdx = SIN_THETA;
    // double dxrdy = -SIN_THETA;
    // double dyrdy = COS_THETA;
    // gradient[0] = dxrdx * (4 * xr * xr * xr - 4 * xr + yr + 0.3) + dyrdx * (4 * yr * yr * yr - 8 * yr + xr + 0.1);
    // gradient[1] = dxrdy * (4 * xr * xr * xr - 4 * xr + yr + 0.3) + dyrdy * (4 * yr * yr * yr - 8 * yr + xr + 0.1);
    gradient[0] = 4 * 1.34549 * x[0] * x[0] * x[0] + 3 * 1.90211 * x[0] * x[0] * x[1] + 2 * 3.92705 * x[0] * x[1] * x[1] - 2 * 6.44246 * x[0] - 1.90211 * x[1] * x[1] * x[1] + 5.58721 * x[1] + 1.33481;
    gradient[1] = 1.90211 * x[0] * x[0] * x[0] + 2 * 3.92705 * x[0] * x[0] * x[1] - 3 * 1.90211 * x[0] * x[1] * x[1] + 5.58721 * x[0] + 4 * 1.34549 * x[1] * x[1] * x[1] - 2 * 5.55754 * x[1] + 0.904586;
}

double get_probability_estimate(double x, double *bias_centers, double *bias_heights, double *bias_widths, int bias_count, double *weights, double gamma, double beta, double sum_weights)
{
    double probability_estimate = 0;
    for (int i = 0; i < bias_count; i++)
    {
        double dx = x - bias_centers[i];
        probability_estimate += weights[i] * bias_heights[i] * exp(-dx * dx / (2 * bias_widths[i] * bias_widths[i]));
    }
    if (bias_count > 0)
    {
        probability_estimate /= sum_weights;
    }
    return probability_estimate;
}

double get_probability_estimate_gradient(double x, double *bias_centers, double *bias_heights, double *bias_widths, int bias_count, double *weights, double gamma, double beta, double sum_weights)
{
    double dp = 0.0;
    for (int i = 0; i < bias_count; i++)
    {
        double dx = x - bias_centers[i];
        dp += weights[i] * bias_heights[i] * (-dx / (bias_widths[i] * bias_widths[i])) * exp(-dx * dx / (2 * bias_widths[i] * bias_widths[i]));
    }
    if (bias_count > 0)
    {
        dp /= sum_weights;
    }
    return dp;
}

double bias_potential(double x, double *bias_centers, double *bias_heights, double *bias_widths, int bias_count, double *weights, double gamma, double beta, double sum_weights, double Z, double DeltaE)
{
    double probability_estimate = get_probability_estimate(x, bias_centers, bias_heights, bias_widths, bias_count, weights, gamma, beta, sum_weights);
    double epsilon = exp(-beta * DeltaE / (1.0 - (1.0 / gamma)));
    double V = (1.0 - (1.0 / gamma)) * log(probability_estimate / Z + epsilon) / beta;
    return V;
}

void grad_bias(double x, double *bias_centers, double *bias_heights, double *bias_widths, int bias_count, double *weights, double gamma, double beta, double sum_weights, double Z, double DeltaE, double gradient[2])
{
    double probability_estimate = get_probability_estimate(x, bias_centers, bias_heights, bias_widths, bias_count, weights, gamma, beta, sum_weights);
    double probability_estimate_gradient = get_probability_estimate_gradient(x, bias_centers, bias_heights, bias_widths, bias_count, weights, gamma, beta, sum_weights);
    double epsilon = exp(-beta * DeltaE / (1.0 - (1.0 / gamma)));
    double dV = (1.0 - (1.0 / gamma)) * probability_estimate_gradient / (Z * beta * ((probability_estimate / Z) + epsilon));
    gradient[0] = dV;
    gradient[1] = 0;
}

void new_kernel(double *bias_centers, double *bias_heights, double *bias_widths, int *bias_count, int i)
{
    double new_bias_height = bias_heights[i + 1] + bias_heights[i];
    double new_bias_center = (1.0 / new_bias_height) * (bias_heights[i] * bias_centers[i] + bias_heights[i + 1] * bias_centers[i + 1]);
    double new_bias_width = sqrt((1.0 / new_bias_height) * (bias_heights[i] * (pow(bias_widths[i], 2) + pow(bias_centers[i], 2)) + bias_heights[i + 1] * (pow(bias_widths[i + 1], 2) + pow(bias_centers[i + 1], 2))));

    bias_heights[i] = new_bias_height;
    bias_centers[i] = new_bias_center;
    bias_widths[i] = new_bias_width;
    memmove(&bias_heights[i + 1], &bias_heights[i + 2], (*bias_count - i - 2) * sizeof(double));
    memmove(&bias_centers[i + 1], &bias_centers[i + 2], (*bias_count - i - 2) * sizeof(double));
    memmove(&bias_widths[i + 1], &bias_widths[i + 2], (*bias_count - i - 2) * sizeof(double));

    (*bias_count)--;
}

void merge_kernels(double *bias_centers, double *bias_heights, double *bias_widths, int *bias_count)
{
    double distance;

    for (int i = 0; i < *bias_count - 1; i++)
    {
        distance = bias_centers[i + 1] - bias_centers[i];

        int count = 0;
        while (distance < distance_threshold && i + 1 < *bias_count)
        {
            new_kernel(bias_centers, bias_heights, bias_widths, bias_count, i);
            distance = bias_centers[i + 1] - bias_centers[i];
            count++;
        }
    }
}

int binary_search(double *arr, int n, double target)
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

void deposit_gaussian(double x, double width, double *bias_centers, double *bias_heights, double *bias_widths, int *bias_count, int iterations)
{
    if (*bias_count < MAX_BIAS)
    {
        int index = binary_search(bias_centers, *bias_count, x);
        memmove(&bias_centers[index + 1], &bias_centers[index], (*bias_count - index) * sizeof(double));
        memmove(&bias_heights[index + 1], &bias_heights[index], (*bias_count - index) * sizeof(double));
        memmove(&bias_widths[index + 1], &bias_widths[index], (*bias_count - index) * sizeof(double));

        bias_centers[index] = x;
        bias_heights[index] = (1.0 / sqrt(2.0 * M_PI * width * width));
        bias_widths[index] = width;
        (*bias_count)++;
        if ((iterations + 1) % 100 == 0)
        {
            merge_kernels(bias_centers, bias_heights, bias_widths, bias_count);
        }
    }
}

double rand_normal()
{
    double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
    double u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);
}

void md(double dt, int num_MD, int num_dt, int num_SP, double bias_std,
        double *samples_out, double *energy_out, double *bias_out, double *bias_std_out,
        double gamma, double beta, double d, double DeltaE)
{

    double Z = 1.0;

#pragma omp parallel for
    for (int k = 0; k < num_SP; k++)
    {
        const gsl_rng_type *T;
        gsl_rng *r;

        gsl_rng_env_setup();
        T = gsl_rng_default;
        r = gsl_rng_alloc(T);
        unsigned long seed = k;
        gsl_rng_set(r, seed);

        double bias_centers[MAX_BIAS];
        double bias_widths[MAX_BIAS];
        double bias_heights[MAX_BIAS];
        int bias_count = 0;

        double *weights = malloc(num_MD * sizeof(double));
        double sum_weights = 0;
        double sum_squared_weights = 0;

        double N_eff;

        double x = gsl_ran_gaussian(r, 2);
        double y = gsl_ran_gaussian(r, 2);
        double x_vec[2];
        x_vec[0] = x;
        x_vec[1] = y;
        double min_energy = energy_function(x_vec);
        for (int i = 0; i < 10; i++)
        {
            double x_new = gsl_ran_gaussian(r, 2);
            double x_vec_new[2];
            x_vec_new[0] = x_new;
            x_vec_new[1] = x_vec[1];
            double energy_new = energy_function(x_vec_new);
            if (energy_new < min_energy)
            {
                min_energy = energy_new;
                x_vec[0] = x_vec_new[0];
            }
        }

        for (int j = 0; j < num_MD; j++)
        {
            double potential = bias_potential(x_vec[0], bias_centers, bias_heights, bias_widths, bias_count, weights, gamma, beta, sum_weights, Z, DeltaE);

            weights[j] = exp(beta * potential);
            sum_weights += weights[j];
            sum_squared_weights += weights[j] * weights[j];
            N_eff = sum_weights * sum_weights / sum_squared_weights;
            double bias_std_j = bias_std * pow(N_eff * (d + 2) / 4.0, -1.0 / (d + 4.0));

            for (int l = 0; l < 2; l++)
            { // 2 is the number of dimensions
                samples_out[2 * num_MD * k + 2 * j + l] = x_vec[l];
            }
            energy_out[k * num_MD + j] = energy_function(x_vec);
            bias_out[k * num_MD + j] = potential;
            bias_std_out[k * num_MD + j] = bias_std_j;

            printf("%d\n", j);
            deposit_gaussian(x_vec[0], bias_std_j, bias_centers, bias_heights, bias_widths, &bias_count, j);

            // Trying some Langevin step, this is basically gradient descent.
            for (int l = 0; l < num_dt; l++)
            {
                double noise_x = sqrt(dt) * gsl_ran_gaussian(r, 1); // dont think it should matter whether i deposit bias before or after this as the gradient of the deposited bias is zero.
                double noise_y = sqrt(dt) * gsl_ran_gaussian(r, 1); // dont think it should matter whether i deposit bias before or after this as the gradient of the deposited bias is zero.
                double force[2], grad_energy_vec[2], grad_bias_vec[2];
                grad_energy(x_vec, grad_energy_vec);
                grad_bias(x_vec[0], bias_centers, bias_heights, bias_widths, bias_count, weights, gamma, beta, sum_weights, Z, DeltaE, grad_bias_vec);
                force[0] = -grad_energy_vec[0] - grad_bias_vec[0];
                force[1] = -grad_energy_vec[1] - grad_bias_vec[1];
                x_vec[0] += (dt / 2) * force[0] + noise_x;
                x_vec[1] += (dt / 2) * force[1] + noise_y;
            }
        }
    }
}