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

double energy_function(double x[2])
{
    // double rotated_X = cos(-0.6 * M_PI / 4) * x[0] - sin(-0.6 * M_PI / 4) * x[1];
    // double rotated_Y = sin(-0.6 * M_PI / 4) * x[0] + cos(-0.6 * M_PI / 4) * x[1];
    // return pow(rotated_X, 4) + pow(rotated_Y, 4) - 2 * pow(rotated_X, 2) - 4 * pow(rotated_Y, 2) + rotated_X * rotated_Y + 0.3 * rotated_X + 0.1 * rotated_Y;
    return 1.34549 * x[0] * x[0] * x[0] * x[0] + 1.90211 * x[0] * x[0] * x[0] * x[1] + 3.92705 * x[0] * x[0] * x[1] * x[1] - 6.44246 * x[0] * x[0] - 1.90211 * x[0] * x[1] * x[1] * x[1] + 5.58721 * x[0] * x[1] + 1.33481 * x[0] + 1.34549 * x[1] * x[1] * x[1] * x[1] - 5.55754 * x[1] * x[1] + 0.904586 * x[1] + 18.5598;
}

void grad_energy(double x[2], double gradient[2])
{
    // double xr = cos(-0.6 * M_PI / 4) * x[0] - sin(-0.6 * M_PI / 4) * x[1];
    // double yr = sin(-0.6 * M_PI / 4) * x[0] + cos(-0.6 * M_PI / 4) * x[1];
    // double dxrdx = cos(-0.6 * M_PI / 4);
    // double dyrdx = sin(-0.6 * M_PI / 4);
    // double dxrdy = -sin(-0.6 * M_PI / 4);
    // double dyrdy = cos(-0.6 * M_PI / 4);
    // gradient[0] = dxrdx * (4 * pow(xr, 3) - 4 * xr + yr + 0.3) + dyrdx * (4 * pow(yr, 3) - 8 * yr + xr + 0.1);
    // gradient[1] = dxrdy * (4 * pow(xr, 3) - 4 * xr + yr + 0.3) + dyrdy * (4 * pow(yr, 3) - 8 * yr + xr + 0.1);
    gradient[0] = 4 * 1.34549 * x[0] * x[0] * x[0] + 3 * 1.90211 * x[0] * x[0] * x[1] + 2 * 3.92705 * x[0] * x[1] * x[1] - 2 * 6.44246 * x[0] - 1.90211 * x[1] * x[1] * x[1] + 5.58721 * x[1] + 1.33481;
    gradient[1] = 1.90211 * x[0] * x[0] * x[0] + 2 * 3.92705 * x[0] * x[0] * x[1] - 3 * 1.90211 * x[0] * x[1] * x[1] + 5.58721 * x[0] + 4 * 1.34549 * x[1] * x[1] * x[1] - 2 * 5.55754 * x[1] + 0.904586;
}

double bias_potential(double x[2], int num_lambda, double gaussian_centers[num_lambda], double sigma_2, double delta_F[num_lambda])
{
    double sum_for_V = 0;

    for (int i = 0; i < num_lambda; i++)
    {
        sum_for_V += exp(-(pow(energy_function(x) - gaussian_centers[i], 2) / (2 * sigma_2)) + delta_F[i]);
    }
    double V = -log(sum_for_V / num_lambda);
    return V;
}

void grad_bias(double x[2], int num_lambda, double gaussian_centers[num_lambda], double sigma, double sigma_2, double delta_F[num_lambda], double dV[2])
{
    double sum_for_dV = 0;
    double sum_for_V = 0;
    double grad_energy_val[2];
    grad_energy(x, grad_energy_val);

    for (int i = 0; i < num_lambda; i++)
    {
        if (fabs(energy_function(x) - gaussian_centers[i]) > 3 * sigma)
        {
            continue;
        }
        double gaussian_diff = (energy_function(x) - gaussian_centers[i]);
        double gaussian_diff_2 = gaussian_diff * gaussian_diff;
        sum_for_dV += (gaussian_diff / sigma_2) * exp(-(gaussian_diff_2 / (2 * sigma_2)) + delta_F[i]);
        sum_for_V += exp(-(gaussian_diff_2 / (2 * sigma_2)) + delta_F[i]);
    }

    dV[0] = sum_for_dV * grad_energy_val[0] / sum_for_V;
    dV[1] = sum_for_dV * grad_energy_val[1] / sum_for_V;
}

void update_delta_F(double x[2], int lambda_index, int num_lambda, double sigma_2, double gaussian_centers[num_lambda], double delta_F_nominator_sum[num_lambda], double delta_F_denominator_sum[num_lambda], double delta_F[num_lambda], double potential, double max_dF, int sample)
{
    // printf("%f\n", energy_function(x));
    delta_F_nominator_sum[lambda_index] += exp((-pow(energy_function(x) - gaussian_centers[lambda_index], 2) / (2 * sigma_2)) + potential);
    delta_F_denominator_sum[lambda_index] += exp(potential);

    if (sample > -1)
    {
        delta_F[lambda_index] = -log(delta_F_nominator_sum[lambda_index] / delta_F_denominator_sum[lambda_index]);
        if (delta_F[lambda_index] >= max_dF)
        {
            delta_F[lambda_index] = max_dF;
        }
    }
}

void hmc(double dt, int num_samples, int num_HMC, int num_dt, int num_SP, int num_lambda, double sigma, double dE, double max_dF, double *energy_exp,
         double *samples_out, double *energy_out, double *delta_F_out, double *bias_out)
{

    double sigma_2 = sigma * sigma;
    double gaussian_centers[num_lambda];
    for (int i = 0; i < num_lambda; i++)
    {
        gaussian_centers[i] = dE * ((double)i / (num_lambda - 1));
    }

#pragma omp parallel for
    for (int k = 0; k < num_SP; k++)
    {
        const gsl_rng_type *T;
        gsl_rng *r;

        gsl_rng_env_setup();
        T = gsl_rng_default;
        r = gsl_rng_alloc(T);
        unsigned long seed = k + 50;
        gsl_rng_set(r, seed);

        double x = gsl_ran_gaussian(r, 2);
        double y = gsl_ran_gaussian(r, 2);
        double x_vec[2];
        x_vec[0] = x;
        x_vec[1] = y;

        double min_energy = energy_function(x_vec);
        // for (int i = 0; i < 10; i++)
        // {
        //     double x_new = gsl_ran_gaussian(r, 2);
        //     double x_vec_new[2];
        //     x_vec_new[0] = x_new;
        //     x_vec_new[1] = x_vec[1];
        //     double energy_new = energy_function(x_vec_new);
        //     if (energy_new < min_energy)
        //     {
        //         min_energy = energy_new;
        //         x_vec[0] = x_vec_new[0];
        //     }
        // }

        double delta_F[num_lambda], delta_F_nominator_sum[num_lambda], delta_F_denominator_sum[num_lambda];
        for (int i = 0; i < num_lambda; i++)
        {
            delta_F_nominator_sum[i] = 0;
            delta_F_denominator_sum[i] = 0;
            delta_F[i] = energy_exp[i];
            if (delta_F[i] >= max_dF)
            {
                delta_F[i] = max_dF;
            }
        }

        double mean_accept = 0;

        for (int s = 0; s < num_samples; s++)
        {
            if (s % 10000 == 0)
            {
                printf("%d\n", s);
            }
            for (int j = 0; j < num_HMC; j++)
            {
                // sample momenta from Gaussian N(0,1)
                double px = gsl_ran_gaussian(r, 1.0);
                double py = gsl_ran_gaussian(r, 1.0);
                double kin_old = 0.5 * (px * px + py * py);

                // save old state
                double x_old = x_vec[0];
                double y_old = x_vec[1];
                double u_old = energy_function(x_vec) + bias_potential(x_vec, num_lambda, gaussian_centers, sigma_2, delta_F);

                // compute gradient at old state
                double grad_energy_vec[2];
                grad_energy(x_vec, grad_energy_vec);
                double grad_bias_vec[2];
                grad_bias(x_vec, num_lambda, gaussian_centers, sigma, sigma_2, delta_F, grad_bias_vec);

                // initial half-step for momentum
                px -= 0.5 * dt * (grad_energy_vec[0] + grad_bias_vec[0]);
                py -= 0.5 * dt * (grad_energy_vec[1] + grad_bias_vec[1]);

                // leapfrog integration
                for (int lf = 0; lf < num_dt; lf++)
                {
                    // full step position
                    x_vec[0] += dt * px;
                    x_vec[1] += dt * py;

                    // compute gradients at new position
                    grad_energy(x_vec, grad_energy_vec);
                    grad_bias(x_vec, num_lambda, gaussian_centers, sigma, sigma_2, delta_F, grad_bias_vec);

                    // full step momentum (except last iteration corrected below)
                    if (lf != num_dt - 1)
                    {
                        px -= dt * (grad_energy_vec[0] + grad_bias_vec[0]);
                        py -= dt * (grad_energy_vec[1] + grad_bias_vec[1]);
                    }
                }

                // final half-step for momentum
                px -= 0.5 * dt * (grad_energy_vec[0] + grad_bias_vec[0]);
                py -= 0.5 * dt * (grad_energy_vec[1] + grad_bias_vec[1]);

                // compute new Hamiltonian
                double kin_new = 0.5 * (px * px + py * py);
                double u_new = energy_function(x_vec) + bias_potential(x_vec, num_lambda, gaussian_centers, sigma_2, delta_F);

                double H_old = kin_old + u_old;
                double H_new = kin_new + u_new;

                // Metropolis accept/reject
                double dE = H_new - H_old;
                if (!(gsl_rng_uniform(r) < exp(-dE)))
                {
                    // reject → restore old state
                    x_vec[0] = x_old;
                    x_vec[1] = y_old;
                }
                else
                    mean_accept += 1;
            }
            bias_out[k * num_samples + s] = bias_potential(x_vec, num_lambda, gaussian_centers, sigma_2, delta_F);
            for (int l = 0; l < 2; l++)
            { // 2 is the number of dimensions
                samples_out[2 * num_samples * k + 2 * s + l] = x_vec[l];
            }

            energy_out[k * num_samples + s] = energy_function(x_vec);
            for (int l = 0; l < num_lambda; l++)
            {
                delta_F_out[num_lambda * num_samples * k + num_lambda * s + l] = delta_F[l];
            }

            for (int i = 0; i < num_lambda; i++)
            {
                update_delta_F(x_vec, i, num_lambda, sigma_2, gaussian_centers, delta_F_nominator_sum, delta_F_denominator_sum, delta_F, bias_out[k * num_samples + s], max_dF, s);
            }
        }
        mean_accept /= (num_samples * num_HMC);
        printf("%f\n", mean_accept);
    }
}