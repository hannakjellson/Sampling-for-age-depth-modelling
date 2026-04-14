#ifndef ENERGY_FUNCTIONS_H
#define ENERGY_FUNCTIONS_H
#define MAX_NBR_PC 2

#include "configs.h"

int binary_search(double *arr, int n, double target);

void expected_ages(
    int N, double dc, double *cs, double th, int nd,
    double *depths, double *sed_rates, int *indices, double *ages_out);

double fprior(int N, double *nl_sed_rates);

double c14_cond(Data *d, double *c14_expected_ages);

double d18o_cond(Data *d, double *d18o_expected_ages);

double energy_function(
    Data *d, double *c14_expected_ages, double *expected_D18O_ages, double *nl_sed_rates, double *d18o_energy);

void grad_energy_function(Data *d, int *c14_indices, double *expected_c14_ages, int *D18O_indices, double *expected_D18O_ages, double *norm_log_sed_rates, double *gradient, double *d18o_gradient);

double bias_potential(int num_temps, double *betas, double energy, double *delta_F);

void grad_bias(int N, int num_temps, double *betas, double *variables, double energy, double *gradient, double *delta_F, double *bias_gradient);

void update_delta_F(int num_temps, double *betas, double *delta_F_nominator_sum, double *delta_F_denominator_sum, double *delta_F, double *bias_out, double *energy_out, double *df_out, int i, int j, int num_samples, double min_dfd, double *max_dfn);

void stoch_grad_energy_function(
    Data *d, int num_D18O_indices_stoch, int *D18O_indices_stoch, int *c14_indices, double *c14_expected_ages, int *D18O_indices, double *D18O_expected_ages, double *nl_sed_rates, double *gradient);
#endif