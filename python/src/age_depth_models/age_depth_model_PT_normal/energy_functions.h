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
    Data *d, double *c14_expected_ages, double *expected_D18O_ages, double *nl_sed_rates, double *d18o_energy, double beta);

void grad_energy_function(Data *d, int *c14_indices, double *expected_c14_ages, int *D18O_indices, double *expected_D18O_ages, double *norm_log_sed_rates, double *gradient, double beta);

void stoch_grad_energy_function(
    Data *d, int num_D18O_indices_stoch, int *D18O_indices_stoch, int *c14_indices, double *c14_expected_ages, int *D18O_indices, double *D18O_expected_ages, double *nl_sed_rates, double *gradient);
#endif