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

double energy_function(
    Data *d, double *c14_expected_ages, double *nl_sed_rates);

void grad_energy_function(Data *d, int *c14_indices, double *expected_c14_ages, double *norm_log_sed_rates, double *gradient);

#endif