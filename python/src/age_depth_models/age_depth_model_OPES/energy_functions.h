#ifndef ENERGY_FUNCTIONS_H
#define ENERGY_FUNCTIONS_H
#define MAX_NBR_PC 2

int binary_search(const double *arr, int n, double target);

void expected_ages(
    int N, double delta_c, const double *cs, double theta, int num_depths,
    const double *depths, const double *sed_rates, int *indices, double *ages_out);

double energy_function(
    int N, double delta_c, const double *cs, double a, double b, double theta, double beta,
    int num_c14_depths, int num_D18O_depths, int num_D18O_reference_times, const double *c14_ages, const double *c14_depths,
    const double *c14_sigma, int *c14_depth_indices, double *inv_c14_var, double *c14_expected_ages, const double *D18O, const double *D18O_depths, const double *D18O_sigma,
    int *D18O_depth_indices, double *inv_D18O_var, double *expected_D18O_ages, const double *D18O_reference, const double *D18O_reference_times, const double *sed_rates);

void grad_energy_function(
    int N, double delta_c, const double *cs, double a, double b, double theta, double beta,
    int num_c14_depths, int num_D18O_depths, int num_D18O_reference_times, const double *c14_ages, const double *c14_depths,
    const double *c14_sigma, int *c14_indices, double *inv_c14_var, double *expected_c14_ages, const double *D18O, const double *D18O_depths, const double *D18O_sigma,
    int *D18O_indices, double *inv_D18O_var, double *expected_D18O_ages, const double *D18O_reference, const double *D18O_reference_times, const double *sed_rates, double *gradient);

double get_CV_point(int N, double *variables, const double *pc, const double *sp_mean);

double bias_potential(int num_pcs, double *CV_point, int num_lambda, int num_temps, double *gaussian_centers, double *betas, double beta0, double energy, double sigma, double sigma_2, double *delta_F, int start_index[MAX_NBR_PC], int end_index[MAX_NBR_PC], bool umbrella, bool temp);

void grad_bias(int N, double delta_c, int num_pcs, const double *pcs, double *CV_point, double *variables, int num_lambda, int num_temps, double *gaussian_centers, double *betas, double beta0, double energy, double *gradient, double sigma, double sigma_2, double *delta_F, int start_index[MAX_NBR_PC], int end_index[MAX_NBR_PC], bool umbrella, bool temp, double *bias_gradient);

void update_delta_F(int num_pcs, double *CV_point, int num_lambda, int num_temps, double sigma_2, double dE, double *gaussian_centers, double *betas, double beta0, double energy, double *delta_F_nominator_sum, double delta_F_denominator_sum, double *max_delta_F_nominator_sum_term, double max_delta_F_denominator_sum_term, double *delta_F, double potential, bool umbrella, bool temp, int sample, int dfs);

void stoch_grad_energy_function(
    int N, int num_D18O_indices_stoch, int *D18O_indices_stoch, double delta_c, const double *cs, double a, double b, double theta, double beta,
    int num_c14_depths, int num_D18O_depths, int num_D18O_reference_times, const double *c14_ages, const double *c14_depths,
    const double *c14_sigma, int *c14_indices, double *inv_c14_var, double *c14_expected_ages, const double *D18O, const double *D18O_depths, const double *D18O_sigma,
    int *D18O_indices, double *inv_D18O_var, double *D18O_expected_ages, const double *D18O_reference, const double *D18O_reference_times, const double *sed_rates, double *gradient);
#endif