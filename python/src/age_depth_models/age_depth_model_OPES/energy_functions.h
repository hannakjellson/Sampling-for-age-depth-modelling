#ifndef ENERGY_FUNCTIONS_H
#define ENERGY_FUNCTIONS_H

int binary_search(const double *arr, int n, double target);

void expected_ages(
    int N, double delta_c, const double *cs, double theta, int num_depths,
    const double *depths, const double *sed_rates, int *indices, double *ages_out);

double energy_function(
    int N, double delta_c, const double *cs, double a, double b, double theta,
    int num_c14_depths, int num_D18O_depths, int num_D18O_reference_times, const double *c14_ages, const double *c14_depths,
    const double *c14_sigma, int *c14_depth_indices, double *inv_c14_var, double *c14_expected_ages, const double *D18O, const double *D18O_depths, const double *D18O_sigma,
    int *D18O_depth_indices, double *inv_D18O_var, double *expected_D18O_ages, const double *D18O_reference, const double *D18O_reference_times, const double *sed_rates);

void grad_energy_function(
    int N, double delta_c, const double *cs, double a, double b, double theta,
    int num_c14_depths, int num_D18O_depths, int num_D18O_reference_times, const double *c14_ages, const double *c14_depths,
    const double *c14_sigma, int *c14_indices, double *inv_c14_var, double *expected_c14_ages, const double *D18O, const double *D18O_depths, const double *D18O_sigma,
    int *D18O_indices, double *inv_D18O_var, double *expected_D18O_ages, const double *D18O_reference, const double *D18O_reference_times, const double *sed_rates, double *gradient);

double get_CV_point(int N, double theta, double *variables, int problem_index, double delta_c);

double bias_potential(double CV_point, int num_lambda, int num_temps, double *gaussian_centers, double *betas, double beta0, double energy, double sigma, double sigma_2, double *delta_F, int start_index, int end_index, bool umbrella, bool temp);

void grad_bias(int N, double delta_c, int problem_index, double CV_point, double *variables, int num_lambda, int num_temps, double *gaussian_centers, double *betas, double beta0, double energy, double *gradient, double sigma, double sigma_2, double *delta_F, int start_index, int end_index, bool umbrella, bool temp, double *bias_gradient);

void update_delta_F(double CV_point, int num_lambda, int num_temps, double sigma_2, double dE, double *gaussian_centers, double *betas, double beta0, double energy, double *delta_F_nominator_sum, double delta_F_denominator_sum, double *delta_F, double potential, bool umbrella, bool temp);

void stoch_grad_energy_function(
    int N, int num_D18O_indices_stoch, int *D18O_indices_stoch, double delta_c, const double *cs, double a, double b, double theta,
    int num_c14_depths, int num_D18O_depths, int num_D18O_reference_times, const double *c14_ages, const double *c14_depths,
    const double *c14_sigma, int *c14_indices, double *inv_c14_var, double *c14_expected_ages, const double *D18O, const double *D18O_depths, const double *D18O_sigma,
    int *D18O_indices, double *inv_D18O_var, double *D18O_expected_ages, const double *D18O_reference, const double *D18O_reference_times, const double *sed_rates, double *gradient);

double bias_potential_r(double CV_point, double *bias_centers, double *bias_heights, double *bias_widths, int bias_count, double *kernel_weights, double gamma, double sum_weights, double Z, double DeltaE);

void grad_bias_r(int N, double delta_c, int problem_index, double CV_point, double *variables, double *bias_centers, double *bias_heights, double *bias_widths, int bias_count, double *kernel_weights, double gamma, double sum_weights, double Z, double dE, double *gradient);

void deposit_gaussian(double CV_point, double width, double *bias_centers, double *bias_heights, double *bias_widths, double *kernel_weights, double current_weight, double *sum_squared_weights, int *bias_count, double distance_threshold, int max_bias);

double compute_Zn(double *bias_centers, double *bias_heights, double *bias_widths, int bias_count, double *kernel_weights, double gamma, double sum_weights);

#endif