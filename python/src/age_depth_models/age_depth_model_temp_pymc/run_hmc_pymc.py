import pymc as pm
import numpy as np
import jax.numpy as jnp
import pytensor.tensor as pt
import pytensor
pytensor.config.blas__ldflags = "-llapack -lblas -lcblas"

from define_data_and_variables import get_data, get_hmc_config

from pymc.sampling import jax as pmj
from inference_loop import inference_loop
pmj._blackjax_inference_loop = inference_loop


def mean_ages(sedimentation_rates, config, data, data_type="c14"):
    if data_type == "c14":
        depths = data["c14_depths"]
    else:
        depths = data["d18O_depths"]
    indices = pt.searchsorted(config["cs"], depths, side="right") - 1
    cumsum = pt.concatenate([[0], pt.cumsum(sedimentation_rates)])
    CS_pt = pt.constant(config["cs"])
    return (
        data["theta"]
        - cumsum[indices] * config["delta_c"]
        - sedimentation_rates[indices] * (depths - CS_pt[indices])
    )


def interpolate(sed_rates, config, data):
    D18O_times = mean_ages(sed_rates, config, data, data_type="D18O")

    # Get indices of the left side of interpolation intervals
    indices = (
        pt.searchsorted(data["d18O_reference_times"], D18O_times, side="right") - 1
    )
    indices = pt.clip(indices, 0, len(data["d18O_reference_times"]) - 2)

    t0 = pt.constant(data["d18O_reference_times"])[indices]
    t1 = pt.constant(data["d18O_reference_times"])[indices + 1]
    v0 = pt.constant(data["d18O_reference"])[indices]
    v1 = pt.constant(data["d18O_reference"])[indices + 1]

    slope = (v1 - v0) / (t1 - t0)
    D18O_interp = v0 + slope * (D18O_times - t0)

    return D18O_interp


def main():
    config, config_str = get_hmc_config()
    data = get_data()

    with pm.Model() as model:
        # Priors
        sedimentation_rates = pm.Gamma(
            "sed_rates", alpha=config["a"], beta=config["b"], shape=config["N"]
        )
        # theta = pm.Normal("theta", mu=theta_mean, sigma=sigma_theta)

        # Observations likelihood
        expected_ages = mean_ages(sedimentation_rates, config, data)
        c14_obs = pm.Normal(
            "c14_obs",
            mu=expected_ages,
            sigma=data["c14_sigma"],
            observed=data["c14_ages"],
        )

        mask = ~np.isnan(data["d18O"])
        expected_D18O = interpolate(sedimentation_rates, config, data)
        D18O_obs = pm.Normal(
            "d18O_obs",
            mu=expected_D18O,
            sigma=data["d18O_sigma"],
            observed=data["d18O"],
        )

        starting_points = np.load("C:/Users/hanna/Desktop/PhD/Bacon/output/dayu06/N50_H100_dc2_dt0.001_ns100_ndt100_nHMC1_a1.5_b0.21_nlsp100_mi1000_gl1e-05_hmc_sd42_bt1_adt1e-05/start_samples.npy")
        starting_energies = np.load("C:/Users/hanna/Desktop/PhD/Bacon/output/dayu06/N50_H100_dc2_dt0.001_ns100_ndt100_nHMC1_a1.5_b0.21_nlsp100_mi1000_gl1e-05_hmc_sd42_bt1_adt1e-05/start_energies.npy")
        sp_index = np.argmin(starting_energies[:config["num_chains"], :], axis = 1)
        sp = [starting_points[i, sp_index_i, :] for i, sp_index_i in enumerate(sp_index)]
        print(sp)
        initvals = [{'sed_rates' : np.array(init_val)} for init_val in sp]
        # Sample
        trace = pmj.sample_blackjax_nuts(
            int(config["num_samples"]),
            tune=0,
            target_accept=0.9,
            chains=config["num_chains"],
            progressbar=True,
            random_seed = 42,
            initvals=initvals,
        )

    sedimentation_rates = trace.posterior["sed_rates"].to_numpy()
    trace.to_netcdf(f"../../../../output/age_depth_temp_pymc/samples_{config_str}.nc")
    # np.save(f"../../../../output/age_depth_temp_pymc/samples_{config_str}.npy", sedimentation_rates)


if __name__ == "__main__":
    main()
