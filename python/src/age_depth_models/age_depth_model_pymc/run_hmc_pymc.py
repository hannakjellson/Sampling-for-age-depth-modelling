import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import pytensor.tensor as pt
import pytensor

pytensor.config.blas__ldflags = "-llapack -lblas -lcblas"
import arviz as az
import pandas as pd
from define_data_and_variables import get_data, get_hmc_config


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

        # Sample
        trace = pm.sample(
            int(config["num_samples"] / 2),
            tune=int(config["num_samples"] / 2),
            target_accept=0.9,
            return_inferencedata=True,
            chains=config["num_chains"],
        )

    sedimentation_rates = trace.posterior["sed_rates"].to_numpy()
    np.save(f"../../../../output/age_depth_pymc/samples_{config_str}.npy", sedimentation_rates)


if __name__ == "__main__":
    main()
