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
    fig, axs = plt.subplots(1, 4, figsize=(10, 8))

    print(data["Tiout_radiometric"])
    print(data["Tiout_radiometric_depths"])
    # Top-left
    axs[0].plot(data["Tiout_dc13"], data["Tiout_dc13_depths"], '*')
    axs[0].set_yticks(data["Tiout_radiometric_depths"])
    axs[0].set_yticklabels([str(val) for val in data["Tiout_radiometric"]])
    axs[0].set_title("Plot 1")
    axs[0].invert_yaxis()

    # Top-right
    axs[1].plot(data["Oued_Sdas_dc13"], data["Oued_Sdas_dc13_depths"], '*')
    axs[1].set_yticks(data["Oued_Sdas_radiometric_depths"])
    axs[1].set_yticklabels([str(val) for val in data["Oued_Sdas_radiometric"]])
    axs[1].set_title("Plot 2")
    axs[1].invert_yaxis()

    # Bottom-left
    axs[2].plot(data["Talat_Nyssi_dc13"], data["Talat_Nyssi_dc13_depths"], '*')
    axs[2].set_yticks(data["Talat_Nyssi_radiometric_depths"])
    axs[2].set_yticklabels([str(val) for val in data["Talat_Nyssi_radiometric"]])
    axs[2].set_title("Plot 3")
    axs[2].invert_yaxis()

    # Bottom-right
    axs[3].plot(data["Sukharikha_dc13"], data["Sukharikha_dc13_depths"], '*')
    axs[3].set_title("Plot 4")
    axs[3].invert_yaxis()

    plt.tight_layout()
    plt.show()
    exit()

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
    np.save(f"../../../../output/age_depth_pymc_tril/samples_{config_str}.npy", sedimentation_rates)


if __name__ == "__main__":
    main()
