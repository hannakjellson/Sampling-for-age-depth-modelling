import pymc as pm
import numpy as np
import jax.numpy as jnp
import pytensor.tensor as pt
import pytensor
pytensor.config.blas__ldflags = "-llapack -lblas -lcblas"
import matplotlib.pyplot as plt

from define_data_and_variables import data, opes_config, opes_hash

from pymc.sampling import jax as pmj
from inference_loop import inference_loop
pmj._blackjax_inference_loop = inference_loop
import os


def mean_ages(sedimentation_rates, data_type="c14"):
    if data_type == "c14":
        depths = data["c14d"]
    else:
        depths = data["d18od"]
    indices = pt.searchsorted(data["cs"], depths, side="right") - 1
    cumsum = pt.concatenate([[0], pt.cumsum(sedimentation_rates)])
    CS_pt = pt.constant(data["cs"])
    return (
        data["th"]
        - cumsum[indices] * data["dc"]
        - sedimentation_rates[indices] * (depths - CS_pt[indices])
    )


def interpolate(sed_rates):
    D18O_times = mean_ages(sed_rates, data_type="D18O")

    # Get indices of the left side of interpolation intervals
    indices = (
        pt.searchsorted(data["d18ort"], D18O_times, side="right") - 1
    )
    indices = pt.clip(indices, 0, len(data["d18ort"]) - 2)

    t0 = pt.constant(data["d18ort"])[indices]
    t1 = pt.constant(data["d18ort"])[indices + 1]
    v0 = pt.constant(data["d18or"])[indices]
    v1 = pt.constant(data["d18or"])[indices + 1]

    slope = (v1 - v0) / (t1 - t0)
    D18O_interp = v0 + slope * (D18O_times - t0)

    return D18O_interp


def main():
    with pm.Model() as model:
        # Priors
        z = pm.Normal("z", mu=0, sigma=1, shape=data["N"])
        sedimentation_rates = pm.Deterministic(
            "sed_rates",
            pm.math.exp(z * data["ps"] + data["pm"])
        )

        # Observations likelihood
        expected_ages = mean_ages(sedimentation_rates)
        c14_obs = pm.Normal(
            "c14_obs",
            mu=expected_ages,
            sigma=data["c14s"],
            observed=data["c14"],
        )

        expected_D18O = interpolate(sedimentation_rates)
        D18O_obs = pm.Normal(
            "d18O_obs",
            mu=expected_D18O,
            sigma=data["d18os"],
            observed=data["d18o"],
        )
        trace = pmj.sample_blackjax_nuts(
            int(opes_config["hmcc"]["ns"]),
            tune=int(opes_config["hmcc"]["ns"]),
            # target_accept=0.95,
            chains=opes_config["hmcc"]["nch"],
            progressbar=True,
            random_seed = opes_config["hmcc"]["sd"],
        )

    sedimentation_rates = trace.posterior["sed_rates"].to_numpy()

    sedimentation_rates = trace.posterior["sed_rates"].to_numpy()
    output_dir = f"output/{data["dn"]}/{opes_hash}"
    os.makedirs(output_dir, exist_ok=True)
    trace.to_netcdf(f"{output_dir}/results.nc")


if __name__ == "__main__":
    main()
