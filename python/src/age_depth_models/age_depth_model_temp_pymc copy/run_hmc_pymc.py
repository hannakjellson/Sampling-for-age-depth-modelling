import pymc as pm
import numpy as np
import jax.numpy as jnp
import pytensor.tensor as pt
import pytensor
pytensor.config.blas__ldflags = "-llapack -lblas -lcblas"

from define_data_and_variables import opes_config, data

from pymc.sampling import jax as pmj
from inference_loop import inference_loop
pmj._blackjax_inference_loop = inference_loop
from pymc.logprob import logp
import pymc as pm


def mean_ages(sedimentation_rates, depths, indices, data):
    cumsum = pt.concatenate([[0], pt.cumsum(sedimentation_rates)])
    CS_pt = pt.constant(data["cs"])
    return (
        data["th"]
        - cumsum[indices] * data["dc"]
        - sedimentation_rates[indices] * (depths - CS_pt[indices])
    )


def interpolate(sed_rates, config, depths, indices, data):

    # indices = pt.clip(indices, 0, len(data["d18O_reference_times"]) - 2)

    t0 = pt.constant(data["d18ort"])[indices]
    t1 = pt.constant(data["d18ort"])[indices + 1]
    v0 = pt.constant(data["d18or"])[indices]
    v1 = pt.constant(data["d18or"])[indices + 1]

    slope = (v1 - v0) / (t1 - t0)
    D18O_interp = v0 + slope * (d18O_times - t0)

    return D18O_interp


def main():
    c14_indices = pt.searchsorted(data["cs"], data["c14d"], side="right") - 1
    d18o_indices = pt.searchsorted(data["cs"], data["d18od"], side="right") - 1


    with pm.Model() as model:
        # Priors
        sedimentation_rates = pm.LogNormal(
            "sed_rates", mu=data["pm"], sigma=data["ps"], shape=data["N"], default_transform=None,
        )
        # theta = pm.Normal("theta", mu=theta_mean, sigma=sigma_theta)

        # Observations likelihood
        expected_ages = mean_ages(sedimentation_rates, data["c14d"], c14_indices, data)
        c14_obs = pm.Normal(
            "c14_obs",
            mu=expected_ages,
            sigma=data["c14s"],
            observed=data["c14"],
        )

        mask = ~np.isnan(data["d18o"])
        d18O_times = mean_ages(sedimentation_rates, data["d18od"], d18o_indices, data)

        # Get indices of the left side of interpolation intervals
        indices = (
            pt.searchsorted(data["d18O_reference_times"], d18O_times, side="right") - 1
        )
        np.interp(z, depth, norm_age[:, j])
        expected_D18O = interpolate(sedimentation_rates, opes_config, data)
        D18O_obs = pm.Normal(
            "d18O_obs",
            mu=expected_D18O,
            sigma=data["d18os"],
            observed=data["d18o"],
        )

    logp_c14 = pt.sum(logp(pm.Normal.dist(mu=expected_ages, sigma=data["c14s"]), data["c14"]))
    logp_d18o = pt.sum(logp(pm.Normal.dist(mu=expected_D18O, sigma=data["d18os"]), data["d18o"]))

    with model:
        # Sample
        trace = pmj.sample_blackjax_nuts(
            int(opes_config["ns"]),
            tune=int(opes_config["ns"]),
            chains=opes_config["nch"],
            progressbar=True,
            random_seed = 32,
        )

    sedimentation_rates = trace.posterior["sed_rates"].to_numpy()
    trace.to_netcdf(f"../../../../output/age_depth_temp_pymc copy/output/samples.nc")
    # np.save(f"../../../../output/age_depth_temp_pymc/samples_{config_str}.npy", sedimentation_rates)


if __name__ == "__main__":
    main()
