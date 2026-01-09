#TODO: Continue reading on why it might not work with only opes and that it potentially should be seen as a local explorer.

import jax
from jaxns import NestedSampler, Model, Prior, summary
import numpy as np
import pylab as plt
import tensorflow_probability.substrates.jax as tfp
from jax import random, numpy as jnp
from define_data_and_variables import get_data, get_hmc_config
from jaxns import resample, save_results

from jaxns.internals.mixed_precision import mp_policy
from jaxns import Prior, Model

tfpd = tfp.distributions


# 1. Define the Physics/Logic (JAX version of your mean_ages)
def jax_mean_ages(sed_rates, config, data, data_type="c14"):
    depths = data["c14_depths"] if data_type == "c14" else data["d18O_depths"]
    
    # JAX equivalent of searchsorted
    indices = jnp.searchsorted(config["cs"], depths, side="right") - 1
    
    cumsum = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(sed_rates)])
    cs = jnp.array(config["cs"])
    
    return (
        data["theta"]
        - cumsum[indices] * config["delta_c"]
        - sed_rates[indices] * (depths - cs[indices])
    )

def jax_interpolate(sed_rates, config, data):
    D18O_times = jax_mean_ages(sed_rates, config, data, data_type="D18O")
    ref_times = jnp.array(data["d18O_reference_times"])
    ref_vals = jnp.array(data["d18O_reference"])

    indices = jnp.searchsorted(ref_times, D18O_times, side="right") - 1
    indices = jnp.clip(indices, 0, len(ref_times) - 2)

    t0, t1 = ref_times[indices], ref_times[indices + 1]
    v0, v1 = ref_vals[indices], ref_vals[indices + 1]

    slope = (v1 - v0) / (t1 - t0)
    return v0 + slope * (D18O_times - t0)

# Optimized build function
def build_jaxns_model(config, data):
    # 1. Pre-bake everything into JNP arrays to avoid host-to-device transfers
    # and to ensure JIT treats them as constants.
    c14_depths = jnp.array(data["c14_depths"])
    c14_ages = jnp.array(data["c14_ages"])
    c14_sigma = jnp.array(data["c14_sigma"])
    d18O_depths = jnp.array(data["d18O_depths"])
    d18O_vals = jnp.array(data["d18O"])
    d18O_sigma = jnp.array(data["d18O_sigma"])
    cs = jnp.array(config["cs"])
    delta_c = config["delta_c"]
    theta = data["theta"]
    
    ref_times = jnp.array(data["d18O_reference_times"])
    ref_vals = jnp.array(data["d18O_reference"])

    @jax.jit
    def log_likelihood(sed_rates):
        # C14 calculation
        indices_c14 = jnp.searchsorted(cs, c14_depths, side="right") - 1
        cumsum = jnp.cumsum(jnp.concatenate([jnp.array([0.0]), sed_rates]))
        
        expected_ages = (theta - cumsum[indices_c14] * delta_c 
                         - sed_rates[indices_c14] * (c14_depths - cs[indices_c14]))
        
        l1 = jnp.sum(jax.scipy.stats.norm.logpdf(c14_ages, loc=expected_ages, scale=c14_sigma))

        # D18O calculation (Inlined to avoid function call overhead)
        indices_d18 = jnp.searchsorted(cs, d18O_depths, side="right") - 1
        d18_times = (theta - cumsum[indices_d18] * delta_c 
                     - sed_rates[indices_d18] * (d18O_depths - cs[indices_d18]))
        
        # Interpolation
        interp_indices = jnp.clip(jnp.searchsorted(ref_times, d18_times, side="right") - 1, 0, len(ref_times) - 2)
        t0, t1 = ref_times[interp_indices], ref_times[interp_indices + 1]
        v0, v1 = ref_vals[interp_indices], ref_vals[interp_indices + 1]
        expected_D18O = v0 + (v1 - v0) / (t1 - t0) * (d18_times - t0)
        
        l2 = jnp.sum(jax.scipy.stats.norm.logpdf(d18O_vals, loc=expected_D18O, scale=d18O_sigma))
        
        return l1 + l2

    def prior_model():
        # Use config values directly as floats/arrays
        lamda = yield Prior(
            tfpd.Gamma(
                concentration=jnp.full((config["N"],), config["a"], mp_policy.measure_dtype),
                rate=jnp.full((config["N"],), config["b"], mp_policy.measure_dtype)
            ),
            name="lamda"
        )
        return lamda

    return Model(prior_model=prior_model, log_likelihood=log_likelihood)

# 3. Execute Global Discovery
def run_discovery(key, config, data):
    model = build_jaxns_model(config, data)
    
    # SVD-based clustering is built-in to handle multimodality
    ns = NestedSampler(model=model, verbose=True, init_efficiency_threshold=0.)

    nsj = jax.jit(ns.__call__)
    
    termination_reason, state = nsj(key)
    results = ns.to_results(termination_reason= termination_reason, state = state)
    
    # Access the discovered modes
    # results.samples contains the points clustered by their probability islands
    return results

def main():
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    config, config_str = get_hmc_config()
    data = get_data()
    results = run_discovery(subkey, config, data)
    save_results(results, "results.json")

if __name__ == "__main__":
    main()
