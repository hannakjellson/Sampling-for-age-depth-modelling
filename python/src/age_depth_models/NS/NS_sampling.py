import os
# BREAK THE COMPILER HANG: Disable aggressive folding of large windows
os.environ['XLA_FLAGS'] = "--xla_disable_hlo_passes=constant_folding,simplify-reduction"

import jax
from jaxns import NestedSampler, Model, Prior
import tensorflow_probability.substrates.jax as tfp
from jax import random, numpy as jnp
from define_data_and_variables import get_data, get_hmc_config
from jaxns import save_results
from jaxns.internals.mixed_precision import mp_policy

tfpd = tfp.distributions

def build_jaxns_model(config, data):
    # 1. SHIELD LARGE ARRAYS: stop_gradient prevents XLA from 
    # trying to 'pre-calculate' the 1,000,000 element window.
    c14_depths = jax.lax.stop_gradient(jnp.array(data["c14_depths"]))
    c14_ages = jax.lax.stop_gradient(jnp.array(data["c14_ages"]))
    c14_sigma = jax.lax.stop_gradient(jnp.array(data["c14_sigma"]))
    d18O_depths = jax.lax.stop_gradient(jnp.array(data["d18O_depths"]))
    d18O_vals = jax.lax.stop_gradient(jnp.array(data["d18O"]))
    d18O_sigma = jax.lax.stop_gradient(jnp.array(data["d18O_sigma"]))
    cs = jax.lax.stop_gradient(jnp.array(config["cs"]))
    
    ref_times = jax.lax.stop_gradient(jnp.array(data["d18O_reference_times"]))
    ref_vals = jax.lax.stop_gradient(jnp.array(data["d18O_reference"]))
    
    delta_c = config["delta_c"]
    theta = data["theta"]

    @jax.jit
    def log_likelihood(sed_rates):
        # 2. OPTIMIZED CUMSUM: Avoid concatenate inside JIT where possible
        # We use a zero-padded array and set values to avoid graph fragmentation
        cumsum = jnp.zeros(len(sed_rates) + 1).at[1:].set(jnp.cumsum(sed_rates))
        
        # C14 calculation
        indices_c14 = jnp.searchsorted(cs, c14_depths, side="right") - 1
        expected_ages = (theta - cumsum[indices_c14] * delta_c 
                         - sed_rates[indices_c14] * (c14_depths - cs[indices_c14]))
        
        l1 = jnp.sum(jax.scipy.stats.norm.logpdf(c14_ages, loc=expected_ages, scale=c14_sigma))

        # D18O calculation
        indices_d18 = jnp.searchsorted(cs, d18O_depths, side="right") - 1
        d18_times = (theta - cumsum[indices_d18] * delta_c 
                     - sed_rates[indices_d18] * (d18O_depths - cs[indices_d18]))
        
        # Interpolation with safe clipping
        interp_indices = jnp.clip(jnp.searchsorted(ref_times, d18_times, side="right") - 1, 0, len(ref_times) - 2)
        t0, t1 = ref_times[interp_indices], ref_times[interp_indices + 1]
        v0, v1 = ref_vals[interp_indices], ref_vals[interp_indices + 1]
        
        # Vectorized interpolation
        expected_D18O = v0 + (v1 - v0) / (t1 - t0) * (d18_times - t0)
        
        l2 = jnp.sum(jax.scipy.stats.norm.logpdf(d18O_vals, loc=expected_D18O, scale=d18O_sigma))
        
        return l1 + l2

    def prior_model():
        lamda = yield Prior(
            tfpd.Gamma(
                concentration=jnp.full((config["N"],), config["a"], mp_policy.measure_dtype),
                rate=jnp.full((config["N"],), config["b"], mp_policy.measure_dtype)
            ),
            name="lamda"
        )
        return lamda

    return Model(prior_model=prior_model, log_likelihood=log_likelihood)

def run_discovery(key, config, data):
    model = build_jaxns_model(config, data)
    
    # 10,000 points is great for 50D multimodal, but let's monitor VRAM
    ns = NestedSampler(model=model, verbose=True, num_live_points=10000)

    # Use block_until_ready to ensure the compilation warning doesn't hide errors
    print("Compiling model... this may take up to 2 minutes for 10,000 chains.")
    nsj = jax.jit(ns.__call__)
    
    termination_reason, state = nsj(key)
    # Force GPU to finish before Python continues
    jax.block_until_ready(state)
    
    return ns.to_results(termination_reason=termination_reason, state=state)

def main():
    # Force float64 if your model needs the precision, otherwise float32 is 2-4x faster
    # jax.config.update("jax_enable_x64", True) 
    
    key = jax.random.PRNGKey(42)
    config, _ = get_hmc_config()
    data = get_data()
    
    results = run_discovery(key, config, data)
    save_results(results, "results.json")
    print("Sampling complete. Results saved.")

if __name__ == "__main__":
    main()