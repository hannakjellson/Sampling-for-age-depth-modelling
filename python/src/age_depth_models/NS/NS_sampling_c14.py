import os
# BREAK THE COMPILER HANG: Disable aggressive folding of large windows
os.environ['XLA_FLAGS'] = "--xla_disable_hlo_passes=constant_folding,simplify-reduction"

import jax
from jaxns import NestedSampler, Model, Prior
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
from jax import random, numpy as jnp
import json
from define_data_and_variables import get_data, get_NS_config, hash_configs, make_dumpable
from jaxns import save_results
from jaxns.internals.mixed_precision import mp_policy

tfpd = tfp.distributions

def build_jaxns_model(config, data):
    # 1. SHIELD LARGE ARRAYS: stop_gradient prevents XLA from 
    # trying to 'pre-calculate' the 1,000,000 element window.
    c14_depths = jax.lax.stop_gradient(jnp.array(data["c14_depths"]))
    c14_ages = jax.lax.stop_gradient(jnp.array(data["c14_ages"]))
    c14_sigma = jax.lax.stop_gradient(jnp.array(data["c14_sigma"]))
    cs = jax.lax.stop_gradient(jnp.array(config["cs"]))
    
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
        
        return l1

    def prior_model():

        alpha = yield Prior(
            tfpd.Gamma(
                concentration=jnp.full((config["N"],), config["a"], mp_policy.measure_dtype),
                rate=jnp.full((config["N"],), config["b"], mp_policy.measure_dtype)
            ),
            name="sed_rates"
        )

        return alpha

    return Model(prior_model=prior_model, log_likelihood=log_likelihood)

def run_discovery(key, config, data):
    model = build_jaxns_model(config, data)
    
    # 10,000 points is great for 50D multimodal, but let's monitor VRAM
    ns = NestedSampler(model=model, verbose=True, num_live_points=config["num_points"])

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
    
    config = get_NS_config(c14 = True)
    data = get_data()

    hash = hash_configs(config, data)
    output_dir = f"output/{hash}"
    os.makedirs(output_dir, exist_ok=True)
    key = jax.random.PRNGKey(config["sd"])
    
    with open(os.path.join(output_dir, "c14_config.json"), "w") as f:
        dump_config = make_dumpable(config)
        json.dump(dump_config, f, indent=2, skipkeys=True)

    with open(os.path.join(output_dir, "data.json"), "w") as f:
        dump_data = make_dumpable(data)
        json.dump(dump_data, f, indent=2, skipkeys=True)
        
    results = run_discovery(key, config, data)
    save_results(results, f"{output_dir}/results_c14.json")
    print("Sampling complete. Results saved.")

if __name__ == "__main__":
    main()