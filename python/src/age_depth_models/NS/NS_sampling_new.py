import os
# BREAK THE COMPILER HANG: Disable aggressive folding of large windows
os.environ['XLA_FLAGS'] = "--xla_disable_hlo_passes=constant_folding,simplify-reduction"

import jax
from jaxns import NestedSampler, Model, Prior
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
from jax import random, numpy as jnp
from define_data_and_variables import get_data, get_NS_config, hash_configs
from jaxns import save_results
from jaxns.internals.mixed_precision import mp_policy
from jaxns import resample
from jaxns.utils import load_results

def build_jaxns_model(config, data):
    # 1. SHIELD LARGE ARRAYS: stop_gradient prevents XLA from 
    # trying to 'pre-calculate' the 1,000,000 element window.
    c14_depths = jnp.array(data["c14d"])
    c14_ages = jnp.array(data["c14"])
    c14_sigma = jnp.array(data["c14s"])
    d18O_depths = jnp.array(data["d18od"])
    d18O_vals = jnp.array(data["d18o"])
    d18O_sigma = jnp.array(data["d18os"])
    cs = jnp.array(data["cs"])
    
    ref_times = jnp.array(data["d18ort"])
    ref_vals = jnp.array(data["d18or"])
    
    delta_c = data["dc"]
    theta = data["th"]

    
    # flattened = jnp.load(r"C:\Users\hanna\Desktop\PhD\Bacon\python\src\age_depth_models\age_depth_model_c14_normal\output\68831df8d5\samples.npy")[0, 1000:, :]

    # key = jax.random.PRNGKey(config["sd"])
    # key, subkey = jax.random.split(key)

    # sigma_x = jnp.cov(flattened, rowvar=False)
    # mu_x = jnp.mean(flattened, axis = 0)

    # dist = tfd.MultivariateNormalFullCovariance(loc=mu_x, covariance_matrix=sigma_x)

    prior_dist = tfd.LogNormal(
        loc=jnp.full((data["N"],), data["pm"], mp_policy.measure_dtype),
        scale=jnp.full((data["N"],), data["ps"], mp_policy.measure_dtype)
    )
    
    @jax.jit
    def log_likelihood(sed_rates):
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
        
        expected_D18O = jnp.interp(d18_times, ref_times, ref_vals)
        
        l2 = jnp.sum(jax.scipy.stats.norm.logpdf(d18O_vals, loc=expected_D18O, scale=d18O_sigma))

        return l1 + l2
    
    def prior_model():
        alpha = yield Prior(
            prior_dist, 
            name='sed_rates'
        )

        return alpha

    return Model(prior_model=prior_model, log_likelihood=log_likelihood)

def run_discovery(key, config, data):
    model = build_jaxns_model(config, data)
    
    # 10,000 points is great for 50D multimodal, but let's monitor VRAM
    ns = NestedSampler(model=model, verbose=True, num_live_points=config["np"], gradient_guided=config["gg"], difficult_model = config["dm"], devices = jax.devices('cpu'))
    
    termination_reason, state = ns(key)
    
    return ns.to_results(termination_reason=termination_reason, state=state)

def main():
    # Force float64 if your model needs the precision, otherwise float32 is 2-4x faster
    # jax.config.update("jax_enable_x64", True) 
    
    config = get_NS_config()
    data = get_data()
    hash = hash_configs(config, data)
    sd_output_dir = f"output/sd_{config["sd"]}"

    os.makedirs(f"{sd_output_dir}/{hash}", exist_ok=True)
    key = jax.random.PRNGKey(config["sd"])
    
    results = run_discovery(key, config, data)
    save_results(results, f"{sd_output_dir}/{hash}/results_d18o.json")
    print("Sampling complete. Results saved.")

if __name__ == "__main__":
    main()