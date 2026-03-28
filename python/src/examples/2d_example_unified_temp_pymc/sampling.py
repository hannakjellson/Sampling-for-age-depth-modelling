from pymc.sampling import jax as pmj

from inference_loop import inference_loop

import jax.numpy as jnp
import numpy as np

from model import myModel

# import arviz as az

# Monkey-patch the original inference loop with our own
pmj._blackjax_inference_loop = inference_loop

n_chains = 4
# initvals = [{'x' : np.array([1.467282, -0.176202])}, {'x' : np.array([1.294953, -0.848773])}, {'x' : np.array([-1.819696, 1.283960])}, {'x' : np.array([-0.302247, 1.623959])}]

initvals = [{'x': np.array([np.random.normal(0, 2), np.random.normal(0, 2)])} for _ in range(n_chains)]

with myModel:
    idata = pmj.sample_blackjax_nuts(
        100_000,
        # 1_000_000,      # Hanna's value
        tune=0,
        chains=n_chains,
        progressbar=True,
        # target_accept=0.7,
        random_seed=1301512,
        initvals=initvals,
        # jitter=False,
    )
idata.to_netcdf('./samples.nc')
