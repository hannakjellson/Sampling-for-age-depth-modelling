import numpy as np
import pandas as pd
import os


def get_hmc_config():
    dt = 0.1
    num_samples = 100000
    num_HMC = 1
    num_dt = 700
    num_chains = 4
    sigma = 2.2
    num_temps = 7



    config = {
        "dt": dt,
        "num_samples" : num_samples,
        "num_HMC": num_HMC,
        "num_dt": num_dt,
        "num_chains": num_chains,
        "num_temps" : num_temps,
        "sigma": sigma,
    }

    config_str = "_".join(
    f"{k}{v:.2f}" if isinstance(v, float) else f"{k}{v}"
    for k, v in config.items()
    if isinstance(v, (int, float))
    )  

    return config, config_str
