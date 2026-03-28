import numpy as np
import pandas as pd
import os


def get_hmc_config():
    dt = 0.05
    num_samples = 10000
    num_HMC = 1
    num_dt = 10
    dE = 40
    num_chains = 4
    num_lambda = 20
    sigma = dE / (2 * (num_lambda - 1))
    max_dF = 40



    config = {
        "dt": dt,
        "num_samples" : num_samples,
        "num_HMC": num_HMC,
        "num_dt": num_dt,
        "num_chains": num_chains,
        "num_lambda" : num_lambda,
        "dE" : dE,
        "sigma": sigma,
        "max_dF": max_dF,
    }

    config_str = "_".join(
    f"{k}{v:.2f}" if isinstance(v, float) else f"{k}{v}"
    for k, v in config.items()
    if isinstance(v, (int, float))
    )  

    return config, config_str
