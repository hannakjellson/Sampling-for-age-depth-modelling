import numpy as np
import pandas as pd
import os


def get_hmc_config():
    dt = 0.05
    num_MD = 10000
    num_dt = 500
    num_chains = 1
    bias_std = 0.1
    gamma = 40
    beta = 1.0
    d = 1 #same as N
    DeltaE = 5.0


    config = {
        "dt": dt,
        "num_MD": num_MD,
        "num_dt": num_dt,
        "num_chains": num_chains,
        "bias_std": bias_std,
        "gamma" : gamma,
        "beta" : beta,
        "d" : d,
        "DeltaE" : DeltaE,
    }

    return config
