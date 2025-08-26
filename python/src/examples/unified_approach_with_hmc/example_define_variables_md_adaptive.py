import numpy as np
import pandas as pd
import os


def get_hmc_config():
    dt = 0.01
    num_samples = 2000000
    num_HMC = 10
    num_dt = 10
    num_chains = 1
    sigma = 0.05
    num_lambda = 121



    config = {
        "dt": dt,
        "num_samples" : num_samples,
        "num_HMC": num_HMC,
        "num_dt": num_dt,
        "num_chains": num_chains,
        "num_lambda" : num_lambda,
        "sigma": sigma,
    }

    return config
