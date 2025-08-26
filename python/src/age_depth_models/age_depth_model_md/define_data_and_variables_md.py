import numpy as np
import pandas as pd
import os


def read_data(data):
    base_path = "../../../data/"
    D18O_timeseries = None

    if data.lower() == "dayu06":
        df = pd.read_csv(
            os.path.join(base_path, "inputdata_250306A/Dayu cave.txt"), sep="\t"
        )
        D18O_timeseries = pd.read_csv(
            os.path.join(base_path, "inputdata_250306A/d18O_timeseries.txt"), sep="\t"
        )

    elif data.lower() == "dayu07":
        df = pd.read_csv(
            os.path.join(base_path, "inputdata_250307A/Dayu cave.txt"), sep="\t"
        )
        D18O_timeseries = pd.read_csv(
            os.path.join(base_path, "inputdata_250307A/d18O_timeseries.txt"), sep="\t"
        )

    elif data.lower() == "shenqi":
        df = pd.read_csv(os.path.join(base_path, "Shenqi cave.txt"), sep="\t")

    df.columns = df.columns.str.replace("%", "").str.strip()
    if D18O_timeseries is not None:
        D18O_timeseries.columns = D18O_timeseries.columns.str.replace(
            "%", ""
        ).str.strip()

    depths = df["depth"].to_numpy()
    c14_ages = df["cal_c14_age"].to_numpy()
    c14_sigma = df["sigma_age"].to_numpy()
    true_ages = df["true_age"].to_numpy()
    D18O = df["d18O"].to_numpy()
    D18O_sigma = df["sigma_d18O"].to_numpy()

    D18O_reference_times = (
        D18O_timeseries["Year"].to_numpy() if D18O_timeseries is not None else None
    )
    D18O_reference = (
        D18O_timeseries["d18O"].to_numpy() if D18O_timeseries is not None else None
    )

    return (
        depths,
        c14_ages,
        c14_sigma,
        true_ages,
        D18O,
        D18O_sigma,
        D18O_reference_times,
        D18O_reference,
    )


def get_data():
    (
        depths,
        c14_ages,
        c14_sigma,
        true_ages,
        D18O,
        D18O_sigma,
        D18O_reference_times,
        D18O_reference,
    ) = read_data("dayu07")
    c14_mask = ~np.isnan(c14_ages)
    D18O_mask = ~np.isnan(D18O)

    c14_depths = depths[c14_mask]
    c14_ages = c14_ages[c14_mask]
    c14_sigma = c14_sigma[c14_mask]

    D18O = D18O[D18O_mask][::-1]
    D18O_sigma = D18O_sigma[D18O_mask][::-1]
    D18O_depths = depths[D18O_mask][
        ::-1
    ]  # This does not overlap with the c14 depths in the file.

    data = {
        "theta": true_ages[0],
        "c14_ages": c14_ages,
        "c14_depths": c14_depths,
        "c14_sigma": c14_sigma,
        "num_c14_depths": len(c14_depths),
        "d18O": D18O,
        "d18O_sigma": D18O_sigma,
        "d18O_depths": D18O_depths,
        "d18O_reference_times": D18O_reference_times,
        "d18O_reference": D18O_reference,
        "num_D18O_depths": len(D18O_depths),
        "num_D18O_reference_times": len(D18O_reference_times),
    }

    return data


def get_hmc_config():
    N = 50
    H = 100.0
    delta_c = H / N
    cs = np.linspace(0, H, N + 1)
    dt = 0.0001 # 0.0001 looks good with bias_std=0.1, gamma =2, beta = 1 and deltaE=50. 
    M = np.eye(N).flatten()
    num_MD = 10000
    num_chains = 5
    a = 1.5
    b = 0.21
    bias_std = 0.1 # Guess
    problem_index = 45
    gamma = 2
    beta = 1
    d = 1
    DeltaE = 50 #Guess


    config = {
        "N": N,
        "H": H,
        "delta_c": delta_c,
        "cs": cs,
        "dt": dt,
        "M": M,
        "num_MD": num_MD,
        "num_chains": num_chains,
        "a": a,
        "b": b,
        "bias_std": bias_std,
        "problem_index": problem_index,
        "gamma": gamma,
        "beta": beta,
        "d": d,
        "DeltaE": DeltaE,
    }

    return config
