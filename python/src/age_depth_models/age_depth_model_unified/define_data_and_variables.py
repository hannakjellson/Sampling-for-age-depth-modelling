import numpy as np
import pandas as pd
import os


def read_data(data):
    base_path = "../../../../data/"
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
    elif data.lower() == "dayu26":
        df = pd.read_csv(
            os.path.join(base_path, "inputdata_250826/Dayu cave.txt"), sep="\t"
        )
        D18O_timeseries = pd.read_csv(
            os.path.join(base_path, "inputdata_250826/d18O_timeseries.txt"), sep="\t"
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
    ) = read_data("dayu26")
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
    true_ages_d18O = true_ages[D18O_mask][::-1]

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
        "true_ages_D18O": true_ages_d18O,
    }

    return data


def get_hmc_config():
    N = 10 # 50
    H = 20 # 100
    dc = H / N
    cs = np.linspace(0, H, N + 1)
    dt = 0.0025
    nsamples = 10000
    cutout = 10
    ndt = 100
    nHMC = 10
    nchains = 4
    pidx = 5
    a = 1.5
    b = 0.21
    dflim = 40
    startbias = 1250 # Todo: try to find out when and why it goes unstable if i increase delta F and think about what happens if i do, i guess I allow to go to less likeli regions, which is good for finding them but not good for uniform sampling.
    endbias = 1550
    sigma = 0.4
    nlambda = (int)(1 + ((endbias - startbias) / (sigma)))
    dist = 40

    config = {
        "N": N,
        "H": H,
        "dc": dc,
        "cs": cs,
        "dt": dt,
        "ndt": ndt,
        "nHMC": nHMC,
        "ns": nsamples,
        "nch": nchains,
        "nl": nlambda, 
        "pidx": pidx,
        "sig": sigma,
        "a": a,
        "b": b,
        "co" : cutout,
        "dflim" : dflim,
        "sb" : startbias,
        "eb" : endbias,
        "dist" : dist,
    }
    
    config_str = "_".join(
    f"{k}{v:.2f}" if isinstance(v, float) else f"{k}{v}"
    for k, v in config.items()
    if isinstance(v, (int, float))
    )   

    return config, config_str
