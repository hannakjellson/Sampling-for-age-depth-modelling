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
    ) = read_data("dayu06")
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


def get_hmc_config(find_min = False, bias = "", cap = False, temp = False):
    keys = ["N", "H", "dc", "cs", "dt", "ns", "co", "ndt", "nHMC", "nch", "a", "b", "ces", "cw", "s", "dE", "d", "sb", "eb", "nl", "g", "thr", "sbt", "ebt", "nt", "nlsp", "mi", "gl"]
    config=dict.fromkeys(keys)
    if not find_min:
        config["N"] = 50                                                                      # Number of variables
        config["H"] = 100                                                                     # Sediment depth
        config["dc"] = config["H"] / config["N"]                                              # Segment depth
        config["cs"] = np.linspace(0, config["H"], config["N"] + 1)                           # Segment discretization
        config["dt"] = 0.0025                                                                 # Step size
        config["ns"] = 10000                                                                  # Number of samples
        config["co"] = 10                                                                     # Cutout
        config["ndt"] = 10                                                                    # Number of Leapfrog steps
        config["nHMC"] = 10                                                                   # Number of HMC steps between sampling
        config["nch"] = 5                                                                     # Number of chains
        config["a"] = 1.5                                                                     # Gamma prior shape
        config["b"] = 0.21                                                                    # Gamma prior rate
                
        if cap:                   
            config["ces"] = 0.1                                                               # Cap energy scaling
            config["cw"] = 100                                                                # Cap width
        if bias !="":                     
            config["s"] = 0.4                                                                 # Bias sigma
            config["dE"] = 50                                                                 # Max bias / Approximate size of valleys
                
            if bias !="":                      
                if bias == "unified":                     
                    config["d"] = 40                                                          # Nbr of sigmas to include when computing bias and bias gradient
                    config["sb"] = 1250                                                       # Starting value for umbrella bias
                    config["eb"] = 1550                                                       # End value for umbrella bias 
                    config["nl"] = (int)(1 + ((config["eb"] - config["sb"]) / (config["s"]))) # Number of umbrellas in each CV direction
                
                if bias == "rethinking":
                    config["g"] = 40                                                          # Scaling parameter
                    config["thr"] = config["s"] / 2                                           # Threshold for merging
                            
            if temp and bias == "unified":                      
                config["sbt"] = 1                                                             # Starting value for temp bias
                config["ebt"] = 100                                                           # End value for temp bias
                config["nt"] = 10                                                             # Number of temperatures
    else:                     
        config["N"] = 50                                                                      # Number of variables
        config["H"] = 100                                                                     # Sediment height
        config["dc"] = config["H"] / config["N"]                                              # Segment depth
        config["cs"] = np.linspace(0, config["H"], config["N"] + 1)                           # Segment discretization
        config["nch"] = 12                                                                    # Number of chains
        config["a"] = 1.5                                                                     # Gamma prior shape
        config["b"] = 0.21                                                                    # Gamma prior rate
                
        # For find_min_energy                     
        config["nlsp"] = 24                                                                   # Number of starting points
        config["mi"] = 1000                                                                   # Maximum number of iterations
        config["dt"] = 0.0001                                                                 # Stepsize
        config["gl"] = 1e-3                                                                   # Gradient limit
                        
    config_str = "_".join(
    f"{k}{v:.2f}" if isinstance(v, float) else f"{k}{v}"
    for k, v in config.items()
    if isinstance(v, (int, float))
    )   

    return config, config_str
