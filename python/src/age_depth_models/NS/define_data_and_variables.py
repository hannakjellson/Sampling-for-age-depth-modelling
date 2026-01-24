import numpy as np
import pandas as pd
import os
import json
import hashlib

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


def get_NS_config(c14 = False):
    N = 50
    H = 100
    delta_c = H / N
    cs = np.linspace(0, H, N + 1)
    num_points = 1000 if c14 else 1000 
    a = 1.5
    b = 0.21
    sd = 42

    config = {
        "N": N,
        "H": H,
        "delta_c": delta_c,
        "cs": cs,
        "num_points": num_points,
        "a": a,
        "b": b,
        "sd": sd,
    }

    return config

def sanitize(obj):
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "shape": obj.shape,
            "dtype": str(obj.dtype),
        }
    elif isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    else:
        return obj
    
def hash_configs(*configs, algo="sha256", length=10):
    """
    Create a stable hash from one or more config dicts.
    """
    sanitized = sanitize(configs)

    # Canonical JSON: sorted keys, no whitespace
    canonical = json.dumps(
        sanitized,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    h = hashlib.new(algo)
    h.update(canonical)

    return h.hexdigest()[:length]

def make_dumpable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: make_dumpable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_dumpable(v) for v in obj]
    return obj