import numpy as np
import pandas as pd
import os
import json
import hashlib

def get_data():
    name = "dayu19A"
    N = 50
    H = 100

    base_path = "../../../../data/"

    if name.lower() == "dayu19a":
        df = pd.read_csv(
            os.path.join(base_path, "inputdata_260219A/Dayu cave.txt"), sep="\t"
        )
        d18o_timeseries = pd.read_excel(
            os.path.join(base_path, "inputdata_260219A/ECHAM5_d18O_Dayu_Cave.xlsx")
        )
        
    elif name.lower() == "dayu19b":
        df = pd.read_csv(
            os.path.join(base_path, "inputdata_260219B/Dayu cave.txt"), sep="\t"
        )
        d18o_timeseries = pd.read_excel(
            os.path.join(base_path, "inputdata_260219B/ECHAM5_d18O_Dayu_Cave.xlsx")
        )
    elif name.lower() == "dayu19c":
        df = pd.read_csv(
            os.path.join(base_path, "inputdata_260219C/Dayu cave.txt"), sep="\t"
        )
        d18o_timeseries = pd.read_excel(
            os.path.join(base_path, "inputdata_260219C/ECHAM5_d18O_Dayu_Cave.xlsx")
        )

    depths = df["depth"].to_numpy()
    c14_ages = df["cal_c14_age"].to_numpy()
    c14_sigma = df["sigma_age"].to_numpy()
    true_ages = df["true_age"].to_numpy()
    D18O = df["d18O"].to_numpy()
    D18O_sigma = df["sigma_d18O"].to_numpy()

    D18O_reference_times = (
        d18o_timeseries["Year"].to_numpy() if d18o_timeseries is not None else None
    )
    D18O_reference = (
        d18o_timeseries["Filtered d18O"].to_numpy() if d18o_timeseries is not None else None
    )

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
        "N": N,
        "nc14d": len(c14_depths),
        "nd18o": len(D18O_depths),
        "nd18or": len(D18O_reference_times),

        "H": H,
        "dc": H / N,
        "pm": 1.71472,
        "ps": 0.7107,
        "th": true_ages[0],

        "cs": np.ascontiguousarray(np.linspace(0, H, N + 1), dtype = np.float64),

        "c14": c14_ages,
        "c14d": c14_depths,
        "c14s": c14_sigma,

        "d18o": D18O,
        "d18os": D18O_sigma,
        "d18od": D18O_depths,

        "d18ort": D18O_reference_times,
        "d18or": D18O_reference,
        "d18ota": true_ages_d18O,

        "dn": name
    }

    return data


def get_NS_config(c14 = False):
    num_points = 1000 if c14 else 1000 
    sd = 42

    config = {
        "np": num_points,
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