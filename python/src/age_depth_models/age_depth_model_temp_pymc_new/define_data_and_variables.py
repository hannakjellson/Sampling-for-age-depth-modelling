import numpy as np
import pandas as pd
import os
import hashlib
import json

def get_adam_config():
    adam_config = {
        "nsp": 100,
        "mi": 1000000,
        "sd": 1000,

        "dt": 0.00001,
        "gl": 0.00001,
    }

    return adam_config

def get_hmc_config():
    hmc_config = {
        "ndt": 700,
        "nch": 5,
        "ns": 1000,
        "sd": 1000,

        "dt": 0.0008,

        "sp": None,
    }
    return hmc_config

def get_opes_config():
    hmc_config = get_hmc_config()  # assume this returns an HMCConfig as a dict or struct
    ebt = 20
    nt = 40

    opes_config = {
        "hmcc": hmc_config,  # keep the nested config as a dict
        "nhmc": 1,
        "nt": nt,
        "ebt": ebt,
        "dfd": 100.0,
        "dfa": 220.0,
        "bs": np.ascontiguousarray(1 / np.geomspace(1, ebt, nt)),  # convert to list for JSON/dict
        "sb": 1,
    }
    return opes_config

def get_data():
    name = "dayu06"
    N = 50
    H = 100

    base_path = "../../../../data/"
    d18o_timeseries = None

    if name.lower() == "dayu06":
        df = pd.read_csv(
            os.path.join(base_path, "inputdata_250306A/Dayu cave.txt"), sep="\t"
        )
        d18o_timeseries = pd.read_csv(
            os.path.join(base_path, "inputdata_250306A/d18O_timeseries.txt"), sep="\t"
        )
    elif name.lower() == "dayu12a":
        df = pd.read_csv(
            os.path.join(base_path, "inputdata_260212A/Dayu cave.txt"), sep="\t"
        )
        d18o_timeseries = pd.read_csv(
            os.path.join(base_path, "inputdata_260212A/d18O_timeseries.txt"), sep="\t"
        )
        
    elif name.lower() == "dayu12b":
        df = pd.read_csv(
            os.path.join(base_path, "inputdata_260212B/Dayu cave.txt"), sep="\t"
        )
        d18o_timeseries = pd.read_csv(
            os.path.join(base_path, "inputdata_260212B/d18O_timeseries.txt"), sep="\t"
        )
    elif name.lower() == "dayu12c":
        df = pd.read_csv(
            os.path.join(base_path, "inputdata_260212C/Dayu cave.txt"), sep="\t"
        )
        d18o_timeseries = pd.read_csv(
            os.path.join(base_path, "inputdata_260212C/d18O_timeseries.txt"), sep="\t"
        )

    elif name.lower() == "dayu26":
        df = pd.read_csv(
            os.path.join(base_path, "inputdata_250826/Dayu cave.txt"), sep="\t"
        )
        d18o_timeseries = pd.read_csv(
            os.path.join(base_path, "inputdata_250826/d18O_timeseries.txt"), sep="\t"
        )

    df.columns = df.columns.str.replace("%", "").str.strip()
    if d18o_timeseries is not None:
        d18o_timeseries.columns = d18o_timeseries.columns.str.replace(
            "% ", ""
        ).str.strip()

    print(df.keys())
    depths = df["depth"].to_numpy()
    c14_ages = df["cal_c14_age"].to_numpy()
    c14_sigma = df["sigma_age"].to_numpy()
    true_ages = df["true_age"].to_numpy()
    d18o = df["d18O"].to_numpy()
    d18o_sigma = df["sigma_d18O"].to_numpy()

    d18o_reference_times = (
        d18o_timeseries["Year"].to_numpy()
    )
    d18o_reference = (
        d18o_timeseries["d18O"].to_numpy()
    )

    c14_mask = ~np.isnan(c14_ages)
    d18o_mask = ~np.isnan(d18o)

    c14_depths = depths[c14_mask]
    c14_ages = c14_ages[c14_mask]
    c14_sigma = c14_sigma[c14_mask]

    d18o = d18o[d18o_mask][::-1]
    d18o_sigma = d18o_sigma[d18o_mask][::-1]
    d18o_depths = depths[d18o_mask][
        ::-1
    ]  # This does not overlap with the c14 depths in the file.
    true_ages_d18O = true_ages[d18o_mask][::-1]

    data = {
        "N": N,
        "nc14": len(c14_depths),
        "nd18o": len(d18o_depths),
        "nd18or": len(d18o_reference_times),

        "H": H,
        "dc": H / N,
        "pm": 1.71472,
        "ps": 0.7107,
        "th": true_ages[0],

        "cs": np.ascontiguousarray(np.linspace(0, H, N + 1), dtype = np.float64),

        "c14": np.ascontiguousarray(c14_ages, dtype = np.float64),
        "c14d": np.ascontiguousarray(c14_depths, dtype = np.float64),
        "c14s": np.ascontiguousarray(c14_sigma, dtype = np.float64),
        "ic14v": np.ascontiguousarray(1/c14_sigma**2, dtype = np.float64),
        
        "d18o": np.ascontiguousarray(d18o, dtype = np.float64),
        "d18od": np.ascontiguousarray(d18o_depths, dtype = np.float64),
        "d18os": np.ascontiguousarray(d18o_sigma, dtype = np.float64),
        "id18ov": np.ascontiguousarray(1/d18o_sigma**2, dtype = np.float64),
        
        "d18or": np.ascontiguousarray(d18o_reference, dtype = np.float64),
        "d18ort": np.ascontiguousarray(d18o_reference_times, dtype = np.float64),
        "d18ota": np.ascontiguousarray(true_ages_d18O, dtype = np.float64),

        "dn": name, # .encode("utf-8")?
    }

    return data

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

adam_config = get_adam_config()
opes_config = get_opes_config()
data = get_data()

opes_hash = hash_configs(opes_config, data)
