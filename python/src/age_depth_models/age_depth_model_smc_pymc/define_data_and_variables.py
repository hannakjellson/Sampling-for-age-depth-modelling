import numpy as np
import pandas as pd
import os
import hashlib
import json

def get_smc_config():
    nch = 10000

    smc_config = {
        "nch":nch,
    }
    return smc_config

def get_data():
    name = "dayu19A"
    N = 50
    H = 100

    base_path = "../../../../data/"
    d18o_timeseries = None

    # if name.lower() == "dayu06":
    #     df = pd.read_csv(
    #         os.path.join(base_path, "inputdata_250306A/Dayu cave.txt"), sep="\t"
    #     )
    #     d18o_timeseries = pd.read_excel(
    #         os.path.join(base_path, "inputdata_250306A/ECHAM5_d18O_Dayu_Cave.xlsx")
    #     )

    # elif name.lower() == "dayu26":
    #     df = pd.read_csv(
    #         os.path.join(base_path, "inputdata_250826/Dayu cave.txt"), sep="\t"
    #     )
    #     d18o_timeseries = pd.read_csv(
    #         os.path.join(base_path, "inputdata_250826/d18O_timeseries.txt"), sep="\t"
    #     )

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

    # df.columns = df.columns.str.replace("%", "").str.strip()
    # if d18o_timeseries is not None:
    #     d18o_timeseries.columns = d18o_timeseries.columns.str.replace(
    #         "%", ""
    #     ).str.strip()

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
        d18o_timeseries["Filtered d18O"].to_numpy()
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
    ]
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

        "cs": np.linspace(0, H, N + 1),

        "c14": c14_ages,
        "c14d": c14_depths,
        "c14s": c14_sigma,
        
        "d18o": d18o,
        "d18od": d18o_depths,
        "d18os": d18o_sigma,
        
        "d18or": d18o_reference,
        "d18ort": d18o_reference_times,
        "d18ota": true_ages_d18O,

        "dn": name,
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

    canonical = json.dumps(
        sanitized,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    h = hashlib.new(algo)
    h.update(canonical)

    return h.hexdigest()[:length]

data = get_data()