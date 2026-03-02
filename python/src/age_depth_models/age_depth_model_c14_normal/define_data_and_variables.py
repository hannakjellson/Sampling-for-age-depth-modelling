import numpy as np
import pandas as pd
import os
import ctypes
import hashlib
import json


class HMCConfig(ctypes.Structure):
    _fields_ = [
        ("ndt", ctypes.c_int64), # number of time steps
        ("nch", ctypes.c_int64), # number of chains
        ("ns", ctypes.c_int64), # number of samples
        ("sd", ctypes.c_int64), # seed

        ("dt", ctypes.c_double), # time step

        ("sp", ctypes.POINTER(ctypes.c_double)), # starting points
    ]

class Data(ctypes.Structure):
    _fields_ = [
        ("N", ctypes.c_int64), # number of sedimentation rates
        ("nc14", ctypes.c_int64), # number of c14 data points
        ("nd18o", ctypes.c_int64), # number of d18o data points
        ("nd18or", ctypes.c_int64), # number of d18o reference values

        ("H", ctypes.c_double), # sediment depth
        ("dc", ctypes.c_double), # segment depth
        ("pm", ctypes.c_double), # prior mu
        ("ps", ctypes.c_double), # prior sigma
        ("th", ctypes.c_double), # theta

        ("cs", ctypes.POINTER(ctypes.c_double)), # discrete depth points
        ("c14", ctypes.POINTER(ctypes.c_double)), # c14 ages
        ("c14d", ctypes.POINTER(ctypes.c_double)), # c14 depths
        ("c14s", ctypes.POINTER(ctypes.c_double)), # c14 sigma
        ("ic14v", ctypes.POINTER(ctypes.c_double)), # c14 inverse variance
        ("d18o", ctypes.POINTER(ctypes.c_double)), # d18o values
        ("d18od", ctypes.POINTER(ctypes.c_double)), # d18o depths
        ("d18os", ctypes.POINTER(ctypes.c_double)), # d18o sigma
        ("id18ov", ctypes.POINTER(ctypes.c_double)), # d18o inverse variance
        ("d18or", ctypes.POINTER(ctypes.c_double)), # d18o reference values
        ("d18ort", ctypes.POINTER(ctypes.c_double)), # d18o reference times
        ("d18ota", ctypes.POINTER(ctypes.c_double)), # d18o true ages

        ("dn", ctypes.c_char_p), # data name

    ]

def dict_to_struct(d: dict, struct_type):
    obj = struct_type()
    for field, _ in struct_type._fields_:
        if field in d:
            value = d[field]
            if isinstance(value, np.ndarray):
                value = value.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            elif isinstance(value, str):
                value = value.encode("utf-8")
            setattr(obj, field, value)
    return obj

def get_hmc_config():
    hmc_config = {
        "ndt": 700,
        "nch": 5,
        "ns": 10000,
        "sd": 42,

        "dt": 0.005,

        "sp": None,
    }
    return hmc_config

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
        d18o_timeseries = pd.read_excel(
            os.path.join(base_path, "inputdata_250306A/ECHAM5_d18O_Dayu_Cave.xlsx")
        )

    elif name.lower() == "dayu26":
        df = pd.read_csv(
            os.path.join(base_path, "inputdata_250826/Dayu cave.txt"), sep="\t"
        )
        d18o_timeseries = pd.read_excel(
            os.path.join(base_path, "inputdata_250826/ECHAM5_d18O_Dayu_Cave.xlsx")
        )

    elif name.lower() == "dayu19a":
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

data = get_data()

hmc_config = get_hmc_config()

c_data = dict_to_struct(
    data, Data
)
