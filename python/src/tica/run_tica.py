import ctypes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from define_data_and_variables import get_data, get_hmc_config
import datetime as datetime


def define_c_types(lib):
    lib.hmc.argtypes = [
        ctypes.c_int,  # N
        ctypes.c_double,  # H
        ctypes.c_double,  # dc
        ctypes.POINTER(ctypes.c_double),  # cs
        ctypes.c_double,  # dt
        ctypes.c_int,  # ndt
        ctypes.c_int,  # nHMC
        ctypes.c_int,  # nsamples
        ctypes.c_int,  # nchains
        ctypes.c_int,  # cutoff
        ctypes.c_double,  # a
        ctypes.c_double,  # b
        ctypes.c_double,  # theta
        ctypes.c_int,  # num_c14_depths
        ctypes.c_int,  # num_D18O_depths
        ctypes.c_int,  # num_D18O_reference_times
        ctypes.POINTER(ctypes.c_double),  # c14_ages
        ctypes.POINTER(ctypes.c_double),  # c14_depths
        ctypes.POINTER(ctypes.c_double),  # c14_sigma
        ctypes.POINTER(ctypes.c_double),  # D18O
        ctypes.POINTER(ctypes.c_double),  # D18O_depths
        ctypes.POINTER(ctypes.c_double),  # D18O_sigma
        ctypes.POINTER(ctypes.c_double),  # D18O_reference
        ctypes.POINTER(ctypes.c_double),  # D18O_reference_times
        ctypes.POINTER(ctypes.c_double),  # samples
    ]

    lib.hmc.restype = None

    return lib


def main():
    data = get_data()
    config, config_str = get_hmc_config()

    # Load library depending on OS
    os.add_dll_directory("C:/msys64/ucrt64/bin")
    lib = ctypes.CDLL("./hmc.dll")
    lib = define_c_types(lib)

    cs = np.ascontiguousarray(config["cs"], dtype=np.float64)

    c14_ages = np.ascontiguousarray(data["c14_ages"], dtype=np.float64)
    c14_depths = np.ascontiguousarray(data["c14_depths"], dtype=np.float64)
    c14_sigma = np.ascontiguousarray(data["c14_sigma"], dtype=np.float64)

    D18O = np.ascontiguousarray(data["d18O"], dtype=np.float64)
    D18O_depths = np.ascontiguousarray(data["d18O_depths"], dtype=np.float64)
    D18O_sigma = np.ascontiguousarray(data["d18O_sigma"], dtype=np.float64)
    D18O_reference_times = np.ascontiguousarray(
        data["d18O_reference_times"], dtype=np.float64
    )
    D18O_reference = np.ascontiguousarray(data["d18O_reference"], dtype=np.float64)

    total_times_N = config["nch"] * (config["ns"] - config["cutoff"]) * config["N"]
    samples = (ctypes.c_double * total_times_N)()

    lib.hmc(
        ctypes.c_int(config["N"]),
        ctypes.c_double(config["H"]),
        ctypes.c_double(config["dc"]),
        cs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(config["dt"]),
        ctypes.c_int(config["ndt"]),
        ctypes.c_int(config["nHMC"]),
        ctypes.c_int(config["ns"]),
        ctypes.c_int(config["nch"]),
        ctypes.c_int(config["cutoff"]),       
        ctypes.c_double(config["a"]),
        ctypes.c_double(config["b"]),
        ctypes.c_double(data["theta"]),
        ctypes.c_int(data["num_c14_depths"]),
        ctypes.c_int(data["num_D18O_depths"]),
        ctypes.c_int(data["num_D18O_reference_times"]),
        c14_ages.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        c14_depths.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        c14_sigma.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        D18O.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        D18O_depths.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        D18O_sigma.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        D18O_reference.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        D18O_reference_times.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        samples,
    )
    # print("hello")
    
    samples = np.reshape(samples, (config["nch"], config["ns"] - config["cutoff"], config["N"]))
    np.save(f"../../../output/tica/samples_{config_str}.npy", samples)


if __name__ == "__main__":
    main()
