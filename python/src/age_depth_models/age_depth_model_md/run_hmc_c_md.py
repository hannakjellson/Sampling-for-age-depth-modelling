import ctypes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from define_data_and_variables_md import get_data, get_hmc_config


def define_c_types(lib):
    lib.md.argtypes = [
        ctypes.c_int,  # N
        ctypes.c_double,  # H
        ctypes.c_double,  # delta_c
        ctypes.POINTER(ctypes.c_double),  # cs
        ctypes.c_double,  # dt
        ctypes.POINTER(ctypes.c_double),  # M
        ctypes.c_int,  # num_MD
        ctypes.c_int,  # num_chains
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
        ctypes.c_double,    # bias_std
        ctypes.c_int,   # problem_index
        ctypes.c_double,    # gamma
        ctypes.c_double,    # beta
        ctypes.c_double,   # d
        ctypes.c_double,    # DeltaE
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # samples_out
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # energy_out
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # bias_out
    ]

    lib.md.restype = None

    return lib


def main():
    data = get_data()
    config = get_hmc_config()

    # Load library depending on OS
    os.add_dll_directory("C:/msys64/ucrt64/bin")
    lib = ctypes.CDLL("./md.dll")
    lib = define_c_types(lib)

    cs = np.ascontiguousarray(config["cs"], dtype=np.float64)
    M = np.ascontiguousarray(config["M"], dtype=np.float64)

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

    samples_out = ctypes.POINTER(ctypes.c_double)()
    energy_out = ctypes.POINTER(ctypes.c_double)()
    bias_out = ctypes.POINTER(ctypes.c_double)()


    lib.md(
        ctypes.c_int(config["N"]),
        ctypes.c_double(config["H"]),
        ctypes.c_double(config["delta_c"]),
        cs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(config["dt"]),
        M.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(config["num_MD"]),
        ctypes.c_int(config["num_chains"]),
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
        ctypes.c_double(config["bias_std"]),
        ctypes.c_int(config["problem_index"]),
        ctypes.c_double(config["gamma"]),
        ctypes.c_double(config["beta"]),
        ctypes.c_double(config["d"]),
        ctypes.c_double(config["DeltaE"]),
        ctypes.pointer(samples_out),
        ctypes.pointer(energy_out),
        ctypes.pointer(bias_out),
    )

    samples = np.ctypeslib.as_array(
        samples_out, shape=(config["num_chains"], config["num_MD"], config["N"])
    )
    np.save("../../../output/samples_md.npy", samples)

    energy_values = np.ctypeslib.as_array(
        energy_out, shape=(config["num_chains"], config["num_MD"])
    )
    np.save("../../../output/energy_values_md.npy", energy_values)

    bias_values = np.ctypeslib.as_array(
        bias_out, shape=(config["num_chains"], config["num_MD"])
    )
    np.save("../../../output/bias_values_md.npy", bias_values)


if __name__ == "__main__":
    main()
