import ctypes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from define_data_and_variables import get_data, get_hmc_config
from datetime import datetime
from multiprocessing import Pool, cpu_count


def define_c_types(lib):
    lib.hmc.argtypes = [
        ctypes.c_int,  # N
        ctypes.c_double,  # H
        ctypes.c_double,  # delta_c
        ctypes.POINTER(ctypes.c_double),  # cs
        ctypes.c_double,  # dt
        ctypes.c_int,  # num_dt
        ctypes.c_int,  # num_HMC
        ctypes.c_int,  # num_chains
        ctypes.c_int,  # num_samples
        ctypes.c_int,   # problem_index
        ctypes.c_double,  # bias_sigma
        ctypes.c_double,  # a
        ctypes.c_double,  # b
        ctypes.c_double,  # theta
        ctypes.c_int,  # num_c14_depths
        ctypes.c_int,  # num_D18O_depths
        ctypes.c_int,  # num_D18O_reference_times
        ctypes.c_double,  # startbias
        ctypes.c_double,  # endbias
        ctypes.POINTER(ctypes.c_double),  # c14_ages
        ctypes.POINTER(ctypes.c_double),  # c14_depths
        ctypes.POINTER(ctypes.c_double),  # c14_sigma
        ctypes.POINTER(ctypes.c_double),  # D18O
        ctypes.POINTER(ctypes.c_double),  # D18O_depths
        ctypes.POINTER(ctypes.c_double),  # D18O_sigma
        ctypes.POINTER(ctypes.c_double),  # D18O_reference
        ctypes.POINTER(ctypes.c_double),  # D18O_reference_times
        ctypes.c_double,  # gamma
        ctypes.c_double,  # beta
        ctypes.c_double,  # DeltaE
        ctypes.c_double,  # threshold
        ctypes.POINTER(ctypes.c_double),  # samples_out
        ctypes.POINTER(ctypes.c_double),  # energy_out
        ctypes.POINTER(ctypes.c_double),  # bias_out
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

    total = config["nch"] * config["ns"]
    total_times_N = total * config["N"]
    samples_out = (ctypes.c_double * total_times_N)()
    energy_out = (ctypes.c_double * total)()
    bias_out = (ctypes.c_double * total)()

    lib.hmc(
        ctypes.c_int(config["N"]),
        ctypes.c_double(config["H"]),
        ctypes.c_double(config["dc"]),
        cs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(config["dt"]),
        ctypes.c_int(config["ndt"]),
        ctypes.c_int(config["nHMC"]),
        ctypes.c_int(config["nch"]),
        ctypes.c_int(config["ns"]),
        ctypes.c_int(config["pidx"]),
        ctypes.c_double(config["sigma"]),
        ctypes.c_double(config["a"]),
        ctypes.c_double(config["b"]),
        ctypes.c_double(data["theta"]),
        ctypes.c_int(data["num_c14_depths"]),
        ctypes.c_int(data["num_D18O_depths"]),
        ctypes.c_int(data["num_D18O_reference_times"]),
        ctypes.c_double(config["sb"]),
        ctypes.c_double(config["eb"]),
        c14_ages.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        c14_depths.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        c14_sigma.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        D18O.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        D18O_depths.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        D18O_sigma.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        D18O_reference.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        D18O_reference_times.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(config["gamma"]),
        ctypes.c_double(config["beta"]),
        ctypes.c_double(config["dE"]),
        ctypes.c_double(config["th"]),
        samples_out,
        energy_out,
        bias_out,
    )

    samples = np.ctypeslib.as_array(samples_out)
    samples = np.reshape(samples, (config["nch"], config["ns"], config["N"]))
    np.save(f"../../../../output/ad_reth_cap/samples_{config_str}.npy", samples)

    energy_values = np.ctypeslib.as_array(energy_out)
    energy_values = np.reshape(energy_values, (config["nch"], config["ns"]))
    np.save(f"../../../../output/ad_reth_cap/energy_{config_str}.npy", energy_values)

    bias_values = np.ctypeslib.as_array(bias_out)
    bias_values = np.reshape(bias_values, (config["nch"], config["ns"]))
    np.save(f"../../../../output/ad_reth_cap/bias_{config_str}.npy", bias_values)

    print("Resampling\n")
    resampled_samples = []
    cutout = 10

    weights = np.exp(bias_values)
    for i in range(config["nch"]):
        weights_i = weights[i, cutout:] / sum(weights[i, cutout:])
        indices = np.random.choice(
            np.arange(cutout, len(weights_i) + cutout),
            size=int(len(weights_i)),
            replace=True,
            p=weights_i,
        )
        print(f"Chain {i} done resampling")
        resampled_samples.append(samples[i, indices, :])
            

    resampled_samples = np.array(resampled_samples)
    np.save(f"../../../../output/ad_reth_cap/resamp_{config_str}.npy", resampled_samples)


if __name__ == "__main__":
    main()
