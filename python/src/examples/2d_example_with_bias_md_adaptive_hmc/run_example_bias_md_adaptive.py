import ctypes
import numpy as np
import matplotlib.pyplot as plt
import os
from example_define_variables_md_adaptive import get_hmc_config


def define_c_types(lib):
    lib.md.argtypes = [
        ctypes.c_double,  # dt
        ctypes.c_int,   # num_samples
        ctypes.c_int,  # num_HMC
        ctypes.c_int,  # num_dt
        ctypes.c_int,  # num_chains
        ctypes.c_double,  # sigma
        ctypes.POINTER(ctypes.c_double),  # samples_out
        ctypes.POINTER(ctypes.c_double),  # energy_out
        ctypes.POINTER(ctypes.c_double),  # bias_out
        ctypes.POINTER(ctypes.c_double),  # bias_std_out
        ctypes.c_double,    # gammma
        ctypes.c_double,    # beta
        ctypes.c_double,   #d, has to be a double to not yield weird results in c.
        ctypes.c_double,    #DeltaE
    ]
    lib.md.restype = None

    return lib


def main():
    os.add_dll_directory("C:/msys64/ucrt64/bin")
    lib = ctypes.CDLL("./example_with_bias_md_adaptive.dll")
    lib = define_c_types(lib)

    config = get_hmc_config()

    total = config["num_chains"] * config["num_samples"]
    total_times_2 = total * 2
    samples_out = (ctypes.c_double * total_times_2)()
    energy_out = (ctypes.c_double * total)()
    bias_out = (ctypes.c_double * total)()
    bias_std_out = (ctypes.c_double * total)()

    lib.md(
        ctypes.c_double(config["dt"]),
        ctypes.c_int(config["num_samples"]),
        ctypes.c_int(config["num_HMC"]),
        ctypes.c_int(config["num_dt"]),
        ctypes.c_int(config["num_chains"]),
        ctypes.c_double(config["sigma"]),
        samples_out,
        energy_out,
        bias_out,
        bias_std_out,
        ctypes.c_double(config["gamma"]),
        ctypes.c_double(config["beta"]),
        ctypes.c_double(config["d"]), # Has to be a double to not yield weird results in c.
        ctypes.c_double(config["DeltaE"]),
    )

    samples = np.ctypeslib.as_array(samples_out)
    print(len(samples))
    samples = np.reshape(samples, (config["num_chains"], config["num_samples"], 2))
    np.save("../../../output/example_with_bias_samples_md_adaptive_2d_new_kernel.npy", samples)
    print(samples)

    energy_values = np.ctypeslib.as_array(energy_out)
    energy_values = np.reshape(energy_values, (config["num_chains"], config["num_samples"]))
    np.save("../../../output/example_with_bias_energy_values_md_adaptive_2d_new_kernel.npy", energy_values)

    bias_weights = np.ctypeslib.as_array(bias_out)
    bias_weights = np.reshape(bias_weights, (config["num_chains"], config["num_samples"]))
    np.save("../../../output/example_with_bias_weights_md_adaptive_2d_new_kernel.npy", bias_weights)

    bias_std_out = np.ctypeslib.as_array(bias_std_out)
    bias_std_out = np.reshape(bias_std_out, (config["num_chains"], config["num_samples"]))
    np.save("../../../output/example_with_bias_std_out_md_adaptive_2d_new_kernel.npy", bias_std_out)

if __name__ == "__main__":
    main()
