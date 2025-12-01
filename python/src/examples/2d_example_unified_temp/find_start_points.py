import ctypes
import numpy as np
import matplotlib.pyplot as plt
import os
from define_variables import get_hmc_config


def define_c_types(lib):
    lib.hmc.argtypes = [
        ctypes.c_double,  # dt
        ctypes.c_int,   # num_samples
        ctypes.c_int,  # num_HMC
        ctypes.c_int,  # num_dt
        ctypes.c_int,  # num_chains
        ctypes.c_int,  # num_lambda
        ctypes.c_double,  # dt
        ctypes.POINTER(ctypes.c_double),  # samples_out
        ctypes.POINTER(ctypes.c_double),  # energy_out
        ctypes.POINTER(ctypes.c_double),  # delF_out
        ctypes.POINTER(ctypes.c_double),  # bias_out
    ]
    lib.hmc.restype = None

    return lib


def main():
    os.add_dll_directory("C:/msys64/ucrt64/bin")
    lib = ctypes.CDLL("./hmc.dll")
    lib = define_c_types(lib)

    config, config_str = get_hmc_config()

    num_chains = 100
    num_samples = 100000
    dt = 0.1
    num_HMC = 1
    num_dt = 100
    total = num_chains * num_samples
    total_times_2 = total * 2
    total_times_num_lambda = total
    samples_out = (ctypes.c_double * total_times_2)()
    energy_out = (ctypes.c_double * total)()
    delF_out = (ctypes.c_double * total_times_num_lambda)()
    bias_out = (ctypes.c_double * total)()

    lib.hmc(
        ctypes.c_double(dt),
        ctypes.c_int(num_samples),
        ctypes.c_int(num_HMC),
        ctypes.c_int(num_dt),
        ctypes.c_int(num_chains),
        ctypes.c_int(1),
        1, 
        samples_out,
        energy_out,
        delF_out,
        bias_out,
    )

    samples = np.ctypeslib.as_array(samples_out)
    samples = np.reshape(samples, (num_chains, num_samples, 2))
    np.save(f"../../../../output/2d_example_unified_temp/start_samples.npy", samples)

    energy_values = np.ctypeslib.as_array(energy_out)
    energy_values = np.reshape(energy_values, (num_chains, num_samples))
    np.save(f"../../../../output/2d_example_unified_temp/start_energy_values.npy", energy_values)

    delF_values = np.ctypeslib.as_array(delF_out)
    delF_values = np.reshape(delF_values, (num_chains, num_samples))
    np.save(f"../../../../output/2d_example_unified_temp/start_delF.npy", delF_values)

    bias_values = np.ctypeslib.as_array(bias_out)
    bias_values = np.reshape(bias_values, (num_chains, num_samples))
    np.save(f"../../../../output/2d_example_unified_temp/start_bias_values.npy", bias_values)


if __name__ == "__main__":
    main()
