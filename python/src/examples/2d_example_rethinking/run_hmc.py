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
    lib.hmc.restype = None

    return lib


def main():
    os.add_dll_directory("C:/msys64/ucrt64/bin")
    lib = ctypes.CDLL("./hmc.dll")
    lib = define_c_types(lib)

    config, config_str = get_hmc_config()

    total = config["num_chains"] * config["num_samples"]
    total_times_2 = total * 2
    samples_out = (ctypes.c_double * total_times_2)()
    energy_out = (ctypes.c_double * total)()
    bias_out = (ctypes.c_double * total)()
    bias_std_out = (ctypes.c_double * total)()

    lib.hmc(
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
    np.save(f"../../../../output/2d_example_rethinking/samples_{config_str}.npy", samples)
    print(samples)

    energy_values = np.ctypeslib.as_array(energy_out)
    energy_values = np.reshape(energy_values, (config["num_chains"], config["num_samples"]))
    np.save(f"../../../../output/2d_example_rethinking/energy_values_{config_str}.npy", energy_values)

    bias_weights = np.ctypeslib.as_array(bias_out)
    bias_weights = np.reshape(bias_weights, (config["num_chains"], config["num_samples"]))
    np.save(f"../../../../output/2d_example_rethinking/bias_values_{config_str}.npy", bias_weights)

    bias_std_out = np.ctypeslib.as_array(bias_std_out)
    bias_std_out = np.reshape(bias_std_out, (config["num_chains"], config["num_samples"]))
    np.save(f"../../../../output/2d_example_rethinking/bias_std_{config_str}.npy", bias_std_out)

    
    print("Resampling")
    resampled_samples = []
    cutout = 100

    weights = np.exp(bias_weights)

    for i in range(config["num_chains"]):
        weights_i = weights[i, cutout:] / sum(weights[i, cutout:])
        indices = np.random.choice(
            np.arange(cutout, len(weights_i) + cutout),
            size=int(len(weights_i)),
            replace=True,
            p=weights_i,
        )
        print(f"Resampled chain {i}")
        resampled_samples.append(samples[i, indices, :])
    resampled_samples = np.array(resampled_samples)

    np.save(f"../../../../output/2d_example_rethinking/resampsamp_{config_str}.npy", resampled_samples)

if __name__ == "__main__":
    main()
