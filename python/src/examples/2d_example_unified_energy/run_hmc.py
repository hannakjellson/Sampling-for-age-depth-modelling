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
        ctypes.c_double,  # sigma
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

    total = config["num_chains"] * config["num_samples"]
    total_times_2 = total * 2
    total_times_num_lambda = total * config["num_lambda"]
    samples_out = (ctypes.c_double * total_times_2)()
    energy_out = (ctypes.c_double * total)()
    delF_out = (ctypes.c_double * total_times_num_lambda)()
    bias_out = (ctypes.c_double * total)()

    lib.hmc(
        ctypes.c_double(config["dt"]),
        ctypes.c_int(config["num_samples"]),
        ctypes.c_int(config["num_HMC"]),
        ctypes.c_int(config["num_dt"]),
        ctypes.c_int(config["num_chains"]),
        ctypes.c_int(config["num_lambda"]),
        ctypes.c_double(config["sigma"]),
        samples_out,
        energy_out,
        delF_out,
        bias_out,
    )

    samples = np.ctypeslib.as_array(samples_out)
    samples = np.reshape(samples, (config["num_chains"], config["num_samples"], 2))
    np.save(f"../../../../output/2d_example_unified_energy/samples_{config_str}.npy", samples)

    energy_values = np.ctypeslib.as_array(energy_out)
    energy_values = np.reshape(energy_values, (config["num_chains"], config["num_samples"]))
    np.save(f"../../../../output/2d_example_unified_energy/energy_values_{config_str}.npy", energy_values)

    delF_values = np.ctypeslib.as_array(delF_out)
    delF_values = np.reshape(delF_values, (config["num_chains"], config["num_samples"], config["num_lambda"]))
    np.save(f"../../../../output/2d_example_unified_energy/delF_{config_str}.npy", delF_values)

    bias_values = np.ctypeslib.as_array(bias_out)
    bias_values = np.reshape(bias_values, (config["num_chains"], config["num_samples"]))
    np.save(f"../../../../output/2d_example_unified_energy/bias_values_{config_str}.npy", bias_values)

    print("Resampling")
    resampled_samples = []
    cutout = 100

    weights = np.exp(bias_values)

    for i in range(config["num_chains"]):
        weights_i = weights[i, cutout:] / sum(weights[i, cutout:])
        indices = np.random.choice(
            np.arange(cutout, len(weights_i) + cutout),
            size=len(weights_i),
            replace=True,
            p=weights_i,
        )
        resampled_samples.append(samples[i, indices, :])
        print(f"Resampled chain {i}")
             

    resampled_samples = np.array(resampled_samples)
    np.save(f"../../../../output/2d_example_unified_energy/resampsamp_{config_str}.npy", resampled_samples)


if __name__ == "__main__":
    main()
