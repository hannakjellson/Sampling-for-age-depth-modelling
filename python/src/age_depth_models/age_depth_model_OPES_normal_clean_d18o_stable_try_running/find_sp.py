import ctypes
import numpy as np
import os
from pathlib import Path
import platform
import json
from define_data_and_variables import ADAMConfig, Data, adam_config, data, c_adam_config, c_data, adam_hash, make_dumpable
import matplotlib.pyplot as plt

def main():
    base_dir = Path(__file__).parent 
    output_dir = base_dir / f"output/{data['dn']}/{adam_hash}"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "adam_config.json"), "w") as f:
        dump_adam_config = make_dumpable(adam_config)
        json.dump(dump_adam_config, f, indent=2, skipkeys=True)

    with open(os.path.join(output_dir, "data_config.json"), "w") as f:
        dump_data = make_dumpable(data)
        json.dump(dump_data, f, indent=2, skipkeys=True)

    # Load library depending on OS
    if platform.system() == "Windows":
        os.add_dll_directory("C:/msys64/ucrt64/bin")
        lib = ctypes.CDLL("./adams.dll")
    else:
        # Linux / macOS
        lib = ctypes.CDLL("./adams.so")

    lib.adams.argtypes = [
        ctypes.POINTER(ADAMConfig),
        ctypes.POINTER(Data),
        ctypes.POINTER(ctypes.c_double),  # energy_out
        ctypes.POINTER(ctypes.c_double),  # d18o_energy_out
        ctypes.POINTER(ctypes.c_double),  # samples_out
    ]

    lib.adams.restype = None

    # Outputs
    energy_out = (ctypes.c_double * adam_config["nsp"])() 
    d18o_energy_out = (ctypes.c_double * adam_config["nsp"])() 
    len_samples = adam_config["nsp"] * data["N"]
    samples_out = (ctypes.c_double * len_samples)()

    # Call the C function
    lib.adams(
        ctypes.byref(c_adam_config),
        ctypes.byref(c_data), 
        energy_out,
        d18o_energy_out, 
        samples_out,
    )

    # Convert outputs to numpy
    energies = np.ctypeslib.as_array(energy_out)
    d18o_energies = np.ctypeslib.as_array(d18o_energy_out)
    samples = np.ctypeslib.as_array(samples_out).reshape(adam_config["nsp"], data["N"])
    print(np.min(d18o_energies))
    plt.figure()
    plt.plot(d18o_energies)
    plt.show()

    cumulative_sums = np.cumsum(samples, axis=1) * data["dc"]
    zeros = np.zeros((*samples.shape[:1], 1))
    age_offsets = np.concatenate((zeros, cumulative_sums), axis=1)
    print(np.shape(age_offsets))

    model_ages = data["th"] - age_offsets

    # Age depth trace plot
    fig, ax = plt.subplots(figsize=(6, 4))
    num_samples = len(samples)
    for j in range(num_samples):
        ax.plot(
            data["cs"],
            model_ages[j, :],
            color='green',
            alpha=1,
            linewidth=1,
        )
    for i, c in enumerate(data["cs"]):
        ax.axvline(x=c, color='k', linestyle='--', linewidth = 0.1)

    ax.set_xlim(0, np.max(data["cs"]))
    ax.set_ylim(np.min(data["d18ort"]), np.max(data["d18ort"]))
    plt.plot(data["c14d"], np.squeeze(data["c14"]), "ko", markersize=4)
    # plt.plot(config["cs"], data["theta"] - config["dc"] * np.cumsum((config["a"]/config["b"]) * np.ones_like(config["cs"])), color = 'k', linewidth=0.5)
    plt.xlabel("Depth")
    plt.ylabel("Age")
    plt.show()


    outdir_samples = output_dir / "adam_samples.npy"
    outdir_energies = output_dir / "adam_energies.npy"
    outdir_d18o_energies = output_dir / "adam_d18o_energies.npy"
    np.save(outdir_samples, samples)
    np.save(outdir_energies, energies)
    np.save(outdir_d18o_energies, d18o_energies)

if __name__ == "__main__":
    main()
