import ctypes
import numpy as np
import os
import json

from define_data_and_variables import HMCConfig, Data, hmc_config, hmc_hash, c_hmc_config, data, c_data, make_dumpable, version
import platform
from pathlib import Path

def main():
    base_dir = Path(__file__).parent 

    sp_dir = base_dir / f"output/{data['dn']}/{version}"
    output_dir = sp_dir / f"{hmc_hash}"
    os.makedirs(output_dir, exist_ok=True)

    # Save hmc_config
    with open(os.path.join(output_dir, "hmc_config.json"), "w") as f:
        dump_hmc_config = make_dumpable(hmc_config)
        json.dump(dump_hmc_config, f, indent=2)

    # Save data
    with open(os.path.join(output_dir, "data.json"), "w") as f:
        dump_data = make_dumpable(data)
        json.dump(dump_data, f, indent=2)

    # Load library depending on OS
    if platform.system() == "Windows":
        os.add_dll_directory("C:/msys64/ucrt64/bin")
        lib = ctypes.CDLL("./hmc.dll")
    else:
        # Linux / macOS
        lib = ctypes.CDLL("./hmc.so")

    lib.hmc.argtypes = [
        ctypes.POINTER(HMCConfig),
        ctypes.POINTER(Data),
        ctypes.POINTER(ctypes.c_double),  # samples_out
        ctypes.POINTER(ctypes.c_double),  # energy_out
    ]

    lib.hmc.restype = None

    total = hmc_config["nch"] * hmc_config["ns"]
    total_times_N = total * data["N"]
    samples_out = (ctypes.c_double * total_times_N)()
    energy_out = (ctypes.c_double * total)()

    lib.hmc(
        ctypes.byref(c_hmc_config),
        ctypes.byref(c_data),
        samples_out,
        energy_out,
    )

    samples = np.ctypeslib.as_array(samples_out)
    samples = np.reshape(samples, (hmc_config["nch"], hmc_config["ns"], data["N"]))
    outdir_samples = output_dir / "samples.npy"
    np.save(outdir_samples, samples)

    energy_values = np.ctypeslib.as_array(energy_out)
    energy_values = np.reshape(energy_values, (hmc_config["nch"], hmc_config["ns"]))
    outdir_energy = output_dir / "energy.npy"
    np.save(outdir_energy, energy_values)

    print("------Done sampling------")

if __name__ == "__main__":
    main()
