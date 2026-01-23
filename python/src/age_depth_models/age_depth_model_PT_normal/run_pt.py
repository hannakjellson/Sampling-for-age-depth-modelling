import ctypes
import numpy as np
import scipy as sc
import os
from datetime import datetime
import json

from define_data_and_variables import HMCConfig, PTConfig, Data, pt_config, data, c_data, adam_hash, dict_to_struct, make_dumpable, hash_configs
import platform
from pathlib import Path
import re
from scipy.spatial.distance import cdist

def main():
    base_dir = Path(__file__).parent 

    adam_dir = base_dir / f"output/{data['dn']}/{adam_hash}"

    sp = np.load(adam_dir / "adam_samples.npy")
    energies = np.load(adam_dir / "adam_energies.npy")

    indices = np.argsort(energies)
    sp = sp[indices][:pt_config["nt"]]
    pt_config["hmcc"]["sp"] = np.ascontiguousarray(sp)

    pt_hash = hash_configs(pt_config, data)
    output_dir = adam_dir / f"{pt_hash}"
    os.makedirs(output_dir, exist_ok=True)

    c_hmc_config = dict_to_struct(
        pt_config["hmcc"], HMCConfig
    )

    c_pt_config = pt_config.copy()
    c_pt_config["hmcc"] = ctypes.pointer(c_hmc_config)

    c_pt_config = dict_to_struct(
        c_pt_config, PTConfig
    )

    # Save pt_config
    with open(os.path.join(output_dir, "pt_config.json"), "w") as f:
        dump_pt_config = make_dumpable(pt_config)
        json.dump(dump_pt_config, f, indent=2)


    #### Running Adams

    # Load library depending on OS
    if platform.system() == "Windows":
        os.add_dll_directory("C:/msys64/ucrt64/bin")
        lib = ctypes.CDLL("./pt.dll")
    else:
        # Linux / macOS
        lib = ctypes.CDLL("./pt.so")

    lib.pt.argtypes = [
        ctypes.POINTER(PTConfig),
        ctypes.POINTER(Data),
        ctypes.POINTER(ctypes.c_double),  # samples_out
        ctypes.POINTER(ctypes.c_double),  # energy_out
        ctypes.POINTER(ctypes.c_double),  # d18o_energy_out
    ]

    lib.pt.restype = None

    total = pt_config["nt"] * pt_config["hmcc"]["ns"]
    total_times_N = total * data["N"]
    samples_out = (ctypes.c_double * total_times_N)()
    energy_out = (ctypes.c_double * total)()
    d18o_energy_out = (ctypes.c_double * total)()

    lib.pt(
        ctypes.byref(c_pt_config),
        ctypes.byref(c_data),
        samples_out,
        energy_out,
        d18o_energy_out,
    )

    samples = np.ctypeslib.as_array(samples_out)
    samples = np.reshape(samples, (pt_config["nt"], pt_config["hmcc"]["ns"], data["N"]))
    outdir_samples = output_dir / "samples.npy"
    np.save(outdir_samples, samples)

    energy_values = np.ctypeslib.as_array(energy_out)
    energy_values = np.reshape(energy_values, (pt_config["nt"], pt_config["hmcc"]["ns"]))
    outdir_energy = output_dir / "energy.npy"
    np.save(outdir_energy, energy_values)

    d18o_energy_values = np.ctypeslib.as_array(d18o_energy_out)
    d18o_energy_values = np.reshape(d18o_energy_values, (pt_config["nt"], pt_config["hmcc"]["ns"]))
    outdir_d18o_energy = output_dir / "d18o_energy.npy"
    np.save(outdir_d18o_energy, d18o_energy_values)

    print("------Done sampling------")

if __name__ == "__main__":
    main()
