import ctypes
import numpy as np
import scipy as sc
import os
from datetime import datetime
import json

from define_data_and_variables import OPESConfig, Data, opes_config, opes_hash, c_opes_config, data, c_data, make_dumpable, version
import platform
from pathlib import Path

def main():
    base_dir = Path(__file__).parent 

    sp_dir = base_dir / f"output/{data['dn']}/{version}"
    output_dir = sp_dir / f"{opes_hash}"
    os.makedirs(output_dir, exist_ok=True)

    # Save opes_config
    with open(os.path.join(output_dir, "opes_config.json"), "w") as f:
        dump_opes_config = make_dumpable(opes_config)
        json.dump(dump_opes_config, f, indent=2)

    # Save data
    with open(os.path.join(output_dir, "data.json"), "w") as f:
        dump_data = make_dumpable(data)
        json.dump(dump_data, f, indent=2)

    # Load library depending on OS
    if platform.system() == "Windows":
        os.add_dll_directory("C:/msys64/ucrt64/bin")
        lib = ctypes.CDLL("./opes.dll")
    else:
        # Linux / macOS
        lib = ctypes.CDLL("./opes.so")

    lib.opes.argtypes = [
        ctypes.POINTER(OPESConfig),
        ctypes.POINTER(Data),
        ctypes.POINTER(ctypes.c_double),  # samples_out
        ctypes.POINTER(ctypes.c_double),  # energy_out
        ctypes.POINTER(ctypes.c_double),  # d18o_energy_out
        ctypes.POINTER(ctypes.c_double),  # bias_out
        ctypes.POINTER(ctypes.c_double),  # df_out
    ]

    lib.opes.restype = None

    total = opes_config["hmcc"]["nch"] * opes_config["hmcc"]["ns"]
    total_times_N = total * data["N"]
    total_times_lambda = total * opes_config["nt"]
    samples_out = (ctypes.c_double * total_times_N)()
    energy_out = (ctypes.c_double * total)()
    d18o_energy_out = (ctypes.c_double * total)()
    bias_out = (ctypes.c_double * total)()
    df_out = (ctypes.c_double * total_times_lambda)()

    lib.opes(
        ctypes.byref(c_opes_config),
        ctypes.byref(c_data),
        samples_out,
        energy_out,
        d18o_energy_out,
        bias_out,
        df_out,
    )

    samples = np.ctypeslib.as_array(samples_out)
    samples = np.reshape(samples, (opes_config["hmcc"]["nch"], opes_config["hmcc"]["ns"], data["N"]))
    outdir_samples = output_dir / "samples.npy"
    np.save(outdir_samples, samples)

    energy_values = np.ctypeslib.as_array(energy_out)
    energy_values = np.reshape(energy_values, (opes_config["hmcc"]["nch"], opes_config["hmcc"]["ns"]))
    outdir_energy = output_dir / "energy.npy"
    np.save(outdir_energy, energy_values)

    d18o_energy_values = np.ctypeslib.as_array(d18o_energy_out)
    d18o_energy_values = np.reshape(d18o_energy_values, (opes_config["hmcc"]["nch"], opes_config["hmcc"]["ns"]))
    outdir_d18o_energy = output_dir / "d18o_energy.npy"
    np.save(outdir_d18o_energy, d18o_energy_values)
    
    bias_values = np.ctypeslib.as_array(bias_out)
    bias_values = np.reshape(bias_values, (opes_config["hmcc"]["nch"], opes_config["hmcc"]["ns"]))
    outdir_bias = output_dir / "bias.npy"
    np.save(outdir_bias, bias_values)

    df_values = np.ctypeslib.as_array(df_out)
    df_values = np.reshape(df_values, (opes_config["hmcc"]["nch"], opes_config["hmcc"]["ns"], opes_config["nt"]))
    outdir_df = output_dir / "df.npy"
    np.save(outdir_df, df_values)

    print("------Done sampling------")

if __name__ == "__main__":
    main()
