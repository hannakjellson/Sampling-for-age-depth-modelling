import ctypes
import numpy as np
import os
from pathlib import Path
import platform
import json
from define_data_and_variables import ADAMConfig, Data, adam_config, data, c_adam_config, c_data, adam_hash, make_dumpable

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

    outdir_samples = output_dir / "adam_samples.npy"
    outdir_energies = output_dir / "adam_energies.npy"
    outdir_d18o_energies = output_dir / "adam_d18o_energies.npy"
    np.save(outdir_samples, samples)
    np.save(outdir_energies, energies)
    np.save(outdir_d18o_energies, d18o_energies)

if __name__ == "__main__":
    main()
