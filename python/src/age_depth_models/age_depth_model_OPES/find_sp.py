import ctypes
import numpy as np
import os
from define_data_and_variables import get_data, get_hmc_config
from pathlib import Path


def define_c_types(lib):
    lib.adams.argtypes = [
        ctypes.c_int,  # N
        ctypes.c_double,  # delta_c
        ctypes.POINTER(ctypes.c_double),  # cs
        ctypes.c_double,  # a
        ctypes.c_double,  # b
        ctypes.c_double,  # theta
        ctypes.c_int,  # num_c14_depths
        ctypes.c_int,  # num_D18O_depths
        ctypes.c_int,  # num_D18O_reference_times
        ctypes.POINTER(ctypes.c_double),  # c14_ages
        ctypes.POINTER(ctypes.c_double),  # c14_depths
        ctypes.POINTER(ctypes.c_double),  # c14_sigma
        ctypes.POINTER(ctypes.c_double),  # D18O
        ctypes.POINTER(ctypes.c_double),  # D18O_depths
        ctypes.POINTER(ctypes.c_double),  # D18O_sigma
        ctypes.POINTER(ctypes.c_double),  # D18O_reference
        ctypes.POINTER(ctypes.c_double),  # D18O_reference_times
        ctypes.c_int,  # num_local_sp
        ctypes.c_int,  # max_iter
        ctypes.c_double,  # stepsize
        ctypes.POINTER(ctypes.c_double),  # Eout
        ctypes.c_double,  # grad_lim
        ctypes.POINTER(ctypes.c_double),  # samples_out
    ]
    lib.adams.restype = None
    return lib


def main():
    data = get_data()
    config, config_str = get_hmc_config(find_min = True)

    os.add_dll_directory("C:/msys64/ucrt64/bin")
    lib = ctypes.CDLL("./adams.dll")
    lib = define_c_types(lib)

    cs = np.ascontiguousarray(config["cs"], dtype=np.float64)

    c14_ages = np.ascontiguousarray(data["c14_ages"], dtype=np.float64)
    c14_depths = np.ascontiguousarray(data["c14_depths"], dtype=np.float64)
    c14_sigma = np.ascontiguousarray(data["c14_sigma"], dtype=np.float64)

    D18O = np.ascontiguousarray(data["d18O"], dtype=np.float64)
    D18O_depths = np.ascontiguousarray(data["d18O_depths"], dtype=np.float64)
    D18O_sigma = np.ascontiguousarray(data["d18O_sigma"], dtype=np.float64)
    D18O_reference = np.ascontiguousarray(data["d18O_reference"], dtype=np.float64)
    D18O_reference_times = np.ascontiguousarray(
        data["d18O_reference_times"], dtype=np.float64
    )

    # Outputs
    Eout = (ctypes.c_double * config["nlsp"])() 
    len_samples = config["nlsp"] * config["N"]
    samples_out = (ctypes.c_double * len_samples)()

    # Call the C function
    lib.adams(
        ctypes.c_int(config["N"]),
        ctypes.c_double(config["dc"]),
        cs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(config["a"]),
        ctypes.c_double(config["b"]),
        ctypes.c_double(data["theta"]),
        ctypes.c_int(len(c14_depths)),
        ctypes.c_int(len(D18O_depths)),
        ctypes.c_int(len(D18O_reference_times)),
        c14_ages.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        c14_depths.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        c14_sigma.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        D18O.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        D18O_depths.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        D18O_sigma.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        D18O_reference.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        D18O_reference_times.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(config["nlsp"]),
        ctypes.c_int(config["mi"]),
        ctypes.c_double(config["dt"]),
        Eout,
        ctypes.c_double(config["gl"]),
        samples_out,
    )

    # Convert outputs to numpy
    energies = np.ctypeslib.as_array(Eout)
    samples = np.ctypeslib.as_array(samples_out).reshape(config["nlsp"], config["N"])

    outdir_Emin = Path(__file__).resolve().parent / "../../../../output/age_depth_OPES" / f"Emin_{config_str}.npy"
    outdir_samples = Path(__file__).resolve().parent / "../../../../output/age_depth_OPES" / f"samples_min_{config_str}.npy"
    np.save(outdir_Emin, energies)
    np.save(outdir_samples, samples)


    


if __name__ == "__main__":
    main()
