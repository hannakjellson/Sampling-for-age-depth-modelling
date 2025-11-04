import ctypes
import numpy as np
import os
from define_data_and_variables import get_data, get_hmc_config
from pathlib import Path
import platform

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
        ctypes.c_int,  # seed
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

def define_c_types_hmc(lib):
    lib.hmc.argtypes = [
        ctypes.c_int,  # N
        ctypes.c_int,  # ndt
        ctypes.c_int,  # nHMC
        ctypes.c_int,  # nchains
        ctypes.c_int,  # nsamples
        ctypes.c_int,  # nlambda
        ctypes.c_int,  # ntemp
        ctypes.c_int,  # npc
        ctypes.c_int,  # num_c14_depths
        ctypes.c_int,  # num_D18O_depths
        ctypes.c_int,  # num_D18O_reference_times
        ctypes.c_int,  # seed
        ctypes.c_double,  # H
        ctypes.c_double,  # dt
        ctypes.c_double,  # dc
        ctypes.c_double,  # sigma
        ctypes.c_double,  # a
        ctypes.c_double,  # b
        ctypes.c_double,  # theta
        ctypes.c_double,  # dE
        ctypes.c_double,  # startbias
        ctypes.c_double,  # endbias
        ctypes.c_double,  # startbiastemp
        ctypes.c_double,  # endbiastemp
        ctypes.c_double,  # dist
        ctypes.c_double,  # gamma
        ctypes.c_double,  # distance_threshold
        ctypes.c_double,  # cap_energy_scaling
        ctypes.c_double,  # cap_width
        ctypes.POINTER(ctypes.c_double),  # betas
        ctypes.POINTER(ctypes.c_double),  # cs
        ctypes.POINTER(ctypes.c_double),  # pcs
        ctypes.POINTER(ctypes.c_double),  # sp
        ctypes.POINTER(ctypes.c_double),  # sp_mean
        ctypes.POINTER(ctypes.c_double),  # sp_energies
        ctypes.POINTER(ctypes.c_double),  # c14_ages
        ctypes.POINTER(ctypes.c_double),  # c14_depths
        ctypes.POINTER(ctypes.c_double),  # c14_sigma
        ctypes.POINTER(ctypes.c_double),  # D18O
        ctypes.POINTER(ctypes.c_double),  # D18O_depths
        ctypes.POINTER(ctypes.c_double),  # D18O_sigma
        ctypes.POINTER(ctypes.c_double),  # D18O_reference
        ctypes.POINTER(ctypes.c_double),  # D18O_reference_times
        ctypes.POINTER(ctypes.c_double),  # samples_out
        ctypes.POINTER(ctypes.c_double),  # energy_out
        ctypes.POINTER(ctypes.c_double),  # bias_out
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_bool,
    ]

    lib.hmc.restype = None

    return lib


def main():
    data_name = "dayu06"
    data = get_data(data_name)
    config, config_str = get_hmc_config(find_min = True)
    config = {k: (float("nan") if v is None else v) for k, v in config.items()}
    path = Path(__file__).resolve().parent / f"../../../../output/{data_name}/{config_str}"
    path = path.resolve()
    os.makedirs(path, exist_ok=True)
    
    # Load library depending on OS
    if platform.system() == "Windows":
        os.add_dll_directory("C:/msys64/ucrt64/bin")
        lib = ctypes.CDLL("./adams.dll")
        lib_hmc = ctypes.CDLL("./hmc.dll")
    else:
        # Linux / macOS
        lib = ctypes.CDLL("./adams.so")
        lib_hmc = ctypes.CDLL("./hmc.so")
    lib = define_c_types(lib)
    lib_hmc = define_c_types_hmc(lib_hmc)

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
        ctypes.c_int(config["sd"]),
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
        ctypes.c_double(config["adt"]),
        Eout,
        ctypes.c_double(config["gl"]),
        samples_out,
    )

    # Convert outputs to numpy
    energies = np.ctypeslib.as_array(Eout)
    samples = np.ctypeslib.as_array(samples_out).reshape(config["nlsp"], config["N"])
    print(energies)

    outdir_samples = path / "samples_min.npy"
    outdir_Emin = path / "Emin.npy"
    np.save(outdir_samples, samples)
    np.save(outdir_Emin, energies)

    pcs = np.zeros(1)
    pcs = np.ascontiguousarray(pcs)

    betas = np.zeros(1)
    betas = np.ascontiguousarray(betas)
    
    sp_mean = np.mean(samples, axis = 0)

    total = config["nlsp"] * config["ns"]
    total_times_N = total * config["N"]
    samples_out = (ctypes.c_double * total_times_N)()
    energy_out = (ctypes.c_double * total)()
    bias_out = (ctypes.c_double * total)()
    config_str_input = config_str.encode("utf-8")
    config_find_min_str_input = ''.encode("utf-8")
    data_name_input = ''.encode("utf-8")

    lib_hmc.hmc(
        ctypes.c_int(config["N"]),
        ctypes.c_int(config["ndt"]),
        ctypes.c_int(config["nHMC"]),
        ctypes.c_int(config["nlsp"]),
        ctypes.c_int(config["ns"]),
        ctypes.c_int(config["nl"] if type(config["nl"]) == int else -1), # if its nan its intepreted as a float and yields error
        ctypes.c_int(config["nt"] if type(config["nt"]) == int else -1),
        ctypes.c_int(config["npc"] if type(config["npc"]) == int else -1),
        ctypes.c_int(data["num_c14_depths"]),
        ctypes.c_int(data["num_D18O_depths"]),
        ctypes.c_int(data["num_D18O_reference_times"]),
        ctypes.c_int(config["sd"]),
        ctypes.c_double(config["H"]),
        ctypes.c_double(config["dt"]),
        ctypes.c_double(config["dc"]),
        ctypes.c_double(config["s"]),
        ctypes.c_double(config["a"]),
        ctypes.c_double(config["b"]),
        ctypes.c_double(data["theta"]),
        ctypes.c_double(config["dE"]),
        ctypes.c_double(config["sb"]),
        ctypes.c_double(config["eb"]),
        ctypes.c_double(config["sbt"]),
        ctypes.c_double(config["ebt"]),
        ctypes.c_double(config["d"]),
        ctypes.c_double(config["g"]),
        ctypes.c_double(config["thr"]),
        ctypes.c_double(config["ces"]),
        ctypes.c_double(config["cw"]),
        betas.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        cs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        pcs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        samples.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sp_mean.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        energies.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        c14_ages.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        c14_depths.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        c14_sigma.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        D18O.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        D18O_depths.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        D18O_sigma.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        D18O_reference.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        D18O_reference_times.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        samples_out,
        energy_out,
        bias_out,
        config_str_input,
        config_find_min_str_input,
        data_name_input,
        ctypes.c_bool(False),
    )

    # Convert outputs to numpy
    energies = np.ctypeslib.as_array(energy_out).reshape(config["nlsp"], config["ns"])
    samples = np.ctypeslib.as_array(samples_out).reshape(config["nlsp"], config["ns"], config["N"])

    outdir_Emin = path / "start_energies.npy"
    outdir_samples = path / "start_samples.npy"
    np.save(outdir_Emin, energies)
    np.save(outdir_samples, samples)


    


if __name__ == "__main__":
    main()
