import ctypes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from define_data_and_variables import get_data, get_hmc_config
import datetime as datetime
import platform
from pathlib import Path


def define_c_types(lib):
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
        ctypes.c_bool,  # uniform_temp
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
    ]

    lib.hmc.restype = None

    return lib


def main():
    data = get_data()
    config, config_str = get_hmc_config(find_min = False, bias = "unified", cap = False, temp = True, umbrella = False)
    config = {k: (float("nan") if v is None else v) for k, v in config.items()}
    np.random.seed(config["sd"])

    config_find_min, config_find_min_str = get_hmc_config(find_min = True)
    config_find_min = {k: (float("nan") if v is None else v) for k, v in config_find_min.items()}

    # Load library depending on OS
    if platform.system() == "Windows":
        os.add_dll_directory("C:/msys64/ucrt64/bin")
        lib = ctypes.CDLL("./hmc.dll")
    else:
        # Linux / macOS
        lib = ctypes.CDLL("./hmc.so")
    lib = define_c_types(lib)

    cs = np.ascontiguousarray(config["cs"], dtype=np.float64)

    c14_ages = np.ascontiguousarray(data["c14_ages"], dtype=np.float64)
    c14_depths = np.ascontiguousarray(data["c14_depths"], dtype=np.float64)
    c14_sigma = np.ascontiguousarray(data["c14_sigma"], dtype=np.float64)

    D18O = np.ascontiguousarray(data["d18O"], dtype=np.float64)
    D18O_depths = np.ascontiguousarray(data["d18O_depths"], dtype=np.float64)
    D18O_sigma = np.ascontiguousarray(data["d18O_sigma"], dtype=np.float64)
    D18O_reference_times = np.ascontiguousarray(
        data["d18O_reference_times"], dtype=np.float64
    )
    D18O_reference = np.ascontiguousarray(data["d18O_reference"], dtype=np.float64)

    energies = Path(Path(__file__).resolve().parent / f"../../../../output/age_depth_OPES/Emin_{config_find_min_str}.npy").resolve()
    energies = np.load(energies)

    sp = Path(__file__).resolve().parent / f"../../../../output/age_depth_OPES/samples_min_{config_find_min_str}.npy"
    sp = np.load(sp)
    
    energy_idx = np.argsort(energies) # might bug if there are not enough starting points.
    sp_smallest = sp[energy_idx][:20]
    sp_mean= np.mean(sp_smallest, axis = 0)
    sp_centered = sp_smallest-sp_mean
    cov = np.cov(sp_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Eigenvalues and eigenvectors
    eig_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eig_idx]
    eigenvectors = eigenvectors[:, eig_idx]

    pcs = eigenvectors[:, :config["npc"]] if not np.isnan(config["npc"]) else eigenvectors[:, 0]
    pcs = np.ascontiguousarray(pcs.T) if not np.isnan(config["npc"]) else np.ascontiguousarray(eigenvectors[:, 0].T)
    
    # Starting points and energies
    if config["rsp"]:
        sp = np.random.gamma(config["a"], scale=1/config["b"], size=(config["nch"], config["N"]))
        sp_energies = np.zeros_like(energies)[:config["nch"]] # Doesnt make sense to run with random starting points and cap anyway.
    else:
        if(config["hmc"]):
            outdir_startE = Path(__file__).resolve().parent / "../../../../output/age_depth_OPES" / f"start_energies_{config_find_min_str}.npy"
            outdir_start_samples = Path(__file__).resolve().parent / "../../../../output/age_depth_OPES" / f"start_samples_{config_find_min_str}.npy"
            
            sp_energies = np.load(outdir_startE)
            sp = np.load(outdir_start_samples)

            flat_E = sp_energies.flatten()
            flat_samples = sp.reshape(config_find_min["nlsp"]*config_find_min["ns"], -1)
            
            q25, q75 = np.percentile(flat_E, [25, 75])
            candidate_mask = (flat_E >= q25) & (flat_E <= q75)
            candidate_E = flat_E[candidate_mask]
            candidate_samples = flat_samples[candidate_mask]
            indices = np.random.choice(len(candidate_samples), size=config["nch"], replace=False)
            sp = candidate_samples[indices]
            sp_energies = candidate_E[indices]
            print(sp_energies)
        else:
            sp = sp[energy_idx, :]
            energies = energies[energy_idx]
            sp_energies = energies[:config["nch"]]
            sp = sp[:config["nch"], :]
    print(sp_energies)
    print(sp)

    total = config["nch"] * config["ns"]
    total_times_N = total * config["N"]
    samples_out = (ctypes.c_double * total_times_N)()
    energy_out = (ctypes.c_double * total)()
    bias_out = (ctypes.c_double * total)()
    config_str_input = config_str.encode("utf-8")

    lib.hmc(
        ctypes.c_int(config["N"]),
        ctypes.c_int(config["ndt"]),
        ctypes.c_int(config["nHMC"]),
        ctypes.c_int(config["nch"]),
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
        ctypes.c_bool(config["ut"]),
        cs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        pcs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sp_mean.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sp_energies.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
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
    )

    samples = np.ctypeslib.as_array(samples_out)
    samples = np.reshape(samples, (config["nch"], config["ns"], config["N"]))
    outdir_samples = Path(__file__).resolve().parent / "../../../../output/age_depth_OPES" / f"samples_{config_str}.npy"
    np.save(outdir_samples, samples)

    energy_values = np.ctypeslib.as_array(energy_out)
    energy_values = np.reshape(energy_values, (config["nch"], config["ns"]))
    outdir_energy = Path(__file__).resolve().parent / "../../../../output/age_depth_OPES" / f"energy_{config_str}.npy"
    np.save(outdir_energy, energy_values)
    
    bias_values = np.ctypeslib.as_array(bias_out)
    bias_values = np.reshape(bias_values, (config["nch"], config["ns"]))
    outdir_bias = Path(__file__).resolve().parent / "../../../../output/age_depth_OPES" / f"bias_{config_str}.npy"
    np.save(outdir_bias, bias_values)

    print("Resampling\n")
    resampled_samples = []
    cutout = config["co"]

    weights = np.exp(bias_values)
    for i in range(config["nch"]):
        weights_i = weights[i, cutout:] / sum(weights[i, cutout:])
        indices = np.random.choice(
            np.arange(cutout, len(weights_i) + cutout),
            size=int(len(weights_i)),
            replace=True,
            p=weights_i,
        )
        print(f"Chain {i} done resampling")
        resampled_samples.append(samples[i, indices, :])
            

    resampled_samples = np.array(resampled_samples)
    outdir_resamp = Path(__file__).resolve().parent / "../../../../output/age_depth_OPES" / f"resamp_{config_str}.npy"
    np.save(outdir_resamp, resampled_samples)


if __name__ == "__main__":
    main()
