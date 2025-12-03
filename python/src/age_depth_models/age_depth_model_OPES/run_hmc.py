import ctypes
import numpy as np
import scipy as sc
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
        ctypes.c_double,  # beta
        ctypes.c_double,  # dE
        ctypes.c_double,  # startbias
        ctypes.c_double,  # endbias
        ctypes.c_double,  # startbiastemp
        ctypes.c_double,  # endbiastemp
        ctypes.c_double,  # dist
        ctypes.c_double,  # cap_energy_scaling
        ctypes.c_double,  # energy_exp
        ctypes.c_int,  # delta_F start update
        ctypes.c_int,  # energies ->len(delta_F_nominator_start)
        ctypes.POINTER(ctypes.c_double),  # delta_F_nominator_start
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
    temp = True
    cap = False
    umbrella = False
    config, config_str = get_hmc_config(find_min = False, bias = "unified", cap = cap, temp = temp, umbrella = umbrella)
    config = {k: (float("nan") if v is None else v) for k, v in config.items()}

    data_name = "dayu06"
    data = get_data(data_name)
    np.random.seed(config["sd"])

    config_find_min, config_find_min_str = get_hmc_config(find_min = True)
    config_find_min = {k: (float("nan") if v is None else v) for k, v in config_find_min.items()}

    mid_name = config_find_min_str if not config["rsp"] else f"seed{config['sd']}"
    
    base_dir = Path(__file__).parent 
    output_dir = base_dir / f"../../../../output/{data_name}/{mid_name}/{config_str}"
    output_dir = output_dir.resolve()
    os.makedirs(output_dir, exist_ok=True)

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

    energies = Path(base_dir / f"../../../../output/{data_name}/{config_find_min_str}/Emin.npy").resolve()
    energies = np.load(energies)

    sp = Path(base_dir / f"../../../../output/{data_name}/{config_find_min_str}/samples_min.npy").resolve()
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
    
    betas = [0]
    betas = np.ascontiguousarray(betas)

    # Starting points and energies
    if not np.isnan(config["rsp"]) and config["rsp"]:
        sp = np.random.gamma(config["a"], scale=1/config["b"], size=(config["nch"], config["N"]))
        sp_energies = np.zeros_like(energies)[:config["nch"]] # Doesnt make sense to run with random starting points and cap anyway.
        output_dir = base_dir / f"seed{config['sd']}/{config_str}"
        np.save(Path(base_dir / f"../../../../output/{data_name}/seed{config['sd']}" / "starting_points.npy").resolve(), sp)
    else:
        if not np.isnan(config_find_min["hmc"]) and config_find_min["hmc"]:
            outdir_startE = Path(__file__).resolve().parent / "../../../../output" / f"{data_name}" / f"{config_find_min_str}" / "start_energies.npy"
            outdir_start_samples = Path(__file__).resolve().parent / "../../../../output" / f"{data_name}" / f"{config_find_min_str}" / "start_samples.npy"
            
            sp_energies = np.load(outdir_startE)
            sp = np.load(outdir_start_samples)

            flat_E = sp_energies.flatten()
            energy_exp= np.min(flat_E)
            # energy_exp = np.mean(flat_E)
            print(energy_exp)
            flat_samples = sp.reshape(config_find_min["nlsp"]*config_find_min["ns"], -1)
            
            if not np.isnan(config["unb"]) and config["unb"]:
                def n_eff(beta):
                    weights = np.exp(-(beta - 1)*flat_E)
                    return np.sum(weights)**2 - 0.5 * len(weights) * np.sum(weights**2)
                sol_neg= abs(sc.optimize.brentq(n_eff, 1/config["ebt"], 1/config["sbt"])-1)
                config["nt"] = int((1/config["sbt"] - 1/config["ebt"])/sol_neg)
                betas = np.linspace(1/config["ebt"], 1/config["sbt"], config["nt"])
                betas = np.ascontiguousarray(betas)
                np.save(base_dir / f"../../../../output/{data_name}/{config_find_min_str}/temps_sbt{config['sbt']}_ebt{config['ebt']}.npy", 1/betas)
                
            # sp_new = []
            # sp_energies_new = []
            np.random.seed(config["sd"])

            # for i in range(config["nch"]):
            #     flat_E = sp_energies[i, :]
            #     flat_samples = sp[i, :, :]

            q25, q75 = np.percentile(flat_E, [75, 100])
            candidate_mask = (flat_E >= q25) & (flat_E <= q75)
            candidate_E = flat_E[candidate_mask]
            candidate_samples = flat_samples[candidate_mask, :]
            indices = np.random.choice(len(candidate_samples), size=config["nch"], replace=False)
            sp = candidate_samples[indices, :]
            sp_energies = candidate_E[indices]
            # sp_new.append(sp_chain)
            # sp_energies_new.append(sp_energies_chain)
            sp = np.array(sp)
            sp_energies = np.array(sp_energies)
            print(sp_energies)
            # sp = sp[energy_idx, :]
            # energies = sp_energies[energy_idx]
            # sp_energies = energies[:config["nch"]]
            # sp = sp[:config["nch"], :]

        else:
            sp = sp[energy_idx, :]
            energies = energies[energy_idx]
            sp_energies = energies[:config["nch"]]
            sp = sp[:config["nch"], :]
    # print(sp_energies)
    # print(sp)

    if not config["unb"]:
        temps = np.empty((1))
        if not np.isnan(config["ai"]) and config["ai"]:
            outdir_start_temps = Path(__file__).resolve().parent / "../../../../output" / f"{data_name}" / f"start_Ts.npy"
            temps = np.load(outdir_start_temps)
            config["nt"] = len(temps)
        elif temp and not config["ai"] and not config["unb"]:
            factor = np.linspace(0, config["nt"] - 1, config["nt"]) / (config["nt"] - 1) if config["nt"] != 1 else 0
            if not np.isnan(config["ut"]) and config["ut"]:
                temps = config["sbt"] + (factor * (config["ebt"] - config["sbt"]))
            else:
                temps = config["sbt"] * ((config["ebt"] / config["sbt"])**factor)
        elif not temp:
            temps = np.zeros(1)
        betas = 1/temps
        betas = np.ascontiguousarray(betas)

    q0, q25 = np.percentile(flat_E, [0, 75])
    candidate_mask = (flat_E >= q0) & (flat_E <= q25)
    flat_E = flat_E[candidate_mask]
    print(np.min(flat_E))
    thousand_energies = flat_E[::int(len(flat_E) / 100)]
    print(thousand_energies)
    delta_F_nominator = np.ascontiguousarray(np.sum(np.exp(-(betas[None, :] - betas[0])*thousand_energies[:, None]), axis = 0))
    print(-np.log(delta_F_nominator / len(thousand_energies)))
    total = config["nch"] * config["ns"]
    total_times_N = total * config["N"]
    samples_out = (ctypes.c_double * total_times_N)()
    energy_out = (ctypes.c_double * total)()
    bias_out = (ctypes.c_double * total)()
    config_str_input = config_str.encode("utf-8")
    config_find_min_str_input = config_find_min_str.encode("utf-8")
    data_name_input = data_name.encode("utf-8")

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
        ctypes.c_double(config["bt"] if not np.isnan(config["bt"]) else 1.0),
        ctypes.c_double(config["dE"]),
        ctypes.c_double(config["sb"]),
        ctypes.c_double(config["eb"]),
        ctypes.c_double(config["sbt"]),
        ctypes.c_double(config["ebt"]),
        ctypes.c_double(config["d"]),
        ctypes.c_double(config["ces"]),
        ctypes.c_double(energy_exp),
        ctypes.c_int(config["dfs"]),
        ctypes.c_int(len(thousand_energies)), 
        delta_F_nominator.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(config["cw"]),
        betas.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
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
        config_find_min_str_input,
        data_name_input,
        ctypes.c_bool(config["shb"])
    )

    samples = np.ctypeslib.as_array(samples_out)
    samples = np.reshape(samples, (config["nch"], config["ns"], config["N"]))
    outdir_samples = output_dir / "samples.npy"
    np.save(outdir_samples, samples)

    energy_values = np.ctypeslib.as_array(energy_out)
    energy_values = np.reshape(energy_values, (config["nch"], config["ns"]))
    outdir_energy = output_dir / "energy.npy"
    np.save(outdir_energy, energy_values)
    
    bias_values = np.ctypeslib.as_array(bias_out)
    bias_values = np.reshape(bias_values, (config["nch"], config["ns"]))
    outdir_bias = output_dir / "bias.npy"
    np.save(outdir_bias, bias_values)

    print("Resampling\n")
    resampled_samples = []
    cutout = config["co"]

    weights = np.exp(bias_values)
    for i in range(config["nch"]):
        if not config["shb"]:
            weights_i = weights[i, cutout:] / sum(weights[i, cutout:])
        else:
            weights_i = weights[i, cutout:] / sum(weights[:, cutout:])
            weights_i /= sum(weights_i)
        indices = np.random.choice(
            np.arange(cutout, len(weights_i) + cutout),
            size=int(len(weights_i)),
            replace=True,
            p=weights_i,
        )
        print(f"Chain {i} done resampling")
        resampled_samples.append(samples[i, indices, :])
            
    resampled_samples = np.array(resampled_samples)
    outdir_resamp = output_dir / "resamp.npy"
    np.save(outdir_resamp, resampled_samples)


if __name__ == "__main__":
    main()
