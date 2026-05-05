import ctypes
import numpy as np
import os
import json

from define_data_and_variables import HMCConfig, Data, hmc_config, c_hmc_config, data, c_data, make_dumpable
import platform
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd

def main():
    base_path = "../../../../data/"

    output_dir = os.path.join(base_path, data["dn"])
    os.makedirs(output_dir, exist_ok=True)

    # Save hmc_config
    # with open(os.path.join(output_dir, "hmc_config.json"), "w") as f:
    #     dump_hmc_config = make_dumpable(hmc_config)
        # json.dump(dump_hmc_config, f, indent=2)
    # Save data
    # with open(os.path.join(output_dir, "data.json"), "w") as f:
    #     dump_data = make_dumpable(data)
    #     json.dump(dump_data, f, indent=2)

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
    outdir_samples = f"{output_dir}/samples.npy"
    # np.save(outdir_samples, samples)

    energy_values = np.ctypeslib.as_array(energy_out)
    energy_values = np.reshape(energy_values, (hmc_config["nch"], hmc_config["ns"]))
    outdir_energy =  f"{output_dir}/energy.npy"
    # np.save(outdir_energy, energy_values)
    print("------Done sampling------")

    print("------Generating 3 sets of reference data and plotting------")
    # Labels
    c14_label = r"$^{230}$Th" if "d18o" in data["dn"] else r"Tephra Layer"
    d18o_label = "$\\delta^{18}$O [‰]" if "d18o" in data["dn"] else "inc [°]" if "inc" in data["dn"] else "dec [°]"

    # Computing histograms of all samples for plotting
    chain = 4
    X = samples[chain].T
    D, N = X.shape

    age = np.vstack([
        np.ones((1, N)) * data["th"],
        data["th"] - data["dc"] * np.cumsum(X, axis=0)
    ])

    norm_age = age

    # Time bins
    dt = 1 if np.max(data["d18ort"]) - np.min(data["d18ort"]) < 2000 else 10
    t_edges = np.arange(np.min(data["d18ort"]), np.max(data["d18ort"]) + dt, dt)
    t = t_edges[1:] - dt / 2
    T = t.size

    # Depth interpolation grid
    dz = data["H"] / 1000
    z = np.arange(0, data["dc"] * D + dz, dz)
    Z = z.size

    # Allocate result
    C = np.full((Z, T), np.nan)
    A = np.array([
        np.interp(z, data["cs"], norm_age[:, j])
        for j in range(N)
    ]).T

    for j in range(Z):
        C[j, :], _ = np.histogram(A[j, :], bins=t_edges, density = True)

    C /= (np.sum(C, axis=1, keepdims=True) + 1e-12)
    C[C<=0] = np.nan

    # Prior mean and std for plotting
    prior_mean = data["th"] - data["cs"] * np.exp(data["pm"] + (data["ps"]**2 / 2))
    prior_2_std = 2 * np.sqrt(data["cs"] * data["dc"] * (np.exp(data["ps"]**2)-1) * np.exp(2*data["pm"] + data["ps"]**2))
    
    rng = np.random.default_rng(seed=hmc_config["sd"])
    model_ages_list = []
    cutout = 1000
    samples = samples[:, cutout:, :]
    flat_samples = samples.reshape(-1, samples.shape[-1])

    for i in range(3):
        random_idx = rng.integers(0, flat_samples.shape[0])

        sample= flat_samples[random_idx]

        cumulative_sums = np.cumsum(sample) * data["dc"]
        zeros = np.zeros((*sample.shape[:1], 1))
        age_offset = np.concatenate((np.array([0]), cumulative_sums))
        model_age = data["th"] - age_offset
        model_ages_list.append(model_age)

        depths_interp = np.interp(data["d18ort"], model_age[::-1], data["cs"][::-1], right = np.nan, left = np.nan)[::-1]
        d18O_actual = np.interp(data["d18od"], depths_interp, data["d18or"][::-1], right = np.nan, left = np.nan) + rng.normal(loc=0, scale=data["d18os"], size=len(data["d18od"]))
        years_actual = np.interp(data["d18od"], data["cs"], model_age, right = np.nan, left = np.nan)
        years_c14_actual = np.interp(data["c14d"],data["cs"], model_age, right = np.nan, left = np.nan)

        init_nan = np.full(1, np.nan)
        init_nans_c14 = np.full(1 + len(data["c14d"]), np.nan)
        extra_nans = np.full(len(data["d18od"]), np.nan)

        depths = np.concatenate((np.array([0]), data["c14d"], data["d18od"]))
        true_ages = np.concatenate((np.array([data["th"]]), years_c14_actual, years_actual))
        cal_c14_ages = np.concatenate((init_nan, data["c14"], extra_nans))
        sigma_age = np.concatenate((init_nan, data["c14s"], extra_nans))
        d18O = np.concatenate((init_nans_c14, d18O_actual))
        sigma_d18O = np.concatenate((init_nans_c14, data["d18os"]))

        df = pd.DataFrame({
            "depth": depths,
            "true_age": true_ages,
            "age": cal_c14_ages,
            "sigma_age": sigma_age,
            "ref": d18O,
            "sigma_ref": sigma_d18O
        }) 

        file_path = Path(output_dir) / f"{i}" / "data.txt"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, sep="\t", index=False)

        # np.save(f"{output_dir}/{i}/true_sample.npy", sample)


        # --------Plotting generated age-depth model and samples----------
        fig, ax = plt.subplots(figsize=(6, 4))
        plt.set_cmap(plt.cm.Greys)

        # samples
        im = ax.pcolormesh(z, t, C.T, norm=colors.LogNorm(1e-4, 1))

        cbar = plt.colorbar(im)
        cbar.set_label("Marginal Density")

        # generated sample
        ax.plot(
            data["cs"],
            model_age,
            color="red",
            alpha=1,
            linewidth=1,
            label = rf"$A_{{{data["dn"][:3]}}}(d)$" if "d18o" in data["dn"] else rf"$A_{{{data["dn"][:3]}}}(d)$",
        )

        for j, c in enumerate(data["cs"]):
            ax.axvline(x=c, color='k', linestyle='--', linewidth = 0.1)
        
        # prior
        plt.fill_between(
            data["cs"], 
            prior_mean - prior_2_std, 
            prior_mean + prior_2_std, 
            color="blue", 
            alpha=0.1,
            linewidth=0,
            label=r"2$\sigma$ prior"
        )
        ax.set_xlim(0, np.max(data["cs"]))
        if "d18o" in data["dn"]:
            ax.set_ylim(1200, 2000)
            plt.xlabel("Distance from top of stalagmite [mm]")
        else:
            ax.set_ylim(-8000, 2000)
            plt.xlabel("Depth of sediment [cm]")

        # direct dates
        plt.errorbar(data["c14d"], data["c14"], yerr = data["c14s"], fmt ="o", color = 'k', capsize = 2, markersize=4, label = c14_label)
        plt.ylabel("Year CE")
        plt.legend(loc = "upper right")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{i}/ad_sample.jpg")
        # plt.show()

        # ---------Plotting generated data and reference function-----------
        fig, (ax3, ax2) = plt.subplots(2, 1, figsize=(6, 4), sharex=False)

        valid_mask = ~np.isnan(d18O)

        ax2.errorbar(
            depths[valid_mask],
            d18O[valid_mask], 
            yerr=2 * sigma_d18O[valid_mask],
            fmt='.', 
            markersize=5,
            capsize=4, 
            elinewidth=1, 
            capthick=0,
            color="black",
            ecolor='gray', 
            alpha=1,
            label="Measurements ± uncertainty"
        )

        if "biw" in data["dn"]:
            ax2.set_xlabel("Depth of sediment [cm]") 
            ax2.set_ylabel(r"Declination [°]")
            ax3.set_ylabel(r"Declination [°]")
            ax2.set_ylim(-30, 20)
            ax3.set_ylim(-30, 20)
        elif "d18o" in data["dn"]:
            ax2.set_xlabel("Distance from top of stalagmite [mm]") 
            ax2.set_ylabel("$\\delta^{18}$O [‰]")
            ax3.set_ylabel("$\\delta^{18}$O [‰]")
            if "dayu" in data["dn"]:
                ax2.set_ylim(-9.5, -8)
                ax3.set_ylim(-9.5, -8)
            else:
                ax2.set_ylim(-8, -6.5)
                ax3.set_ylim(-8, -6.5)

        # Discretization
        for d in data["cs"]:
            ax2.axvline(x=d, color='k', linestyle='--', linewidth=0.2, alpha=0.5)

        # data
        ax3.plot(data["d18ort"], data["d18or"], color="k", linewidth=0.8)
        ax3.set_xlabel("Year CE")
        ax3.invert_xaxis()
        ax3.xaxis.tick_top()
        ax3.xaxis.set_label_position('top')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{i}/data_fig.jpg")
        # plt.show()



if __name__ == "__main__":
    main()
