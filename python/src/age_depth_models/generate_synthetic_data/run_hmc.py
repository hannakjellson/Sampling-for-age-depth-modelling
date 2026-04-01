import ctypes
import numpy as np
import scipy as sc
import os
from datetime import datetime
import json

from define_data_and_variables import HMCConfig, Data, hmc_config, c_hmc_config, data, c_data, make_dumpable
import platform
from pathlib import Path
import re
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.transforms as mtransforms
from matplotlib.patches import ConnectionPatch
import pandas as pd

def main():
    base_path = "../../../../data/"

    output_dir = os.path.join(base_path, data["dn"])
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
    outdir_samples = f"{output_dir}/samples.npy"
    np.save(outdir_samples, samples)

    energy_values = np.ctypeslib.as_array(energy_out)
    energy_values = np.reshape(energy_values, (hmc_config["nch"], hmc_config["ns"]))
    outdir_energy =  f"{output_dir}/energy.npy"
    np.save(outdir_energy, energy_values)
    print("------Done sampling------")

    print("------Generating 3 sets of reference data------")
    c14_label = r"$^{230}$Th" if "d18o" in data["dn"] else r"$^{14}C$"
    d18o_label = "$\\delta^{18}$O [‰]" if "d18o" in data["dn"] else "inc [°]" if "inc" in data["dn"] else "dec [°]"
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

        np.save(f"{output_dir}/{i}/true_sample.npy", sample)

        fig, ax = plt.subplots(figsize=(6, 4))
        for j, c in enumerate(data["cs"]):
            ax.axvline(x=c, color='k', linestyle='--', linewidth = 0.1)
        ax.errorbar(
            data["d18od"], 
            d18O_actual, 
            data["d18os"]*np.ones(len(d18O_actual)), 
            color = "black",
            fmt='o',              # circular markers
            linestyle='none',     # no line between points
            capsize=3,
            markersize = 4,
            alpha = 0.2,
        )
        ax.plot(depths_interp, data["d18or"][::-1], linewidth = 2.5, color = "black")
        ax.plot(depths_interp, data["d18or"][::-1], linewidth = 2.5, color = "red", linestyle = "dashed")
        ax.set_xlabel("Distance from top of stalagmite [mm]")
        ax.set_ylabel(d18o_label)
        plt.savefig(os.path.join(base_path, data["dn"], f"{i}", "generated_data.jpg"))

    print("------Saved 3 sets of reference data------")
    print("----------Plotting----------")
    if "d18o" in data["dn"]:
        non_filtered_ref = pd.read_csv(os.path.join(output_dir, "ref_unfiltered.txt"), sep = "\t")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(non_filtered_ref["Year"], non_filtered_ref["ref"], color = "grey", linewidth = 1, label = "ECHAM5-wiso")
        ax.plot(data["d18ort"], data["d18or"], color = "black", linewidth = 1, label = "ECHAM5-wiso, low-pass filtered")
        ax.legend(loc = "upper right")
        ax.set_xlabel("Year CE")
        ax.set_ylabel(d18o_label)
        plt.savefig(os.path.join(output_dir, "d18o_echam.jpg"))
        plt.show()

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

    for i in range(3):
        fig, ax = plt.subplots(figsize=(6, 4))
        plt.set_cmap(plt.cm.Greys)

        im = ax.imshow(
            C.T,
            extent=[z[0], z[-1], t[0], t[-1]],
            origin='lower',
            aspect='auto',
            # vmin=0,
            # vmax=0.01 * N,
            norm=colors.LogNorm()
        )

        cbar = plt.colorbar(im)
        cbar.set_label("Marginal Density")
        plt.xlabel("Distance from top of stalagmite [mm]")
        plt.ylabel("Year CE")

        ax.plot(
            data["cs"],
            model_ages_list[i],
            color="red",
            alpha=1,
            linewidth=1,
            label = rf"$A_{{{data["dn"][:3]}}}(d)$"
        )

        for j, c in enumerate(data["cs"]):
            ax.axvline(x=c, color='k', linestyle='--', linewidth = 0.1)
        

        ax.set_xlim(0, np.max(data["cs"]))
        ax.set_ylim(np.min(data["d18ort"]), np.max(data["d18ort"]))
        plt.errorbar(data["c14d"], data["c14"], yerr = data["c14s"], fmt ="o", color = 'k', capsize = 2, markersize=4, label = c14_label)
        plt.xlabel("Distance from top of stalagmite [mm]")
        plt.ylabel("Year CE")
        plt.legend(loc = "upper right")
        plt.savefig(f"{output_dir}/{i}/ad_sample.jpg")
        plt.show()


        ### Fancy figure
        fig, (ax1, ax2, ax3) = plt.subplots(
            ncols=3,
            figsize=(10, 8),
            gridspec_kw={'width_ratios': [1, 2, 2]},
            sharey=False
        )

        ax1.sharey(ax2)

        ax2.errorbar(
            d18O_actual, data["d18od"],
            xerr=data["d18os"],
            fmt='.',                # circle markers
            markersize=7,
            capsize=4,               # small caps on error bars
            elinewidth=2,          # thin error bar lines
            capthick=0,
            color="red",
            ecolor='gray',           # error bar color
            alpha=0.5,
            label="Measurements ± uncertainty"
        )
        ax2.xaxis.tick_top()
        ax2.xaxis.set_label_position('top')
        ax2.set_xlabel(d18o_label)
        ax1.invert_yaxis()

        extra_depths = data["c14d"]

        ax2.scatter(
            [1]*len(extra_depths),   # x-position (right edge)
            extra_depths,
            marker='*',
            color='black',
            transform=ax2.get_yaxis_transform(),
            clip_on=False
        )

        for j, d in enumerate(data["c14d"]):
            ax2.axhline(y=d, color='k', linestyle='--', linewidth = 0.5, zorder=0)
            # ax1.axhline(y=d, color='r', linestyle='--', linewidth = 0.5, zorder=0)
            if j == len(data["c14"]) - 1:
                y = d + 2
            else:
                y = d
            ax2.text(
                x=ax2.get_xlim()[0] + 0.01,    # right end of x-axis
                y=y ,             # slightly above the line     
                s=f"{d}",          # text
                ha='left',            # align text to the right
                va='bottom',           # bottom of text at y + offset
                # fontsize=8
            )


        ax3.plot(data["d18or"], data["d18ort"], color = "k", linewidth = 0.8)
        ax3.plot(data["d18or"], data["d18ort"], '.', color = "k", markersize = 1)
        ax3.set_ylabel("Year CE")
        ax3.set_xlabel(d18o_label)
        ax3.xaxis.tick_top()
        ax3.xaxis.set_label_position('top')

        # Move y-axis to the right
        ax3.yaxis.tick_right()                # Move tick marks to the right
        ax3.yaxis.set_label_position("right") # Move y-axis label to the right
        ax3.spines['right'].set_position(('outward', 0))  # Position the right spine

        for j, d in enumerate(data["c14"]):
            ax3.axhline(y=d, color='k', linestyle='--', linewidth = 0.5, zorder=0)
            # ax1.axhline(y=d, color='r', linestyle='--', linewidth = 0.5, zorder=0)
            if j == len(data["c14"]) - 1:
                y = d - 25
            else:
                y = d
            ax3.text(
                x=ax3.get_xlim()[1] - 0.01,    # right end of x-axis
                y=y,             # slightly above the line     
                s=f"{d} $\\pm$ {data["c14s"][j]}",          # text
                ha='right',            # align text to the right
                va='bottom',           # bottom of text at y + offset
                # fontsize=8
            )

        ax1.xaxis.set_visible(False)
        ax1.set_ylabel("Distance from top of stalagmite [mm]")

        for j, d in enumerate(data["cs"]):
            ax1.axhline(y=d, color='k', linestyle='--', linewidth = 0.1)
            ax2.axhline(y=d, color='k', linestyle='--', linewidth = 0.1)
            if (j>0 and j <=3):
                    ax1.text(
                        x=ax1.get_xlim()[0] + 0.5,    # center of x-axis
                        y=d ,             # slightly above the line
                        s=f"$x_{{{j}}}$",          # text
                        ha='center',            # align text to the right
                        va='bottom',           # bottom of text at y + offset
                        # fontsize=8
                    )

                    ax1.text(
                        x=ax1.get_xlim()[0] + 0.01,    # right end of x-axis
                        y=data["cs"][j-1] ,             # slightly above the line
                        s=f"$d_{{{j-1}}}$",          # text
                        ha='left',            # align text to the right
                        va='bottom',           # bottom of text at y + offset
                        # fontsize=8
                    )
            elif (j > data["N"] - 3 and j < data["N"]):
                ax1.text(
                    x=ax1.get_xlim()[0] + 0.5,    # right end of x-axis
                    y=d ,             # slightly above the line
                    s=f"$x_{{K-{data["N"] - j}}}$",          # text
                    ha='center',            # align text to the right
                    va='bottom',           # bottom of text at y + offset
                )
                    
                ax1.text(
                    x=ax1.get_xlim()[0] + 0.01,    # right end of x-axis
                    y=data["cs"][j] ,             # slightly above the line
                    s=f"$d_{{K-{data["N"] - j}}}$",          # text
                    ha='left',            # align text to the right
                    va='bottom',           # bottom of text at y + offset
                    # fontsize=8
                )# fontsize=8
            elif (j == data["N"]):
                ax1.text(
                    x=ax1.get_xlim()[0] + 0.5,    # right end of x-axis
                    y=d ,             # slightly above the line
                    s=f"$x_K$",          # text
                    ha='center',            # align text to the right
                    va='bottom',           # bottom of text at y + offset
                    # fontsize=8
                )

                ax1.text(
                    x=ax1.get_xlim()[0] + 0.01,    # right end of x-axis
                    y=data["cs"][j] ,             # slightly above the line
                    s=f"$d_K$",          # text
                    ha='left',            # align text to the right
                    va='bottom',           # bottom of text at y + offset
                    # fontsize=8
                )# fontsize=8
            elif (j % 2 == 0 and j != 0):
                ax1.text(
                    x=ax1.get_xlim()[0] + 0.5,    # right end of x-axis
                    y=d ,             # slightly above the line
                    s=f"$\\vdots$",          # text
                    ha='center',            # align text to the right
                    va='bottom',           # bottom of text at y + offset
                    # fontsize=8
                )

                ax1.text(
                    x=ax1.get_xlim()[0] + 0.01,    # right end of x-axis
                    y=d ,             # slightly above the line
                    s=f"$\\vdots$",          # text
                    ha='left',            # align text to the right
                    va='bottom',           # bottom of text at y + offset
                    # fontsize=8
                )

        plt.tight_layout(rect = [0, 0.05, 1, 1])
        ax2.tick_params(axis='y', left=False, labelleft=False)
        pos2 = ax2.get_position()
        pos1 = ax1.get_position()
        ax1.set_position([pos1.x0, pos2.y0, pos2.x0 - pos1.x0, pos2.height])

        for j, d in enumerate(data["c14d"]):
            # Create a line connecting (xlim_max of ax2, depth) to (xlim_min of ax3, depth)
            con = ConnectionPatch(
                xyA=(ax2.get_xlim()[1], d), # Point in ax2 (right side)
                xyB=(ax3.get_xlim()[0], data["c14"][j]), # Point in ax3 (left side)
                coordsA="data", coordsB="data",
                axesA=ax2, axesB=ax3,
                color="black", linestyle="--", linewidth=0.5, alpha=0.5
            )
            fig.add_artist(con)

        ax1.set_title("a) Discretization", y=-0.07)
        ax2.set_title(f"b) {d18o_label} and {c14_label} Data", y=-0.07)
        ax3.set_title(f"c) Reference {d18o_label}", y=-0.07)
        plt.savefig(f"{output_dir}/{i}/fancy_fig_classic_new.jpg")
        plt.show()



if __name__ == "__main__":
    main()
