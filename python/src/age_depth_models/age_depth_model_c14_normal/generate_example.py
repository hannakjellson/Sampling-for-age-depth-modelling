import numpy as np
from pathlib import Path
from define_data_and_variables import HMCConfig, Data, hmc_config, data, c_data, dict_to_struct, make_dumpable, hash_configs
import matplotlib.pyplot as plt
from matplotlib import colors

def expected_ages(N, delta_c, cs, theta, num_depths, depths, sed_rates, indices):
    cumulative_sum_vec = [0.0] * (N + 1)
    
    # Build cumulative sum vector
    for i in range(1, N + 1):
        cumulative_sum_vec[i] = cumulative_sum_vec[i - 1] + sed_rates[i - 1] * delta_c
    
    ages_out = [0.0] * num_depths
    
    # Compute ages
    for i in range(num_depths):
        index = indices[i]
        cumulative_sum = cumulative_sum_vec[index]
        cumulative_sum += sed_rates[index] * (depths[i] - cs[index])
        ages_out[i] = theta - cumulative_sum
    
    return ages_out

def get_index(N, cs, depths, depth_index):
    low, high = 0, N
    target = depths[depth_index]
    print(target)
    
    while low < high:
        mid = int(low + (high - low) / 2)
        if cs[mid] < target:
            low = mid + 1
        else:
            high = mid
    
    return low


base_dir = Path(__file__).parent 
np.random.seed(42)
sp = np.random.multivariate_normal(data["pm"]* np.ones(data["N"]), data["ps"]**2 * np.eye(data["N"]), (hmc_config["nch"]))
sp = np.exp(sp)
hmc_config["sp"] = np.ascontiguousarray(sp)
hmc_hash = hash_configs(hmc_config, data)
output_dir = base_dir / f"output/{hmc_hash}"
samples = np.load(output_dir / "samples.npy")[:, 2000:, :]

np.random.seed(2)
samples_1 = samples.reshape(-1, samples.shape[-1])[
    np.random.randint(samples.shape[0] * samples.shape[1])
]

np.random.seed(59)
samples_2 = samples.reshape(-1, samples.shape[-1])[
    np.random.randint(samples.shape[0] * samples.shape[1])
]

np.random.seed(101)
samples_3 = samples.reshape(-1, samples.shape[-1])[
    np.random.randint(samples.shape[0] * samples.shape[1])
]
cumulative_sums_1 = np.cumsum(samples_1) * data["dc"]
cumulative_sums_2 = np.cumsum(samples_2) * data["dc"]
cumulative_sums_3 = np.cumsum(samples_3) * data["dc"]
# zeros = np.zeros((*sp.shape[:1], 1))
age_offsets_1 = np.concatenate((np.array([0]), cumulative_sums_1))
model_ages_1 = data["th"] - age_offsets_1
age_offsets_2 = np.concatenate((np.array([0]), cumulative_sums_2))
model_ages_2 = data["th"] - age_offsets_2
age_offsets_3 = np.concatenate((np.array([0]), cumulative_sums_3))
model_ages_3 = data["th"] - age_offsets_3

flat_samples = samples.reshape(-1, samples.shape[-1])
cumulative_sums = np.cumsum(flat_samples, axis=1) * data["dc"]
zeros = np.zeros((*flat_samples.shape[:1], 1))
age_offsets = np.concatenate((zeros, cumulative_sums), axis=1)
model_ages = data["th"] - age_offsets

chain = 4
X = samples[chain].T
D, N = X.shape

# Depth grid (0:2:2*D)'
depth = np.arange(0, 2 * D + 1, 2)

age = np.vstack([
    np.ones((1, N)) * data["th"],
    data["th"] - data["dc"] * np.cumsum(X, axis=0)
])

# age_mean = np.mean(age, axis = 1)
norm_age = age #- true_age[:, None]

# Time bins
dt = 1
t_edges = np.arange(1200, 2000 + dt, dt)
t = t_edges[1:] - dt / 2
T = t.size

# Depth interpolation grid
dz = 0.1
z = np.arange(0, 2 * D + dz, dz)
Z = z.size

# Allocate result
C = np.full((Z, T), np.nan)
A = np.array([
    np.interp(z, depth, norm_age[:, j])
    for j in range(N)
]).T

for i in range(Z):
    C[i, :], _ = np.histogram(A[i, :], bins=t_edges)

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
cbar.set_label("Counts")
plt.xlabel("Distance from top of stalagmite [mm]")
plt.ylabel("Year CE")

ax.plot(
    data["cs"],
    model_ages_1,
    color='purple',
    alpha=1,
    linewidth=1,
    label = r"$A_1(d)$"
)

ax.plot(
    data["cs"],
    model_ages_2,
    color='orange',
    alpha=1,
    linewidth=1,
    label = r"$A_2(d)$"
)

ax.plot(
    data["cs"],
    model_ages_3,
    color='blue',
    alpha=1,
    linewidth=1,
    label = r"$A_3(d)$"
)

for i, c in enumerate(data["cs"]):
    ax.axvline(x=c, color='k', linestyle='--', linewidth = 0.1)
    

ax.set_xlim(0, np.max(data["cs"]))
ax.set_ylim(1200, 2000)
plt.plot(data["c14d"], np.squeeze(data["c14"]), "ko", markersize=4, label = r"$^{230}Th$")
# plt.plot(config["cs"], data["theta"] - config["dc"] * np.cumsum((config["a"]/config["b"]) * np.ones_like(config["cs"])), color = 'k', linewidth=0.5)
plt.xlabel("Distance from top of stalagmite [mm]")
plt.ylabel("Year CE")
plt.legend(loc = "upper right")
plt.show()

# import pandas as pd
# import os

# base_path = "../../../../data/"
# echam_path = os.path.join(base_path, "inputdata_250306A/ECHAM5_d18O_Dayu_Cave.xlsx")

# df = pd.read_excel(echam_path)
# years = df["Year"].to_numpy()
# d18o = df["d18O"].to_numpy()
# d18o_filtered = df["Filtered d18O"].to_numpy()

# fig, ax = plt.subplots(figsize=(6, 4))
# ax.plot(years, d18o, color = "green", linewidth = 1, label = "ECHAM5-wiso")
# ax.plot(years, d18o_filtered, color = "black", linewidth = 1, label = "ECHAM5-wiso, low-pass filtered")
# ax.legend(loc = "upper right")
# ax.set_xlabel("Year CE")
# ax.set_ylabel(r"$\delta^{18}O$ [‰]")
# plt.show()

# old_dayu_file = pd.read_csv(
#     os.path.join(base_path, "inputdata_250306A/Dayu cave.txt"), sep="\t"
# )
# if old_dayu_file is not None:
#     old_dayu_file.columns = old_dayu_file.columns.str.replace(
#         "%", ""
#     ).str.strip()

# d18o_old = old_dayu_file["d18O"].to_numpy()
# d18o_mask = ~np.isnan(d18o_old)
# actual_depths = old_dayu_file["depth"][d18o_mask].to_numpy()

# years = df["Year"].to_numpy()
# d18O = df["Filtered d18O"].to_numpy()
# mask = (years >= model_ages_1[-1]) & (years <= model_ages_1[0])

# depths = np.interp(years[mask], model_ages_1[::-1], data["cs"][::-1])
# d18O_actual = np.interp(actual_depths, depths[::-1], d18O[mask][::-1]) + np.random.normal(loc = 0, scale = 0.1, size= len(actual_depths))
# years_actual = np.interp(actual_depths, depths[::-1], years[mask][::-1])

# c14_depths = np.array([0.00, 2.50, 27.75, 35.25, 54.00, 72.50, 78.75, 96.75, 98.00])
# years_c14_actual = np.interp(c14_depths, depths[::-1], years[mask][::-1])
# print(years_c14_actual[0])

# fig, ax = plt.subplots(figsize=(6, 4))
# for i, c in enumerate(data["cs"]):
#     ax.axvline(x=c, color='k', linestyle='--', linewidth = 0.1)
# ax.errorbar(
#     actual_depths, 
#     d18O_actual, 
#     0.1*np.ones(len(d18O_actual)), 
#     color = "black",
#     fmt='o',              # circular markers
#     linestyle='none',     # no line between points
#     capsize=3,
#     markersize = 4,
#     alpha = 0.2,
# )
# ax.plot(depths, d18O[mask], linewidth = 2.5, color = "black")
# ax.plot(depths, d18O[mask], linewidth = 2.5, color = "purple", linestyle = "dashed")
# ax.set_xlabel("Distance from top of stalagmite [mm]")
# ax.set_ylabel(r"$\delta^{18}O$ $[‰]$")
# plt.show()

# os.makedirs(os.path.join(base_path, "inputdata_260219C"), exist_ok=True)
# init_nan = np.full(1, np.nan)
# init_nans_c14 = np.full(9, np.nan)
# extra_nans = np.full(len(actual_depths), np.nan)

# depths = np.concatenate((c14_depths, actual_depths))
# true_ages = np.concatenate((years_c14_actual, years_actual))
# cal_c14_ages = np.concatenate((init_nan, np.array([1970.00, 1836.00, 1787.00, 1660.00, 1559.00, 1512.00, 1278.00, 1271.00]), extra_nans))
# sigma_age = np.concatenate((init_nan, np.array([0.50, 1.00, 1.00, 1.50, 2.00, 1.50, 2.50, 4.00]), extra_nans))
# d18O = np.concatenate((init_nans_c14, d18O_actual))
# sigma_d18O = np.concatenate((init_nans_c14, np.full(len(actual_depths), 0.1)))

# df = pd.DataFrame({
#     "depth": depths,
#     "true_age": true_ages,
#     "cal_c14_age": cal_c14_ages,
#     "sigma_age": sigma_age,
#     "d18O": d18O,
#     "sigma_d18O": sigma_d18O
# })

# df.to_csv(
#     os.path.join(base_path, "inputdata_260219C/Dayu cave.txt"),
#     sep="\t",
#     index=False
# )

# np.save(f"{base_path}/inputdata_260219C/true_sample.npy", samples_3)


# import matplotlib.transforms as mtransforms
# from matplotlib.patches import ConnectionPatch

# fig, (ax1, ax2, ax3) = plt.subplots(
#     ncols=3,
#     figsize=(6, 4),
#     gridspec_kw={'width_ratios': [1, 2, 2]},
#     sharey=False
# )

# ax1.sharey(ax2)

# ax2.errorbar(
#     data["d18o"], data["d18od"],
#     xerr=data["d18os"],
#     fmt='.k',                # circle markers
#     markersize=7,
#     capsize=4,               # small caps on error bars
#     elinewidth=2,          # thin error bar lines
#     capthick=0,
#     color='black',
#     ecolor='gray',           # error bar color
#     alpha=0.9,
#     label="Measurements ± uncertainty"
# )
# ax2.xaxis.tick_top()
# ax2.xaxis.set_label_position('top')
# ax2.set_xlabel("$\\delta^{18}O$ [‰]")
# ax1.invert_yaxis()

# extra_depths = data["c14d"]

# ax2.scatter(
#     [1]*len(extra_depths),   # x-position (right edge)
#     extra_depths,
#     marker='*',
#     color='tab:red',
#     transform=ax2.get_yaxis_transform(),
#     clip_on=False
# )

# for i, d in enumerate(data["c14d"]):
#     ax2.axhline(y=d, color='r', linestyle='--', linewidth = 0.5, zorder=0)
#     # ax1.axhline(y=d, color='r', linestyle='--', linewidth = 0.5, zorder=0)
#     if i == len(data["c14"]) - 1:
#         y = d + 2
#     else:
#         y = d
#     ax2.text(
#         x=ax2.get_xlim()[0] + 0.01,    # right end of x-axis
#         y=y ,             # slightly above the line     
#         s=f"{d}",          # text
#         ha='left',            # align text to the right
#         va='bottom',           # bottom of text at y + offset
#         # fontsize=8
#     )


# ax3.plot(data["d18or"], data["d18ort"], color = "k", linewidth = 0.8)
# ax3.plot(data["d18or"], data["d18ort"], '.', color = "k", markersize = 1)
# # for i in range(config["nch"]):
# #     variables = resampled_samples[i,:,:]
# #     for j in range(len(variables)):
# #         d18O_ages = expected_ages(config["N"], config["dc"], config["cs"], data["theta"], len(data["d18O_depths"]), data["d18O_depths"], variables[j,:], indices)
# #         plt.plot(d18O_ages, data["d18O"], '*', color = colors[i])

# # plt.plot(data["true_ages_D18O"], data["d18O"], '*', color = "k")
# ax3.set_ylabel("Year CE")
# ax3.set_xlabel("$\\delta^{18}O$ [‰]")
# ax3.xaxis.tick_top()
# ax3.xaxis.set_label_position('top')

# # Move y-axis to the right
# ax3.yaxis.tick_right()                # Move tick marks to the right
# ax3.yaxis.set_label_position("right") # Move y-axis label to the right
# ax3.spines['right'].set_position(('outward', 0))  # Position the right spine

# for i, d in enumerate(data["c14"]):
#     ax3.axhline(y=d, color='r', linestyle='--', linewidth = 0.5, zorder=0)
#     # ax1.axhline(y=d, color='r', linestyle='--', linewidth = 0.5, zorder=0)
#     if i == len(data["c14"]) - 1:
#         y = d - 25
#     else:
#         y = d
#     ax3.text(
#         x=ax3.get_xlim()[1] - 0.01,    # right end of x-axis
#         y=y,             # slightly above the line     
#         s=f"{d} $\\pm$ {data["c14s"][i]}",          # text
#         ha='right',            # align text to the right
#         va='bottom',           # bottom of text at y + offset
#         # fontsize=8
#     )

# ax1.xaxis.set_visible(False)
# ax1.set_ylabel("Distance from top of stalagmite [mm]")

# for i, d in enumerate(data["cs"]):
#     ax1.axhline(y=d, color='k', linestyle='--', linewidth = 0.1)
#     ax2.axhline(y=d, color='k', linestyle='--', linewidth = 0.1)
#     if (i>0 and i <=3):
#             ax1.text(
#                 x=ax1.get_xlim()[0] + 0.5,    # center of x-axis
#                 y=d ,             # slightly above the line
#                 s=f"$x_{{{i}}}$",          # text
#                 ha='center',            # align text to the right
#                 va='bottom',           # bottom of text at y + offset
#                 # fontsize=8
#             )

#             ax1.text(
#                 x=ax1.get_xlim()[0] + 0.01,    # right end of x-axis
#                 y=data["cs"][i-1] ,             # slightly above the line
#                 s=f"$c_{{{i-1}}}$",          # text
#                 ha='left',            # align text to the right
#                 va='bottom',           # bottom of text at y + offset
#                 # fontsize=8
#             )
#     elif (i > data["N"] - 3 and i < data["N"]):
#         ax1.text(
#             x=ax1.get_xlim()[0] + 0.5,    # right end of x-axis
#             y=d ,             # slightly above the line
#             s=f"$x_{{K-{data["N"] - i}}}$",          # text
#             ha='center',            # align text to the right
#             va='bottom',           # bottom of text at y + offset
#         )
            
#         ax1.text(
#             x=ax1.get_xlim()[0] + 0.01,    # right end of x-axis
#             y=data["cs"][i] ,             # slightly above the line
#             s=f"$c_{{K-{data["N"] - i}}}$",          # text
#             ha='left',            # align text to the right
#             va='bottom',           # bottom of text at y + offset
#             # fontsize=8
#         )# fontsize=8
#     elif (i == data["N"]):
#         ax1.text(
#             x=ax1.get_xlim()[0] + 0.5,    # right end of x-axis
#             y=d ,             # slightly above the line
#             s=f"$x_K$",          # text
#             ha='center',            # align text to the right
#             va='bottom',           # bottom of text at y + offset
#             # fontsize=8
#         )

#         ax1.text(
#             x=ax1.get_xlim()[0] + 0.01,    # right end of x-axis
#             y=data["cs"][i] ,             # slightly above the line
#             s=f"$c_K$",          # text
#             ha='left',            # align text to the right
#             va='bottom',           # bottom of text at y + offset
#             # fontsize=8
#         )# fontsize=8
#     elif (i % 2 == 0 and i != 0):
#         ax1.text(
#             x=ax1.get_xlim()[0] + 0.5,    # right end of x-axis
#             y=d ,             # slightly above the line
#             s=f"$\\vdots$",          # text
#             ha='center',            # align text to the right
#             va='bottom',           # bottom of text at y + offset
#             # fontsize=8
#         )

#         ax1.text(
#             x=ax1.get_xlim()[0] + 0.01,    # right end of x-axis
#             y=d ,             # slightly above the line
#             s=f"$\\vdots$",          # text
#             ha='left',            # align text to the right
#             va='bottom',           # bottom of text at y + offset
#             # fontsize=8
#         )

# plt.tight_layout()
# ax2.tick_params(axis='y', left=False, labelleft=False)
# pos2 = ax2.get_position()
# pos1 = ax1.get_position()
# ax1.set_position([pos1.x0, pos2.y0, pos2.x0 - pos1.x0, pos2.height])

# for i, d in enumerate(data["c14d"]):
#     # Create a line connecting (xlim_max of ax2, depth) to (xlim_min of ax3, depth)
#     con = ConnectionPatch(
#         xyA=(ax2.get_xlim()[1], d), # Point in ax2 (right side)
#         xyB=(ax3.get_xlim()[0], data["c14"][i]), # Point in ax3 (left side)
#         coordsA="data", coordsB="data",
#         axesA=ax2, axesB=ax3,
#         color="red", linestyle="--", linewidth=0.5, alpha=0.5
#     )
#     fig.add_artist(con)

# ax1.set_title("a) Discretization", y=-0.05)
# ax2.set_title("b) $\\delta^{18}O$ and $^{14}C$ Data", y=-0.05)
# ax3.set_title("c) Reference $\\delta^{18}O$", y=-0.05)
# plt.savefig("fancy_fig_new.jpg")
# plt.show()

