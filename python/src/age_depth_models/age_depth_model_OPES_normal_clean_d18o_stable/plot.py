import numpy as np
import matplotlib.pyplot as plt
import os
import gc
from pathlib import Path
from define_data_and_variables import adam_config, opes_config, data, adam_hash, hash_configs
import json
from matplotlib.collections import LineCollection
from matplotlib import colors


base_dir = Path.cwd()

base_path = base_dir / ".." / ".." / ".." / ".." / "data"
if data["dn"].lower() == "dayu19a":
    true_sample = np.load(base_path / "inputdata_260219A" / "true_sample.npy")
    bin_range = (1660, 1720)
    data_color = "red"
    data_name = r"$A_1$"
    ylim_max = 0.4
elif data["dn"].lower() == "dayu19b":
    true_sample = np.load(base_path / "inputdata_260219B" / "true_sample.npy")
    bin_range = (1660, 1720)
    data_color = "orange"
    data_name = r"$A_2$"
    ylim_max = 0.15
if data["dn"].lower() == "dayu19c":
    true_sample = np.load(base_path / "inputdata_260219C" / "true_sample.npy")
    bin_range = (1660, 1720)
    data_color = "yellow"
    data_name = r"$A_3$"
    ylim_max = 0.25
true_ages = np.hstack((data["th"], data["th"] - np.cumsum(true_sample) * data["dc"]))

base_dir = Path.cwd()
adam_dir = base_dir / f"output/{data['dn']}/sd_{opes_config['hmcc']['sd']}"

np.random.seed(opes_config["hmcc"]["sd"])
sp = np.random.lognormal(mean = data["pm"], sigma = data["ps"], size = (opes_config["hmcc"]["nch"], data["N"]))

opes_config["hmcc"]["sp"] = np.ascontiguousarray(sp)
opes_config["df"] = 140 * (opes_config["bs"] - 1)
opes_config["dfn"] = np.exp(-opes_config["df"])*opes_config["dfd"]

opes_hash = hash_configs(opes_config, data)
output_dir = adam_dir / f"{opes_hash}"

hmc_config = opes_config["hmcc"]

cutout = 10000
bias_values = np.load(f"{output_dir}/bias.npy", mmap_mode='r')[:, cutout:]
samples = np.load(output_dir / "samples.npy", mmap_mode='r')[:, cutout:, :]
weights = np.exp(bias_values)

index = 25
interesting_depth = index * data["dc"]
dt = 1
block_size = int(len(samples[0, :, 0])/5000)

flat_samples = samples.reshape(-1, data["N"])
flat_weights = weights.reshape(-1)
ns = len(flat_samples[:, 0])

age = np.hstack([
    np.ones((ns, 1)) * data["th"],
    data["th"] - data["dc"] * np.cumsum(flat_samples, axis=1)
])

# Depth interpolation grid
dz = 0.1
z = np.arange(0, data["H"] + dz, dz)
Z = z.size

t_edges = [1200, 2000]

K = int(ns/block_size)
C_jackknife = np.full((Z, t_edges[-1] - t_edges[0]), np.nan)
C_opes = np.full((Z, t_edges[-1] - t_edges[0]), np.nan)
sigma_est_jackknife = np.full((Z, t_edges[-1] - t_edges[0]), np.nan)
sigma_est_opes = np.full((Z, t_edges[-1] - t_edges[0]), np.nan)

for i in range(0, Z):
    hists = []
    ages = np.array([np.interp(z[i], data["cs"], age[j, :]) for j in range(ns)])
    bins = np.arange(np.floor(np.min(ages)), np.ceil(np.max(ages)), dtype = int)
    if bins.size < 2:
        bins = np.array([np.floor(np.min(ages)), np.ceil(np.max(ages)) + 1], dtype = int)
    
    bin_indices = np.digitize(ages, bins) - 1
    num_bins = len(bins) - 1

    for b in range(num_bins):
        hists.append(np.sum(flat_weights[bin_indices == b]))

    full_hist = np.array(hists)

    block_hists = np.zeros((K, num_bins))
    block_weight_sum = np.zeros((K))
    for k in range(K):
        start, end = k * block_size, (k + 1) * block_size
        # Slice the block's indices and weights
        b_idx = bin_indices[start:end]
        b_w = flat_weights[start:end]
        block_weight_sum[k] = np.sum(b_w)
        
        # Sum weights in this block
        for b in range(num_bins):
            block_hists[k, b] = np.sum(b_w[b_idx == b])

    norm_full_hist = full_hist / np.sum(flat_weights)

    # Jackknife estimate
    loo_results = full_hist - block_hists
    norm_loo_results  = loo_results / (np.sum(flat_weights) - block_weight_sum[:, None])
    jackknife_est = K * norm_full_hist - (K - 1) * np.mean(norm_loo_results, axis=0)
    C_jackknife[i, bins[:-1]-t_edges[0]] = jackknife_est
    sigma_est_jackknife[i, bins[:-1]-t_edges[0]]  = np.sqrt((K-1) * np.sum((norm_loo_results - jackknife_est)**2, axis = 0) / K)

    # OPES estimate
    norm_block_results = block_hists/block_weight_sum[:, None]
    opes_est = block_weight_sum@norm_block_results/np.sum(block_weight_sum)
    meff = np.sum(block_weight_sum)**2 / np.sum(block_weight_sum**2)
    C_opes[i, bins[:-1]-t_edges[0]] = opes_est
    sigma_est_opes[i, bins[:-1]-t_edges[0]] = np.sqrt((block_weight_sum @ (norm_block_results - norm_full_hist)**2 / ((meff - 1) *np.sum(block_weight_sum))))
    print(f"progress: {i/Z}")
    print(f"C_jackknife[i]: {C_jackknife[i][~np.isnan(C_jackknife[i])]}")
    print(f"C_opes[i]: {C_opes[i][~np.isnan(C_opes[i])]}")
    print(f"sigma_est_jackknife[i]: {sigma_est_jackknife[i][~np.isnan(sigma_est_jackknife[i])]}")
    print(f"sigma_est_opes[i]: {sigma_est_opes[i][~np.isnan(sigma_est_opes[i])]}")

print("Saving")
np.save(f"{output_dir}/C_jackknife", C_jackknife)
np.save(f"{output_dir}/sigma_est_jackknife", sigma_est_jackknife)
np.save(f"{output_dir}/C_opes", C_opes)
np.save(f"{output_dir}/sigma_est_opes", sigma_est_opes)

# C_jackknife = np.load(f"{output_dir}/C_jackknife.npy")
# sigma_est_jackknife = np.load(f"{output_dir}/sigma_est_jackknife.npy")
# C_opes = np.load(f"{output_dir}/C_opes.npy")
# sigma_est_opes = np.load(f"{output_dir}/sigma_est_opes.npy")

print("Plotting")
### Plotting Jackknife
fig, ax = plt.subplots(figsize=(6, 4))
plt.set_cmap(plt.cm.Greys)
for i, c in enumerate(data["cs"]):
    if i ==index: 
        ax.axvline(x=c, color='k', linestyle='--', linewidth = 1, label = f"d = {int(interesting_depth)} mm")
    else:
        ax.axvline(x=c, color='k', linestyle='--', linewidth = 0.1)

ax.plot(data["c14d"], np.squeeze(data["c14"]), "ko", markersize=4, label = r"$^{230}Th$")
ax.plot(data["cs"], true_ages, color = data_color, linewidth=3, alpha = 0.2, label = data_name + "(d)")
im = ax.imshow(
    C_jackknife.T,
    extent=[z[0], z[-1], t_edges[0], t_edges[-1]],
    origin='lower',
    aspect='auto',
    # vmin=0,
    # vmax=0.01 * N,
    interpolation='nearest',
    norm=colors.LogNorm(1e-4, 1)
)

plt.legend(loc = "upper right")
cbar = plt.colorbar(im)
cbar.set_label("Marginal Density")
plt.xlabel("Distance from top of stalagmite [mm]")
plt.ylabel("Year CE")
plt.savefig(f"{output_dir}/age_depth_fig_jackknife.jpg")
# plt.show()

### Plotting OPES
fig, ax = plt.subplots(figsize=(6, 4))
plt.set_cmap(plt.cm.Greys)
for i, c in enumerate(data["cs"]):
    if i ==index: 
        ax.axvline(x=c, color='k', linestyle='--', linewidth = 1, label = f"d = {int(interesting_depth)} mm")
    else:
        ax.axvline(x=c, color='k', linestyle='--', linewidth = 0.1)

ax.plot(data["c14d"], np.squeeze(data["c14"]), "ko", markersize=4, label = r"$^{230}Th$")
ax.plot(data["cs"], true_ages, color = data_color, linewidth=3, alpha = 0.2, label = data_name + "(d)")
im = ax.imshow(
    C_opes.T,
    extent=[z[0], z[-1], t_edges[0], t_edges[-1]],
    origin='lower',
    aspect='auto',
    # vmin=0,
    # vmax=0.01 * N,
    interpolation='nearest',
    norm=colors.LogNorm(1e-4, 1)
)

plt.legend(loc = "upper right")
cbar = plt.colorbar(im)
cbar.set_label("Marginal Density")
plt.xlabel("Distance from top of stalagmite [mm]")
plt.ylabel("Year CE")
plt.savefig(f"{output_dir}/age_depth_fig_opes.jpg")
# plt.show()

# 3. Plotting jackknife along depth
interp_depth = 50
ts = np.arange(t_edges[0], t_edges[-1])
bin_edges = np.arange(np.floor(np.min(age[interp_depth])), np.ceil(np.max(age[interp_depth])) + 1, dtype = int)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(bin_centers, C_jackknife[int(interp_depth/dz), bin_edges[:-1] - t_edges[0]], yerr=2*sigma_est_jackknife[int(interp_depth/dz), bin_edges[:-1] - t_edges[0]], fmt='none', color = "grey", capsize=3, label = r"2$\sigma$ errorbar")
ax.bar(bin_centers,  C_jackknife[int(interp_depth/dz), bin_edges[:-1] - t_edges[0]], width=np.diff(bin_edges), alpha=0.3, color='gray', align='center')
ax.set_ylim(0, ylim_max)
ax.set_xlim(bin_range[0], bin_range[1])
ax.set_xlabel("Year CE")
ax.set_ylabel("Marginal Density")
ax.axvline(x=true_ages[index], color=data_color, linestyle='--', linewidth = 1, label = data_name + f"({int(interesting_depth)} mm)")
plt.legend(loc = "upper right")
plt.savefig(f"{output_dir}/samples_along_depth_{interp_depth}_jackknife.jpg")
# plt.show()

# 3. Plotting opes along depth
interp_depth = 50
ts = np.arange(t_edges[0], t_edges[-1])
bin_edges = np.arange(np.floor(np.min(age[interp_depth])), np.ceil(np.max(age[interp_depth])) + 1, dtype = int)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(bin_centers, C_opes[int(interp_depth/dz), bin_edges[:-1] - t_edges[0]], yerr=2*sigma_est_opes[int(interp_depth/dz), bin_edges[:-1] - t_edges[0]], fmt='none', color = "grey", capsize=3, label = r"2$\sigma$ errorbar")
ax.bar(bin_centers,  C_opes[int(interp_depth/dz), bin_edges[:-1] - t_edges[0]], width=np.diff(bin_edges), alpha=0.3, color='gray', align='center')
ax.set_ylim(0, ylim_max)
ax.set_xlim(bin_range[0], bin_range[1])
ax.set_xlabel("Year CE")
ax.set_ylabel("Marginal Density")
ax.axvline(x=true_ages[index], color=data_color, linestyle='--', linewidth = 1, label = data_name + f"({int(interesting_depth)} mm)")
plt.legend(loc = "upper right")
plt.savefig(f"{output_dir}/samples_along_depth_{interp_depth}_opes.jpg")
# plt.show()
