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
print(base_dir)

base_path = base_dir / ".." / ".." / ".." / ".." / "data"
print(data["dn"].lower())
if data["dn"].lower() == "dayu19a":
    true_sample = np.load(base_path / "inputdata_260219A" / "true_sample.npy")
    bin_range = (1676, 1727)
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
    bin_range = (1680, 1720)
    data_color = "yellow"
    data_name = r"$A_3$"
    ylim_max = 0.25
print(true_sample)
true_ages = np.hstack((data["th"], data["th"] - np.cumsum(true_sample) * data["dc"]))
print(true_ages)

base_dir = Path.cwd()
adam_dir = base_dir / f"output/{data['dn']}/sd_{opes_config['hmcc']['sd']}"

np.random.seed(opes_config["hmcc"]["sd"])
sp = np.random.lognormal(mean = data["pm"], sigma = data["ps"], size = (opes_config["hmcc"]["nch"], data["N"]))

opes_config["hmcc"]["sp"] = np.ascontiguousarray(sp)
opes_config["df"] = 140 * (opes_config["bs"] - 1)
print(opes_config["df"][-1])
opes_config["dfn"] = np.exp(-opes_config["df"])*opes_config["dfd"]

opes_hash = hash_configs(opes_config, data)
output_dir = adam_dir / f"{opes_hash}"

hmc_config = opes_config["hmcc"]

cutout = 10000
bias_values = np.load(f"{output_dir}/bias.npy", mmap_mode='r')[:, cutout:]
energy_values = np.load(f"{output_dir}/energy.npy", mmap_mode='r')[:, cutout:]
samples = np.load(output_dir / "samples.npy", mmap_mode='r')[:, cutout:, :]
weights = np.exp(bias_values)

chain = 0
index = 25
interesting_depth = index * data["dc"]
dt = 1
print(np.shape(weights))
block_size = int(len(samples[chain, :, 0])/5000)

ns = len(samples[0, :, 0])
age = np.hstack([
    np.ones((ns, 1)) * data["th"],
    data["th"] - data["dc"] * np.cumsum(samples[chain, :, :], axis=1)
])

# Depth interpolation grid
dz = 0.1
z = np.arange(0, data["H"] + dz, dz)
Z = z.size

t_edges = [1200, 2000]

K = int(len(samples[0, :, 0])/block_size)
C = np.full((Z, t_edges[-1] - t_edges[0]), np.nan)
for i in range(0, Z):
    hists = []
    ages = np.array([np.interp(z[i], data["cs"], age[j, :]) for j in range(ns)])
    print(np.ceil(np.max(ages)))
    bins = np.arange(np.floor(np.min(ages)), np.ceil(np.max(ages)) + 1, dtype = int)
    if bins.size < 2:
        bins = np.array([np.floor(np.min(ages)), np.ceil(np.max(ages)) + 1], dtype = int)
        print("hi")
    
    print(ages)
    print(f"bins:{bins}")
    bin_indices = np.digitize(ages, bins) - 1
    num_bins = len(bins) - 1

    for b in range(num_bins):
        hists.append(np.sum(weights[chain, bin_indices == b]))

    full_hist = np.array(hists / (np.sum(weights[chain])))

    block_hists = np.zeros((K, num_bins))
    for k in range(K):
        start, end = k * block_size, (k + 1) * block_size
        # Slice the block's indices and weights
        b_idx = bin_indices[start:end]
        b_w = weights[chain, start:end]
        
        # Sum weights in this block
        for b in range(num_bins):
            block_hists[k, b] = np.sum(b_w[b_idx == b])

    loo_results = full_hist - block_hists
    norm_loo_results  = loo_results / (np.sum(weights[chain]) - np.array([np.sum(weights[chain, k*block_size:(k+1)*block_size]) for k in range(K)])[:, None])
    jackknife_est = ns * full_hist - (ns - 1) * np.mean(norm_loo_results, axis=0)
    C[i, bins[:-1]-t_edges[0]] = jackknife_est
    print(i/Z)

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
    C.T,
    extent=[z[0], z[-1], t_edges[0], t_edges[-1]],
    origin='lower',
    aspect='auto',
    # vmin=0,
    # vmax=0.01 * N,
    norm=colors.LogNorm()
)

plt.legend(loc = "upper right")
cbar = plt.colorbar(im)
cbar.set_label("Marginal Density")
plt.xlabel("Distance from top of stalagmite [mm]")
plt.ylabel("Year CE")
plt.savefig(f"{output_dir}/resampled_no_mean.jpg")
np.save(f"{output_dir}/C_anders.npz", C)


ns = len(samples[0, :, 0])
age = np.hstack([
    np.ones((ns, 1)) * data["th"],
    data["th"] - data["dc"] * np.cumsum(samples[chain, :, :], axis=1)
])

# Depth interpolation grid
dz = 0.1
z = np.arange(0, data["H"] + dz, dz)
Z = z.size

t_edges = [1200, 2000]

K = int(len(samples[0, :, 0])/block_size)
C = np.full((Z, t_edges[-1] - t_edges[0]), np.nan)
var_est = []
for i in range(1, Z):
    hists = []
    ages = np.array([np.interp(z[i], data["cs"], age[j, :]) for j in range(ns)])
    print(np.ceil(np.max(ages)))
    bins = np.arange(np.floor(np.min(ages)), np.ceil(np.max(ages)) + 1, dtype = int)
    if bins.size < 2:
        bins = np.array([np.floor(np.min(ages)), np.ceil(np.max(ages)) + 1])
    
    bin_indices = np.digitize(ages, bins) - 1
    num_bins = len(bins) - 1

    for b in range(num_bins):
        hists.append(np.sum(weights[chain, bin_indices == b]))

    full_hist = np.array(hists / (np.sum(weights[chain])))

    block_hists = np.zeros((K, num_bins))
    block_weights_sum = np.zeros((K))
    for k in range(K):
        start, end = k * block_size, (k + 1) * block_size
        # Slice the block's indices and weights
        b_idx = bin_indices[start:end]
        b_w = weights[chain, start:end]
        block_weights_sum[k] = np.sum(b_w)
        
        # Sum weights in this block
        for b in range(num_bins):
            block_hists[k, b] = np.sum(b_w[b_idx == b])

    est = np.sum(block_weights_sum*block_hists, axis = 0)/np.sum(block_weights_sum)
    meff = np.sum(block_weights_sum)**2 / np.sum(block_weights_sum**2)
    var_est.append(meff * np.sum(block_weights_sum * (block_hists - full_hist)**2, axis = 0) / ((meff - 1) *np.sum(block_weights_sum)))
    C[i, bins[:-1]-t_edges[0]] = est
    print(i/Z)

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
    C.T,
    extent=[z[0], z[-1], t_edges[0], t_edges[-1]],
    origin='lower',
    aspect='auto',
    # vmin=0,
    # vmax=0.01 * N,
    norm=colors.LogNorm()
)

plt.legend(loc = "upper right")
cbar = plt.colorbar(im)
cbar.set_label("Marginal Density")
plt.xlabel("Distance from top of stalagmite [mm]")
plt.ylabel("Year CE")
plt.savefig(f"{output_dir}/resampled_no_mean_other.jpg")
np.save(f"{output_dir}/C_opes.npz", C)

