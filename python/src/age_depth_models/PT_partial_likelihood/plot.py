import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from define_data_and_variables import pt_config, data, hash_configs, version
from matplotlib import colors

base_dir = Path.cwd()
base_path = base_dir / ".." / ".." / ".." / ".." / "data"
data_name =rf"$A_{{{data['dn'][:3]}{'e' if 'edit' in data['dn'] else ''}}}$"
true_sample = np.load(base_path / data["dn"] / version / "true_sample.npy")
highT = "_highT" if "wah" in data["dn"] else ""
int_version = int(version)
true_ages = np.hstack((data["th"], data["th"] - np.cumsum(true_sample) * data["dc"]))
c14_label = r"$^{230}$Th" if "d18o" in data["dn"] else r"Tephra Layer"
measure = "Distance from top of stalagmite [mm]" if "d18o" in data["dn"] else "Depth of sediment [cm]"

sp_dir = base_dir / f"output/{data['dn']}/{version}"
np.random.seed(pt_config['hmcc']['sd'])
sp = np.random.lognormal(mean = data["pm"], sigma = data["ps"], size = (pt_config["nt"], data["N"]))
pt_config["hmcc"]["sp"] = np.ascontiguousarray(sp)
pt_hash = hash_configs(pt_config, data)
output_dir = sp_dir / f"{pt_hash}"
hmc_config = pt_config["hmcc"]

cutout = 10000
samples = np.load(output_dir / "samples.npy", mmap_mode='r')[:, cutout:, :]

index = 39 #Index for interesting depth
K = 100 #Blocksize
temp_index = 7
t_edges = [1000, 2010]

dt = 1 if not "biw" in data["dn"] else 10
dz = 0.1 if not "biw" in data["dn"] else 1
ylim = [-8000, 2000] if "biw" in data["dn"] else [1200, 2000]
interesting_depth = index * data["dc"]

temp_samples = samples[temp_index, :, :]
print(1/pt_config["bs"][temp_index])
ns_cutout = len(temp_samples[:, 0])
one_array = np.ones((ns_cutout))

age = np.hstack([
    np.ones((ns_cutout, 1)) * data["th"],
    data["th"] - data["dc"] * np.cumsum(temp_samples, axis=1)
])

# Depth interpolation grid
z = np.arange(0, data["H"] + dz, dz)
Z = z.size

# Time interpolation grid
t_bin_edges = np.arange(t_edges[0], t_edges[-1] + dt, dt, dtype = int)
t = t_bin_edges[1:] - dt / 2

block_size = int(ns_cutout/K)
# C = np.full((Z, np.shape(t)[0]), np.nan)
# sigma_est = np.full((Z, np.shape(t)[0]), np.nan)

# for i in range(0, Z):
#     ages = np.array([np.interp(z[i], data["cs"], age[j, :]) for j in range(ns_cutout)])
#     start = (np.min(ages) // dt) * dt
#     stop = (np.max(ages) // dt + 1) * dt + dt
#     bins = np.arange(start, stop, dt, dtype = int)
#     if bins.size < 2:
#         bins = np.array([start, start + dt], dtype = int)

#     bin_center_indices = ((bins[:-1] - t_edges[0])/dt).astype(int)
#     bin_age_indices = np.digitize(ages, bins) - 1
#     num_bins = len(bins) - 1

#     # Computing full histogram
#     full_hist = np.full((num_bins), np.nan)
#     for b in range(num_bins):
#         full_hist[b] = np.sum(one_array[bin_age_indices == b])
#     full_hist /= ns_cutout

#     # Computing block histograms
#     block_hists = np.zeros((K, num_bins))
#     block_weight_sum = np.zeros((K))
#     for k in range(K):
#         start, end = k * block_size, (k + 1) * block_size
#         b_idx = bin_age_indices[start:end]
#         for b in range(num_bins):
#             block_hists[k, b] = np.sum(one_array[start:end][b_idx == b])

#     block_hists /=block_size
#     # Allocating estimates mean histogram and standard deviation
#     C[i, bin_center_indices] = np.mean(block_hists, axis = 0)
#     sigma_est[i, bin_center_indices] = np.sqrt(1/(K*(K-1))*np.sum((block_hists - full_hist)**2, axis = 0))

#     print(f"progress: {i/Z}")

# print("Saving")
# np.save(f"{output_dir}/C" + highT + ".npy", C)
# np.save(f"{output_dir}/sigma_est" + highT + ".npy", sigma_est)

C = np.load(f"{output_dir}/C"  + highT + ".npy")
sigma_est = np.load(f"{output_dir}/sigma_est" + highT + ".npy")

print("Plotting")
fig, ax = plt.subplots(figsize=(6, 4))
plt.set_cmap(plt.cm.Greys)
for i, c in enumerate(data["cs"]):
    if i ==index: 
        ax.axvline(x=c, color='k', linestyle='--', linewidth = 1, label = f"d = {int(interesting_depth)} " + measure[-3:-1])
    else:
        ax.axvline(x=c, color='k', linestyle='--', linewidth = 0.1)

plt.errorbar(data["c14d"], data["c14"], yerr = data["c14s"], fmt ="o", color = 'k', capsize = 2, markersize=4, label = c14_label)
ax.plot(data["cs"], true_ages, color = "red", linewidth=3, alpha = 0.2, label = data_name + "(d)")
im = ax.pcolormesh(z, t, C.T, norm=colors.LogNorm(1e-4, 1))

plt.legend(loc = "upper right")
cbar = plt.colorbar(im)
cbar.set_label("Marginal Density")
plt.ylim(ylim)
plt.xlabel(measure)
plt.ylabel("Year CE")
plt.tight_layout()
plt.savefig(f"{output_dir}/age_depth_fig"  + highT + ".jpg")
plt.show()

# 3. Plotting estimate along depth
depth_C = C[int(interesting_depth/dz), :]
depth_sigma = sigma_est[int(interesting_depth/dz), :]
sum_jk = depth_C + 2*depth_sigma
valid_indices = np.where((~np.isnan(sum_jk)) & (sum_jk > 0.0001))[0]
first_valid_index, last_valid_index = valid_indices[0], valid_indices[-1]

depth_C = depth_C[first_valid_index:last_valid_index + 1]
depth_sigma = depth_sigma[first_valid_index:last_valid_index + 1]
sum_jk = sum_jk[first_valid_index:last_valid_index + 1]
bin_edges = t_bin_edges[first_valid_index:last_valid_index + 2]

fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(t[first_valid_index:last_valid_index + 1], depth_C, yerr=2*depth_sigma, fmt='none', color = "grey", capsize=3, label = r"2$\sigma$ errorbar")
ax.bar(t[first_valid_index:last_valid_index + 1],  depth_C, width=np.diff(bin_edges), alpha=0.3, color='gray', align='center')
ax.set_ylim(0, np.nanmax(sum_jk) + 0.01)
ax.set_xlim(bin_edges[0], bin_edges[-1])
ax.set_xlabel("Year CE")
ax.set_ylabel("Marginal Density")
ax.axvline(x=true_ages[index], color="red", linestyle='--', linewidth = 1, label = data_name + f"({int(interesting_depth)} " + measure[-3:-1] + ")")
plt.legend(loc = "upper right")
plt.tight_layout()
plt.savefig(f"{output_dir}/samples_along_depth_{interesting_depth}" + highT + ".jpg")
plt.show()
