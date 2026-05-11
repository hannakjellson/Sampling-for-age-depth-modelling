import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from define_data_and_variables import opes_config, data, hash_configs, version
from matplotlib import colors


base_dir = Path.cwd()
base_path = base_dir / ".." / ".." / ".." / ".." / "data"
data_name =rf"$A_{{{data['dn'][:3]}{'e' if 'edit' in data['dn'] else ''}}}$"
true_sample = np.load(base_path / data["dn"] / version / "true_sample.npy")
int_version = int(version)
true_ages = np.hstack((data["th"], data["th"] - np.cumsum(true_sample) * data["dc"]))
c14_label = r"$^{230}$Th" if "d18o" in data["dn"] else r"Tephra Layer"
measure = "Distance from top of stalagmite [mm]" if "d18o" in data["dn"] else "Depth of sediment [cm]"

sp_dir = base_dir / f"output/{data['dn']}/{version}"
np.random.seed(opes_config["hmcc"]["sd"])
sp = np.random.lognormal(mean = data["pm"], sigma = data["ps"], size = (opes_config["hmcc"]["nch"], data["N"]))
opes_config["hmcc"]["sp"] = np.ascontiguousarray(sp)
opes_hash = hash_configs(opes_config, data)
output_dir = sp_dir / f"{opes_hash}"
hmc_config = opes_config["hmcc"]

cutout = 20000
bias_values = np.load(f"{output_dir}/bias.npy", mmap_mode='r')[:, cutout:]
d18o_energy_values = np.load(f"{output_dir}/d18o_energy.npy", mmap_mode='r')[:, cutout:]
samples = np.load(output_dir / "samples.npy", mmap_mode='r')[:, cutout:, :]
temp_index = 22
extra_label = f"_{temp_index}" if temp_index!=0 else ""
print(1/opes_config["bs"][temp_index])
weight_exponent = bias_values + (1-opes_config["bs"][temp_index]) * d18o_energy_values
weights = np.exp(weight_exponent - np.max(weight_exponent))
print(weights)

index = 39
K = 100
t_edges = [1000, 2010]

dt = 1 if not "biw" in data["dn"] else 10
dz = 0.1 if not "biw" in data["dn"] else 1
ylim = [-8000, 2000] if "biw" in data["dn"] else [1200, 2000]
interesting_depth = index * data["dc"]

flat_samples = samples.reshape(-1, data["N"])
flat_weights = weights.reshape(-1)
sum_flat_weights = np.sum(flat_weights)
ns_total = len(flat_samples[:, 0])
block_size = int(ns_total/K)
block_weight_sum = np.zeros((K))

for k in range(K):
    start, end = k * block_size, (k + 1) * block_size
    b_w = flat_weights[start:end]
    block_weight_sum[k] = np.sum(b_w)
block_weight_sum_squared = block_weight_sum**2

age = np.hstack([
    np.ones((ns_total, 1)) * data["th"],
    data["th"] - data["dc"] * np.cumsum(flat_samples, axis=1)
])

# Depth interpolation grid
z = np.arange(0, data["H"] + dz, dz)
Z = z.size

interp_idx = np.searchsorted(data["cs"], z) - 1
interp_idx = np.clip(interp_idx, 0, len(data["cs"]) - 2)

x0 = data["cs"][interp_idx]
x1 = data["cs"][interp_idx + 1]

alpha = (z - x0) / (x1 - x0)

# Time interpolation grid
t_bin_edges = np.arange(t_edges[0], t_edges[-1] + dt, dt, dtype = int)
t = t_bin_edges[1:] - dt / 2

# C_jackknife = np.full((Z, np.shape(t)[0]), np.nan)
C_opes = np.full((Z, np.shape(t)[0]), np.nan)
# sigma_est_jackknife = np.full((Z, np.shape(t)[0]), np.nan)
sigma_est_opes = np.full((Z, np.shape(t)[0]), np.nan)

for i in range(0, Z):
    # ages = np.array([np.interp(z[i], data["cs"], age[j, :]) for j in range(ns_total)])
    j = interp_idx[i]
    ages = (
        (1 - alpha[i]) * age[:, j]
        + alpha[i] * age[:, j + 1]
    )
    start = (np.min(ages) // dt) * dt
    stop = (np.max(ages) // dt + 1) * dt + dt
    bins = np.arange(start, stop, dt, dtype = int)
    if bins.size < 2:
        bins = np.array([start, start + dt], dtype = int)
    
    bin_center_indices = ((bins[:-1] - t_edges[0])/dt).astype(int)
    bin_age_indices = np.digitize(ages, bins) - 1
    num_bins = len(bins) - 1

    
    # Computing full histogram
    full_hist = np.full((num_bins), np.nan)
    for b in range(num_bins):
        full_hist[b] = np.sum(flat_weights[bin_age_indices == b])
    norm_full_hist = full_hist / sum_flat_weights

    # Computing block histograms
    block_hists = np.zeros((K, num_bins))
    for k in range(K):
        start, end = k * block_size, (k + 1) * block_size
        b_idx = bin_age_indices[start:end]
        b_w = flat_weights[start:end]
        
        # Sum weights in this block
        block_hists[k] = np.bincount(
            b_idx,
            weights=b_w,
            minlength=num_bins
        )

    # Jackknife estimate
    # loo_results = full_hist - block_hists
    # norm_loo_results  = loo_results / (sum_flat_weights - block_weight_sum[:, None])
    # jackknife_est = K * norm_full_hist - (K - 1) * np.mean(norm_loo_results, axis=0)
    # C_jackknife[i, bin_center_indices] = jackknife_est
    # sigma_est_jackknife[i, bin_center_indices]  = np.sqrt((K-1) * np.sum((norm_loo_results - jackknife_est)**2, axis = 0) / K)

    # OPES estimate
    norm_block_results = block_hists/block_weight_sum[:, None]
    opes_est = block_weight_sum@norm_block_results/sum_flat_weights
    meff = np.sum(block_weight_sum)**2 / np.sum(block_weight_sum_squared)
    C_opes[i, bin_center_indices] = opes_est
    sigma_est_opes[i, bin_center_indices] = np.sqrt((block_weight_sum @ (norm_block_results - norm_full_hist)**2 / ((meff - 1) * sum_flat_weights)))
    print(f"progress: {i/Z:.2f}")

print("Saving")
# np.save(f"{output_dir}/C_jackknife" + extra_label, C_jackknife)
# np.save(f"{output_dir}/sigma_est_jackknife" + extra_label, sigma_est_jackknife)
np.save(f"{output_dir}/C_opes" + extra_label, C_opes)
np.save(f"{output_dir}/sigma_est_opes" + extra_label, sigma_est_opes)

# C_jackknife = np.load(f"{output_dir}/C_jackknife" + extra_label + ".npy")
# sigma_est_jackknife = np.load(f"{output_dir}/sigma_est_jackknife" + extra_label + ".npy")
C_opes = np.load(f"{output_dir}/C_opes" + extra_label + ".npy")
sigma_est_opes = np.load(f"{output_dir}/sigma_est_opes" + extra_label + ".npy")

print("Plotting")
### Plotting Jackknife
# fig, ax = plt.subplots(figsize=(6, 4))
# plt.set_cmap(plt.cm.Greys)
# for i, c in enumerate(data["cs"]):
#     if i ==index: 
#         ax.axvline(x=c, color='k', linestyle='--', linewidth = 1, label = f"d = {int(interesting_depth)} " + measure[-3:-1])
#     else:
#         ax.axvline(x=c, color='k', linestyle='--', linewidth = 0.1)

# plt.errorbar(data["c14d"], data["c14"], yerr = data["c14s"], fmt ="o", color = 'k', capsize = 2, markersize=4, label = c14_label)
# ax.plot(data["cs"], true_ages, color = "red", linewidth=3, alpha = 0.2, label = data_name + "(d)")
# im = ax.pcolormesh(z, t, C_jackknife.T, norm=colors.LogNorm(1e-4, 1))

# plt.legend(loc = "upper right")
# cbar = plt.colorbar(im)
# cbar.set_label("Marginal Density")
# plt.ylim(ylim)
# plt.xlabel(measure)
# plt.ylabel("Year CE")
# plt.tight_layout()
# plt.savefig(f"{output_dir}/age_depth_fig_jackknife" + extra_label + ".jpg")
# plt.show()

### Plotting OPES
fig, ax = plt.subplots(figsize=(6, 4))
plt.set_cmap(plt.cm.Greys)
for i, c in enumerate(data["cs"]):
    if i ==index: 
        ax.axvline(x=c, color='k', linestyle='--', linewidth = 1, label = f"d = {int(interesting_depth)} " + measure[-3:-1])
    else:
        ax.axvline(x=c, color='k', linestyle='--', linewidth = 0.1)

ax.plot(data["c14d"], np.squeeze(data["c14"]), "ko", markersize=4, label = c14_label)
ax.plot(data["cs"], true_ages, color = "red", linewidth=3, alpha = 0.2, label = data_name + "(d)")
im = ax.pcolormesh(z, t, C_opes.T, norm=colors.LogNorm(1e-4, 1))


plt.legend(loc = "upper right")
cbar = plt.colorbar(im)
cbar.set_label("Marginal Density")
plt.ylim(ylim)
plt.xlabel(measure)
plt.ylabel("Year CE")
plt.tight_layout()
plt.savefig(f"{output_dir}/age_depth_fig_opes" + extra_label + ".jpg")
# plt.show()

# 3. Plotting jackknife along depth
# jackknife_depth_C = C_jackknife[int(interesting_depth/dz), :]
# jackknife_depth_sigma = sigma_est_jackknife[int(interesting_depth/dz), :]
# sum_jk = jackknife_depth_C + 2*jackknife_depth_sigma
# valid_indices = np.where((~np.isnan(sum_jk)) & (sum_jk > 0.0001))[0]
# first_valid_index, last_valid_index = valid_indices[0], valid_indices[-1]

# jackknife_depth_C = jackknife_depth_C[first_valid_index:last_valid_index + 1]
# jackknife_depth_sigma = jackknife_depth_sigma[first_valid_index:last_valid_index + 1]
# sum_jk = sum_jk[first_valid_index:last_valid_index + 1]
# bin_edges = t_bin_edges[first_valid_index:last_valid_index + 2]

# fig, ax = plt.subplots(figsize=(6, 4))
# ax.errorbar(t[first_valid_index:last_valid_index + 1], jackknife_depth_C, yerr=2*jackknife_depth_sigma, fmt='none', color = "grey", capsize=3, label = r"2$\sigma$ errorbar")
# ax.bar(t[first_valid_index:last_valid_index + 1],  jackknife_depth_C, width=np.diff(bin_edges), alpha=0.3, color='gray', align='center')
# ax.set_ylim(0, np.max(sum_jk) + 0.01)
# ax.set_xlim(bin_edges[0], bin_edges[-1])
# ax.set_xlabel("Year CE")
# ax.set_ylabel("Marginal Density")
# ax.axvline(x=true_ages[index], color="red", linestyle='--', linewidth = 1, label = data_name + f"({int(interesting_depth)} " + measure[-3:-1] + ")")
# plt.legend(loc = "upper right")
# plt.savefig(f"{output_dir}/samples_along_depth_{interesting_depth}_jackknife" + extra_label + ".jpg")
# plt.show()

# 3. Plotting opes along depth
ts = np.arange(t_edges[0], t_edges[-1] + dt)
opes_depth_C = C_opes[int(interesting_depth/dz), :]
opes_depth_sigma = sigma_est_opes[int(interesting_depth/dz), :]
sum_op = opes_depth_C + 2*opes_depth_sigma
valid_indices = np.where((~np.isnan(sum_op)) & (sum_op > 0.0001))[0]
first_valid_index, last_valid_index = valid_indices[0], valid_indices[-1]

opes_depth_C = opes_depth_C[first_valid_index:last_valid_index+1]
opes_depth_sigma = opes_depth_sigma[first_valid_index:last_valid_index+1]
sum_op = sum_op[first_valid_index:last_valid_index+1]
bin_edges = t_bin_edges[first_valid_index:last_valid_index + 2]

fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(t[first_valid_index:last_valid_index + 1], opes_depth_C, yerr=2*opes_depth_sigma, fmt='none', color = "grey", capsize=3, label = r"2$\sigma$ errorbar")
ax.bar(t[first_valid_index:last_valid_index + 1],  opes_depth_C, width=np.diff(bin_edges), alpha=0.3, color='gray', align='center')
ax.set_ylim(0, np.max(sum_op) + 0.01)
ax.set_xlim(bin_edges[0], bin_edges[-1])
ax.set_xlabel("Year CE")
ax.set_ylabel("Marginal Density")
ax.axvline(x=true_ages[index], color="red", linestyle='--', linewidth = 1, label = data_name + f"({int(interesting_depth)} " + measure[-3:-1] + ")")
plt.legend(loc = "upper right")
plt.savefig(f"{output_dir}/samples_along_depth_{interesting_depth}_opes" + extra_label + ".jpg")
# plt.show()
