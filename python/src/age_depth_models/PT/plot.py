import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from define_data_and_variables import pt_config, data, hash_configs, version
from matplotlib import colors

base_dir = Path.cwd()

base_path = base_dir / ".." / ".." / ".." / ".." / "data"
data_name = rf"$A_{{{data['dn'][:3]}}}$" if not "biw" in data['dn'] else rf"$A_{{{data['dn'][:3] }}}$"
true_sample = np.load(base_path / data["dn"] / version / "true_sample.npy")
int_version = int(version)
true_ages = np.hstack((data["th"], data["th"] - np.cumsum(true_sample) * data["dc"]))
c14_label = r"$^{230}$Th" if "d18o" in data["dn"] else r"Tephra Layer"
measure = "mm" if "d18o" in data["dn"] else "cm"

base_dir = Path.cwd()
sp_dir = base_dir / f"output/{data['dn']}/{version}"
np.random.seed(pt_config['hmcc']['sd'])
sp = np.random.lognormal(mean = data["pm"], sigma = data["ps"], size = (pt_config["nt"], data["N"]))
pt_config["hmcc"]["sp"] = np.ascontiguousarray(sp)

pt_hash = hash_configs(pt_config, data)
output_dir = sp_dir / f"{pt_hash}"

hmc_config = pt_config["hmcc"]

cutout = 10000
samples = np.load(output_dir / "samples.npy", mmap_mode='r')[:, cutout:, :]

index = 39
interesting_depth = index * data["dc"]
dt = 1
K = 100

flat_samples = samples[7, :, :]
print(1/pt_config["bs"][7])
flat_weights = np.ones((len(flat_samples)))
ns = len(flat_samples[:, 0])

age = np.hstack([
    np.ones((ns, 1)) * data["th"],
    data["th"] - data["dc"] * np.cumsum(flat_samples, axis=1)
])

# Depth interpolation grid
dz = 0.1
z = np.arange(0, data["H"] + dz, dz)
Z = z.size

t_edges = [1000, 2010]
ts = np.arange(t_edges[0], t_edges[-1] + dt, dt, dtype = int)

block_size = int(ns/K)
# C_jackknife = np.full((Z, np.shape(ts)[0]), np.nan)
# sigma_est_jackknife = np.full((Z, np.shape(ts)[0]), np.nan)

# for i in range(0, Z):
#     hists = []
#     ages = np.array([np.interp(z[i], data["cs"], age[j, :]) for j in range(ns)])
#     start = (np.min(ages) // dt) * dt
#     stop = (np.max(ages) // dt + 1) * dt + dt
#     bins = np.arange(start, stop, dt, dtype = int)
#     if bins.size < 2:
#         bins = np.array([start, start + dt], dtype = int)
    
#     bin_indices = np.digitize(ages, bins) - 1
#     num_bins = len(bins) - 1

#     for b in range(num_bins):
#         hists.append(np.sum(flat_weights[bin_indices == b]))

#     full_hist = np.array(hists)

#     block_hists = np.zeros((K, num_bins))
#     block_weight_sum = np.zeros((K))
#     for k in range(K):
#         start, end = k * block_size, (k + 1) * block_size
#         # Slice the block's indices and weights
#         b_idx = bin_indices[start:end]
#         b_w = flat_weights[start:end]
#         block_weight_sum[k] = np.sum(b_w)
        
#         # Sum weights in this block
#         for b in range(num_bins):
#             block_hists[k, b] = np.sum(b_w[b_idx == b])

#     norm_full_hist = full_hist / np.sum(flat_weights)

#     # Jackknife estimate
#     loo_results = full_hist - block_hists
#     norm_loo_results  = loo_results / (np.sum(flat_weights) - block_weight_sum[:, None])
#     jackknife_est = K * norm_full_hist - (K - 1) * np.mean(norm_loo_results, axis=0)
#     C_jackknife[i, ((bins[:-1]-t_edges[0])/dt).astype(int)] = jackknife_est
#     sigma_est_jackknife[i, ((bins[:-1]-t_edges[0])/dt).astype(int)]  = np.sqrt((K-1) * np.sum((norm_loo_results - jackknife_est)**2, axis = 0) / K)

#     print(f"progress: {i/Z}")
#     # print(f"C_jackknife[i]: {C_jackknife[i][~np.isnan(C_jackknife[i])]}")
#     # print(f"sigma_est_jackknife[i]: {sigma_est_jackknife[i][~np.isnan(sigma_est_jackknife[i])]}")

# print("Saving")
# np.save(f"{output_dir}/C_jackknife_highT", C_jackknife)
# np.save(f"{output_dir}/sigma_est_jackknife_highT", sigma_est_jackknife)

C_jackknife = np.load(f"{output_dir}/C_jackknife_highT.npy")
sigma_est_jackknife = np.load(f"{output_dir}/sigma_est_jackknife_highT.npy")

print("Plotting")
fig, ax = plt.subplots(figsize=(6, 4))
plt.set_cmap(plt.cm.Greys)
for i, c in enumerate(data["cs"]):
    if i ==index: 
        ax.axvline(x=c, color='k', linestyle='--', linewidth = 1, label = f"d = {int(interesting_depth)} " + measure)
    else:
        ax.axvline(x=c, color='k', linestyle='--', linewidth = 0.1)

plt.errorbar(data["c14d"], data["c14"], yerr = data["c14s"], fmt ="o", color = 'k', capsize = 2, markersize=4, label = c14_label)
ax.plot(data["cs"], true_ages, color = "red", linewidth=3, alpha = 0.2, label = data_name + "(d)")
im = ax.imshow(
    C_jackknife.T,
    extent=[z[0], z[-1], t_edges[0], t_edges[-1]],
    origin='lower',
    aspect='auto',
    interpolation='nearest',
    norm=colors.LogNorm(1e-4, 1)
)

plt.legend(loc = "upper right")
cbar = plt.colorbar(im)
cbar.set_label("Marginal Density")
plt.ylim(1200, 2000)
# plt.xlabel("Depth of sediment [" + measure + "]")
plt.xlabel("Distance from top of stalagmite [" + measure + "]")
plt.ylabel("Year CE")
plt.tight_layout()
plt.savefig(f"{output_dir}/age_depth_fig_jackknife_highT.jpg")
plt.show()

# 3. Plotting jackknife along depth
jackknife_depth_C = C_jackknife[int(interesting_depth/dz), :]
jackknife_depth_sigma = sigma_est_jackknife[int(interesting_depth/dz), :]
sum_jk = jackknife_depth_C + 2*jackknife_depth_sigma
valid_indices = np.where((~np.isnan(sum_jk)) & (sum_jk > 0.0001))[0]
first_valid_index, last_valid_index = valid_indices[0], valid_indices[-1]

jackknife_depth_C = jackknife_depth_C[first_valid_index:last_valid_index + 1]
jackknife_depth_sigma = jackknife_depth_sigma[first_valid_index:last_valid_index + 1]
sum_jk = sum_jk[first_valid_index:last_valid_index + 1]
bin_edges = ts[first_valid_index:last_valid_index + 2]
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(bin_centers, jackknife_depth_C, yerr=2*jackknife_depth_sigma, fmt='none', color = "grey", capsize=3, label = r"2$\sigma$ errorbar")
ax.bar(bin_centers,  jackknife_depth_C, width=np.diff(bin_edges), alpha=0.3, color='gray', align='center')
ax.set_ylim(0, np.max(sum_jk) + 0.01)
ax.set_xlim(bin_edges[0], bin_edges[-1] + 2)
ax.set_xlabel("Year CE")
ax.set_ylabel("Marginal Density")
ax.axvline(x=true_ages[index], color="red", linestyle='--', linewidth = 1, label = data_name + f"({int(interesting_depth)} " + measure + ")")
plt.legend(loc = "upper right")
plt.tight_layout()
plt.savefig(f"{output_dir}/samples_along_depth_{interesting_depth}_jackknife_highT.jpg")
plt.show()
