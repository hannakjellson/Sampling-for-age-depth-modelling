import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from example_define_variables_md_adaptive import get_hmc_config
import arviz as az
from scipy.stats import norm
import imageio.v2 as imageio
from scipy.stats import gaussian_kde
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import Normalize


def energy_function(variables):
    return 1.34549*variables[0]**4 + 1.90211*variables[0]**3*variables[1]+ 3.92705*variables[0]**2*variables[1]**2-6.44246*variables[0]**2- 1.90211*variables[0]*variables[1]**3+5.58721*variables[0]*variables[1] + 1.33481*variables[0] + 1.34549*variables[1]**4- 5.55754*variables[1]**2+0.904586*variables[1] + 18.5598


def main(): # Next step is to send through the biases (height, width etc) here, and plot what i get out of that and to then try longer steps if it looks weird.
    valid_samples = np.load("../../../../output/example_with_bias_samples_md_adaptive_2d_new_kernel.npy")
    energy_values = np.load("../../../../output/example_with_bias_energy_values_md_adaptive_2d_new_kernel.npy")
    bias_values = np.load("../../../../output/example_with_bias_weights_md_adaptive_2d_new_kernel.npy")
    bias_std = np.load("../../../../output/example_with_bias_std_out_md_adaptive_2d_new_kernel.npy")

    config = get_hmc_config()
    x = np.linspace(-3, 3, 1000)
    y = np.linspace(-3, 3, 1000)
    dx, dy = x[1]-x[0], y[1]-y[0]
    X, Y = np.meshgrid(x, y, indexing="xy")

    U = energy_function([X, Y])

    P = np.exp(-U)
    P/=np.sum(P)*dx*dy
    true_marginal_likelihood = np.sum(P, axis = 0) * dy
    free_energy = -np.log(true_marginal_likelihood)
    free_energy-= np.min(free_energy)

    resampled_samples = []
    cutout = int(len(valid_samples)/3)

    weights = np.exp(bias_values)

    for i in range(config["num_chains"]):
        weights_i = weights[i, cutout:] / sum(weights[i, cutout:])
        indices = np.random.choice(
            np.arange(cutout, len(weights_i) + cutout),
            size=int(len(weights_i) / 4),
            replace=True,
            p=weights_i,
        )
        resampled_samples.append(valid_samples[i, indices, :])
    resampled_samples = np.array(resampled_samples)

    fig, axs= plt.subplots(1, config["num_chains"] + 1, figsize=(5*config["num_chains"], 5))
    norm = colors.Normalize(vmin=P.min(), vmax=P.max())

    axs[0].pcolormesh(X, Y, -np.log(P), cmap="YlGnBu_r", norm = norm, shading = "auto")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_title(f"Original probability")

    for i in range(1, config["num_chains"] + 1):
        # Compute a 2D histogram and normalize to get a probability density
        hist, xedges, yedges = np.histogram2d(
            resampled_samples[i-1, :, 0], resampled_samples[i-1, :, 1],
            bins=100, range=[[-3, 3], [-3, 3]], density=True
        )

        # Grid centers
        xpos = (xedges[:-1] + xedges[1:]) / 2
        ypos = (yedges[:-1] + yedges[1:]) / 2
        X, Y = np.meshgrid(xpos, ypos, indexing='ij')

        # Plot as a 2D color map
        cmap = cm.YlGnBu_r
        norm = colors.Normalize(vmin=hist.min(), vmax=hist.max())
        pcm = axs[i].pcolormesh(X, Y, -np.log(hist + 1e-12), cmap=cmap, norm=norm, shading='auto')

        axs[i].set_xlabel("X")
        axs[i].set_ylabel("Y")
        axs[i].set_title(f"Chain {i}")

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(5*config["num_chains"], 5))
    # plt.plot(valid_samples[0,:,0])
    plt.hist(valid_samples[0,:,0], bins = 100)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(5*config["num_chains"], 5))
    counts, bin_edges = np.histogram(resampled_samples[0,:, 0], bins=20, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts[counts == 0] = 1e-12  
    free_energy_est_reweight = -np.log(counts)
    free_energy_est_reweight-= np.min(free_energy_est_reweight)
    ax.plot(x, free_energy, color="black", label=f"true free energy")
    ax.plot(bin_centers, free_energy_est_reweight, '-o', label="free energy estimate after reweight")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(5*config["num_chains"], 5))
    ax.hist(resampled_samples[0, :, 0], bins=1000, color="red", density=True, alpha=0.3, label=f"chain {0}")
    ax.legend()
    ax.plot(x, true_marginal_likelihood, color="black")
    plt.show()

if __name__ == "__main__":
    main()
