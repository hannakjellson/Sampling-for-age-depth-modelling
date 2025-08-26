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
    # angle = -0.6 * np.pi / 4

    # rotated_X = np.cos(angle) * variables[0] - np.sin(angle) * variables[1]
    # rotated_Y = np.sin(angle) * variables[0] + np.cos(angle) * variables[1]

    # return (rotated_X**4 +
    #         rotated_Y**4 -
    #         2 * rotated_X**2 -
    #         4 * rotated_Y**2 +
    #         rotated_X * rotated_Y +
    #         0.3 * rotated_X +
    #         0.1 * rotated_Y)

    return 1.34549*variables[0]**4 + 1.90211*variables[0]**3*variables[1]+ 3.92705*variables[0]**2*variables[1]**2-6.44246*variables[0]**2- 1.90211*variables[0]*variables[1]**3+5.58721*variables[0]*variables[1] + 1.33481*variables[0] + 1.34549*variables[1]**4- 5.55754*variables[1]**2+0.904586*variables[1] + 18.5598


def main(): # Next step is to send through the biases (height, width etc) here, and plot what i get out of that and to then try longer steps if it looks weird.
    valid_samples = np.load("../../../../output/example_with_bias_samples_md_adaptive_2d_new_kernel_new_energy.npy")
    energy_values = np.load("../../../../output/example_with_bias_energy_values_md_adaptive_2d_new_kernel_new_energy.npy")
    bias_values = np.load("../../../../output/example_with_bias_weights_md_adaptive_2d_new_kernel_new_energy.npy")
    bias_std = np.load("../../../../output/example_with_bias_std_out_md_adaptive_2d_new_kernel_new_energy.npy")

    # print(valid_samples[3, :, :])
    # print(bias_values[3, :])

    # print(valid_samples[2, :, :])
    # print(bias_values[2, :])

    config = get_hmc_config()
    x = np.linspace(-3, 3, 50)
    y = np.zeros(50)
    color_vec = plt.cm.viridis(np.linspace(0, 1, config["num_chains"]))

    x_vec = np.array([x, y])

    energy_vec = np.vectorize(lambda a, b: energy_function([a, b]))
    energy_vals = energy_vec(x_vec[0], x_vec[1])

    weights = np.exp(bias_values)
    # print(weights)
    # print(valid_samples)
    # return


    resampled_samples = []
    cutout = int(len(valid_samples)/3)

    # plt.figure()
    # plt.plot(x, energy_vals)
    # plt.show()

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

    # Create a grid of x and y values
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)

    Z = energy_vec(X, Y)

    plt.figure(figsize=(6, 5))
    contour = plt.contourf(X, Y, Z, levels=50, cmap='inferno')  # Filled contours
    plt.colorbar(contour, label="Z value")

    # Optional: add contour lines
    lines = plt.contour(X, Y, Z, levels=10, colors='black', linewidths=0.5)
    plt.clabel(lines, inline=True, fontsize=8)

    for i in range(config["num_chains"]):
        plt.scatter(resampled_samples[i, :, 0], resampled_samples[i, :, 1], color=color_vec[i], label=f"chain {i}", alpha=0.7)

    # Labels and title
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.xlim((-2, 2))
    plt.ylim((-2, 2))
    plt.legend()
    plt.title("2D Contour Plot with Colors")

    plt.show()

    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    dx, dy = x[1]-x[0], y[1]-y[0]
    X, Y = np.meshgrid(x, y)

    Z_energy = energy_vec(X, Y)
    P = np.exp(-Z_energy)
    P/=np.sum(P)*dx*dy
    true_marginal_likelihood = np.sum(P, axis = 0) * dy

    fig, ax = plt.subplots(1, 1, figsize=(5*config["num_chains"], 5))

    ax.hist(resampled_samples[0, :, 0], bins=1000, color="red", density=True, alpha=0.3, label=f"chain {0}")
    ax.legend()
    ax.plot(x, true_marginal_likelihood, color="black")
    plt.show()
    
    # print(resampled_samples)

    # # Energy trace plot
    # fig, ax = plt.subplots(figsize=(6, 4))
    # colors = plt.cm.viridis(np.linspace(0, 1, config["num_chains"]))

    # for i in range(config["num_chains"]):
    #     ax.plot(energy_values[i, :] + bias_values[i, :], label=f"chain {i}", color=colors[i]) # Måste kolla reweightingen!
    # plt.show()

    # # Energy histogram
    # fig, ax = plt.subplots(figsize=(6, 4))
    # colors = plt.cm.viridis(np.linspace(0, 1, config["num_chains"]))

    # for i in range(config["num_chains"]):
    #     plt.hist(
    #         energy_values[i, :]+ bias_values[i,:], bins=20, color=colors[i], edgecolor="black", alpha=0.5
    #     )
    # plt.show()

    # variable
    # fig, ax = plt.subplots(figsize=(6, 4))
    # colors = plt.cm.viridis(np.linspace(0, 1, config["num_chains"]))
    # for i in range(config["num_chains"]):
    # i=0
    # print(f"chain{i}")
    # print(sum(weights[i,:][valid_samples[i,:]>0])) # In some sense, the std should maybe be reset when it finds a new energy region?
    # print(sum(weights[i,:][valid_samples[i,:]<0]))
    # print(len(valid_samples[i,:][valid_samples[i,:]>0]))
    # print(len(valid_samples[i,:][valid_samples[i,:]<0]))
    # print(len(resampled_samples[i,:][resampled_samples[i,:]>0]))
    # print(len(resampled_samples[i,:][resampled_samples[i,:]<0]))
    #     ax.plot(
    #         resampled_samples[i, :],
    #         label=f"chain {i}",
    #         color=colors[i],
    #         marker="*",
    #         linestyle="none",
    #     )
    # plt.show()

    # for i in range(config["num_chains"]):
    #     fig, ax = plt.subplots(figsize=(6, 4))
    #     ax.plot(valid_samples[i,:], '*', color= colors[i])
    #     plt.show()

    # x = np.linspace(-4, 4, 100)
    # fig, ax = plt.subplots(figsize=(6, 4))
    # colors = plt.cm.viridis(np.linspace(0, 1, config["num_chains"]))
    # # print(valid_samples)
    # for i in range(config["num_chains"]):
    #     kde = gaussian_kde(valid_samples[i, :], weights=weights[i,:])
    #     y = kde(x)
    #     # ax.hist(resampled_samples[i, :], bins=20, color=colors[i], density=True, alpha=0.3)
    #     ax.plot(x, y, color=colors[i], label=f"chain {i}")
    #     ax.legend()
    #     ax.plot(x, 1 / np.exp(energy_function(x)), color="black")
    # plt.show()


    # x = np.linspace(-4, 4, 1000)
    # fig, ax = plt.subplots(figsize=(6, 4))
    # colors = plt.cm.viridis(np.linspace(0, 1, config["num_chains"]))
    # # print(valid_samples)
    # for i in range(config["num_chains"]):
    #     ax.hist(valid_samples[i, int(len(valid_samples)/3):], bins=100, color=colors[i], density=True, alpha=0.3, weights=weights[i,int(len(valid_samples)/3):], label=f"chain {i}")
    #     ax.legend()
    #     ax.plot(x, 1 / np.exp(energy_function(x)), color="black")
    # plt.show()

    # fig, ax = plt.subplots(figsize=(6, 4))
    # # colors = plt.cm.viridis(np.linspace(0, 1, config["num_chains"]))
    # full_bias_function = np.zeros((config["num_chains"], len(x)))
    # Z = 1 # Should be computed in the C function later.
    # for i in range(config["num_chains"]):
    #     chain_samples = valid_samples[i, :]
    #     diff_matrix = (1/np.sqrt(2*np.pi*bias_std[i,:]**2)) * np.exp(
    #         -((x[:, None] - chain_samples[None, :])**2) / (2 * bias_std[i,:]**2)
    #     )

    #     weighted_diff_matrix = diff_matrix * weights[i,:]
    #     probability_estimate = np.sum(weighted_diff_matrix, axis=1) / sum(weights[i,:])
    #     epsilon = np.exp(-config["beta"] * config["DeltaE"] / (1 - (1.0 / config["gamma"])))
    #     full_bias_function[i,:] = (1 - (1.0 / config["gamma"])) * np.log(probability_estimate / 1 + epsilon)
    #     # print(full_bias_function[i,:])
    #     # print(probability_estimate)
    #     ax.plot(
    #         x,
    #         energy_function(x) + full_bias_function[i,:],
    #         color=colors[i],
    #         linestyle="--",
    #     )
    #     ax.plot(x, energy_function(x), color="black")
    # plt.show()


    # fig, ax = plt.subplots(figsize=(6, 4))
    # # colors = plt.cm.viridis(np.linspace(0, 1, config["num_chains"]))
    # full_bias_function = np.zeros((config["num_chains"], len(x)))
    # Z = 1 # Should be computed in the C function later.
    # for i in range(config["num_chains"]):
    #     chain_samples = valid_samples[i, :]
    #     diff_matrix = (1/np.sqrt(2*np.pi*bias_std[i,:]**2)) * np.exp(
    #         -((x[:, None] - chain_samples[None, :])**2) / (2 * bias_std[i,:]**2)
    #     )

    #     weighted_diff_matrix = diff_matrix * weights[i,:]
    #     probability_estimate = np.sum(weighted_diff_matrix, axis=1) / sum(weights[i,:])
    #     ax.plot(
    #         x,
    #         -np.log(probability_estimate),
    #         color=colors[i],
    #         linestyle="--",
    #     )
    #     ax.plot(x, energy_function(x), color="black")
    # plt.show()

    # fig, ax = plt.subplots(figsize=(6, 4))
    # colors = plt.cm.viridis(np.linspace(0, 1, config["num_chains"]))
    # Z = 1 # Should be computed in the C function later.
    # for i in range(config["num_chains"]):
    #     chain_samples = valid_samples[i, :]
    #     diff_matrix = (1/np.sqrt(2*np.pi*bias_std[i,:]**2)) * np.exp(
    #         -((x[:, None] - chain_samples[None, :])**2) / (2 * bias_std[i,:]**2)
    #     )
    #     weighted_diff_matrix = diff_matrix * weights[i,:]
    #     # print(weighted_diff_matrix)
    #     probability_estimate = np.sum(weighted_diff_matrix, axis=1) / sum(weights[i,:])
    #     ax.plot(
    #         x,
    #         probability_estimate,
    #         color=colors[i],
    #         linestyle="--",
    #     )
    # ax.plot(x, 1/np.exp(energy_function(x)), color="black")
    # plt.show()


    # frames = []

    # fig, axs = plt.subplots(1, config["num_chains"], figsize=(10, 4))
    # cumulative_sums=[]

    # for k in range(config["num_chains"]):
    #     chain_samples = valid_samples[k,:,0]
    #     diff_matrix = (1/np.sqrt(2*np.pi*bias_std[i,:]**2)) * np.exp(
    #         -((x[:, None] - chain_samples[None, :])**2) / (2 * bias_std[i,:]**2)
    #     )
    #     weighted_diff_matrix = diff_matrix * weights[k,:]
    #     probability_estimates = np.cumsum(weighted_diff_matrix, axis=1) / np.cumsum(weights[k,:])
    #     epsilon = np.exp(-config["beta"] * config["DeltaE"] / (1 - (1.0 / config["gamma"])))
    #     cumulative_sums.append((1 - (1 / config["gamma"])) * np.log(probability_estimates / 1 + epsilon))

    # for k in range(config["num_chains"]):
    #     axs[k].clear()
    #     axs[k].plot(x, energy_vec(x, np.zeros_like(x)), color = "black")
    #     axs[k].plot(x, energy_vec(x, np.zeros_like(x)) + (1.0 - (1.0 / config["gamma"])) * np.log(epsilon), color = color_vec[k])

    # fig.canvas.draw()

    # mat = np.array(fig.canvas.renderer._renderer)

    # frames.append(mat)

    # for i in tqdm(range(0, len(valid_samples[0,:, 0]))):
    #     for k in range(config["num_chains"]):
    #         chain_samples = valid_samples[k,:, 0]
    #         axs[k].clear()
    #         axs[k].plot(x, energy_vec(x, np.zeros_like(x)), color = "black")
    #         cumulative_sum=cumulative_sums[k]
    #         y= energy_vec(x, np.zeros_like(x)) + cumulative_sum[:, i]
    #         axs[k].plot(x, y, color=color_vec[k])
    #         axs[k].set_ylim(-50, 50)
    #         axs[k].set_title(f"sample: {chain_samples[i]:.1f}")
    #     # Render plot to an image (as a numpy array)
    #     fig.canvas.draw()

    #     mat = np.array(fig.canvas.renderer._renderer)

    #     frames.append(mat)

    # plt.close(fig)
    # imageio.mimsave('plot_video_test.mp4', frames, fps=1)


if __name__ == "__main__":
    main()
