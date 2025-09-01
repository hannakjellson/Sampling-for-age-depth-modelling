import numpy as np
import matplotlib.pyplot as plt
from define_data_and_variables import get_data, get_hmc_config
import arviz as az

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


def get_index(N, cs, c14_depths, depth_index):
    low, high = 0, N - 1
    target = c14_depths[depth_index]
    
    while low <= high:
        mid = (low + high) // 2
        if cs[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    
    return low - 1 if low > 0 else 0


def main():
    valid_samples = np.load("../../../../output/samples.npy")
    energy_values = np.load("../../../../output/energy_values.npy")
    bias_values = np.load("../../../../output/bias_values.npy")

    data = get_data()
    config = get_hmc_config()
    colors = plt.cm.viridis(np.linspace(0, 1, config["num_chains"]))
    print(len(data["d18O"]))

    plt.figure()
    for i in range(config["num_chains"]):
        plt.plot(energy_values[i,100:10000] - bias_values[i,100:10000], color = colors[i])
        plt.plot(energy_values[i,100:10000], '*', color = colors[i])
    plt.show()

    return
    CV_values = data["theta"] - config["delta_c"] * np.sum(valid_samples[:, ::1000, :config["problem_index"]], axis = 2)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(config["num_chains"]):
        ax.hist(CV_values[i, :], bins = 100, color = colors[i], label = f"chain {i}", alpha = 0.5, density = True)
    plt.show()
    return
    # print(valid_samples[0, :100, :])
    # return

    # CV_values = data["theta"] - config["delta_c"] * np.sum(valid_samples[:, :, :config["problem_index"]], axis = 2)
    # fig, ax = plt.subplots(figsize=(6, 4))
    # for i in range(config["num_chains"]):
    #     ax.hist(CV_values[i, :], bins = 100, color = colors[i], label = f"chain {i}", alpha = 0.5, density = True)
    # plt.show()


    # # Sedimentation rate trace plot
    # fig, ax = plt.subplots(figsize=(6, 4))

    # for i in range(config["num_chains"]):
    #     ax.plot(valid_samples[i, :, 4], label=f"chain {i}", color=colors[i])
    # plt.show()

    # Resampling
    resampled_samples = []
    cutout = 100

    weights = np.exp(bias_values)

    for i in range(config["num_chains"]):
        weights_i = weights[i, cutout:] / sum(weights[i, cutout:])
        indices = np.random.choice(
            np.arange(cutout, len(weights_i) + cutout),
            size=int(len(weights_i)),
            replace=True,
            p=weights_i,
        )
        print(i)
        resampled_samples.append(valid_samples[i, indices, :])
             

    resampled_samples = np.array(resampled_samples)
    resampled_samples = resampled_samples[:,::10000,:]

    CV_values = data["theta"] - config["delta_c"] * np.sum(resampled_samples[:, :, :config["problem_index"]], axis = 2)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(config["num_chains"]):
        ax.hist(CV_values[i, :], bins = 100, color = colors[i], label = f"chain {i}", alpha = 0.5, density = True)
    plt.show()

    indices = []
    for i in range(len(data["d18O_depths"])):
        index = get_index(config["N"], config["cs"], data["d18O_depths"], i)
        indices.append(index)

    plt.figure()
    plt.plot(data["d18O_reference_times"], data["d18O_reference"])
    for i in range(config["num_chains"]):
        variables = resampled_samples[i,:,:]
        for j in range(len(variables)):
            d18O_ages = expected_ages(config["N"], config["delta_c"], config["cs"], data["theta"], len(data["d18O_depths"]), data["d18O_depths"], variables[j,:], indices)
            plt.plot(d18O_ages, data["d18O"], '*', color = colors[i])

    plt.plot(data["true_ages_D18O"], data["d18O"], '*', color = "k")
    plt.show()


    # # Print R_hat and N_eff using arviz
    # idata = az.convert_to_inference_data(resampled_samples)
    # summary = az.summary(idata, round_to=2)
    # print(summary)

    cumulative_sums = np.cumsum(resampled_samples[:, :,:], axis=2) * config["delta_c"]
    zeros = np.zeros((*resampled_samples[:,:,:].shape[:2], 1))
    age_offsets = np.concatenate((zeros, cumulative_sums), axis=2)

    model_ages = data["theta"] - age_offsets

    mean_model_age = model_ages.mean(axis=(0, 1))

    # Age depth trace plot
    fig, ax = plt.subplots(figsize=(6, 4))
    num_samples = len(resampled_samples[0,:,0])
    for i in range(config["num_chains"]):
        for j in range(num_samples):
            ax.plot(
                config["cs"],
                model_ages[i, j, :],
                color=colors[i],
                alpha=0.3,
                linewidth=0.1,
            )
    for i, c in enumerate(config["cs"]):
        if i != config["problem_index"]:
            ax.axvline(x=c, color='k', linestyle='--')
        else:
            ax.axvline(x=c, color='r', linestyle='--')

    ax.set_xlim(0, np.max(config["cs"]))
    ax.set_ylim(1200, 2000)
    plt.plot(config["cs"], mean_model_age, color="black", alpha=0.3, linewidth=1)
    plt.plot(data["c14_depths"], np.squeeze(data["c14_ages"]), "ko", markersize=4)
    plt.xlabel("Depth")
    plt.ylabel("Age")
    plt.tight_layout()
    plt.show()

    # # Energy trace plot
    # fig, ax = plt.subplots(figsize=(6, 4))
    # colors=plt.cm.viridis(np.linspace(0, 1, config["num_chains"]))

    # for i in range(config["num_chains"]):
    #     ax.plot(energy_values[i, :], label=f"chain {i}", color=colors[i])
    # plt.show()

    # # Energy histogram
    # fig, ax = plt.subplots(figsize=(6, 4))
    # colors=plt.cm.viridis(np.linspace(0, 1, config["num_chains"]))

    # for i in range(config["num_chains"]):
    #     plt.hist(energy_values[i, 1900 :], bins=20, color=colors[i], edgecolor='black', alpha=0.5)
    # plt.show()


if __name__ == "__main__":
    main()
