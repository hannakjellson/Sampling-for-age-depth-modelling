import numpy as np
import matplotlib.pyplot as plt
from define_data_and_variables import get_data, get_hmc_config
import arviz as az


def main():
    valid_samples = np.load("../../../output/samples.npy")
    # energy_values = np.load("../../../output/energy_values.npy")
    data = get_data()
    config = get_hmc_config()

    # Print R_hat and N_eff using arviz
    idata = az.convert_to_inference_data(valid_samples)
    summary = az.summary(idata, round_to=2)
    print(summary)

    cumulative_sums = np.cumsum(valid_samples, axis=2) * config["delta_c"]
    zeros = np.zeros((*valid_samples.shape[:2], 1))
    age_offsets = np.concatenate((zeros, cumulative_sums), axis=2)

    model_ages = data["theta"] - age_offsets

    mean_model_age = model_ages.mean(axis=(0, 1))

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

    # Sedimentation rate trace plot
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, config["num_chains"]))

    for i in range(config["num_chains"]):
        ax.plot(valid_samples[i, :, 4], label=f"chain {i}", color=colors[i])
    plt.show()

    # Age depth trace plot
    fig, ax = plt.subplots(figsize=(6, 4))
    num_samples = int(config["num_MH"] / 2)
    for i in range(config["num_chains"]):
        for j in range(num_samples):
            ax.plot(
                config["cs"],
                model_ages[i, j, :],
                color=colors[i],
                alpha=0.3,
                linewidth=0.1,
            )

    ax.set_xlim(0, np.max(config["cs"]))
    ax.set_ylim(1200, 2000)
    plt.plot(config["cs"], mean_model_age, color="black", alpha=0.3, linewidth=1)
    plt.plot(data["c14_depths"], np.squeeze(data["c14_ages"]), "ko", markersize=4)
    plt.xlabel("Depth")
    plt.ylabel("Age")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
