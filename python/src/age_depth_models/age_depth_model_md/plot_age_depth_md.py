import numpy as np
import matplotlib.pyplot as plt
from define_data_and_variables_md import get_data, get_hmc_config
import arviz as az
import autograd.numpy as anp


def mean_ages(theta, age_depths, sedimentation_rates, config):
    indices = anp.searchsorted(config["cs"], age_depths, side='right') - 1 
    cumsum = anp.hstack((0, anp.cumsum(sedimentation_rates)))
    return theta-cumsum[indices] * config["delta_c"] - sedimentation_rates[indices]*(age_depths - config["cs"][indices])

def main():
    valid_samples = np.load("../../../output/samples_md.npy")
    energy_values = np.load("../../../output/energy_values_md.npy")
    bias_values = np.load("../../../output/bias_values_md.npy")
    data = get_data()
    config = get_hmc_config()
    num_resampled_samples=int(config["num_MD"] / 4)

    print(bias_values)

    weights = np.exp(bias_values)
    print(weights)
    print(np.shape(weights))
    print(np.shape(valid_samples[0,:]))

    resampled_samples = []
    for i in range(config["num_chains"]):
        indices = np.random.choice(
                np.shape(valid_samples[i,:,:])[0],
                size=num_resampled_samples,
                replace=True,
                p=weights[i, :] / sum(weights[i, :]),
            )
        resampled_samples.append(valid_samples[i,indices,:])

    resampled_samples = np.array(resampled_samples)

    # Print R_hat and N_eff using arviz
    idata = az.convert_to_inference_data(resampled_samples)
    summary = az.summary(idata, round_to=2)
    print(summary)

    cumulative_sums = np.cumsum(resampled_samples, axis=2) * config["delta_c"]
    zeros = np.zeros((*resampled_samples.shape[:2], 1))
    age_offsets = np.concatenate((zeros, cumulative_sums), axis=2)

    model_ages = data["theta"] - age_offsets

    mean_model_age = model_ages.mean(axis=(0, 1))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(data["d18O_reference_times"], data["d18O_reference"], color="black")
    colors = plt.cm.viridis(np.linspace(0, 1, config["num_chains"]))
    for i in range(len(data["c14_ages"])):
        ax.axvline(data["c14_ages"][i], color="black", linestyle= "--")

    for i in range(len(resampled_samples[:,0])):
        ax.plot(mean_ages(data["theta"], data["d18O_depths"], resampled_samples[i,0], config), data["d18O"], color=colors[i], marker="*")
    plt.show() # With another choice of parameters, I got a better way through these points that looked objectively better. I guess a next step would be to in some way combine the chains as i try in the example?
    # But is this then going back to parallel tempering? Not really, i dont throw away samples due to it, but need to make it work.



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

    for i in range(config["num_chains"]):
        ax.plot(resampled_samples[i, :, 4], label=f"chain {i}", color=colors[i])
    plt.show()

    # Age depth trace plot
    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(config["num_chains"]):
        for j in range(num_resampled_samples):
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
