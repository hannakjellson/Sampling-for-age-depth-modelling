import numpy as np

from matplotlib import pyplot as plt

from define_variables import get_hmc_config


config, config_str = get_hmc_config()
temps = ((config["num_temps"])**(np.linspace(0, config["num_temps"] - 1, config["num_temps"]) / (config["num_temps"] - 1))) # 1 / BETAS
betas = 1/temps

bias_values = np.load(f"../../../../output/2d_example_unified_temp/bias_values_{config_str}.npy")
energy_values = np.load(f"../../../../output/2d_example_unified_temp/energy_values_{config_str}.npy")
weights = np.exp(-(betas - betas[0])[None, None, :] * energy_values[:, :, None] + bias_values[:, :, None])

n_eff_lambda = np.sum(weights, axis=1)**2 / np.sum(weights**2, axis=1)
n_eff_lambda /= weights.shape[1]
target = 1 / config["num_temps"]


fig, ax = plt.subplots(
    1, 1,
    figsize=(13, 6),
)
for it in range(4):
    ax.plot(
        betas,
        n_eff_lambda[it],
        label=f'Chain #{it+1}',
    )

ax.axhline(target, color='grey', label='Target', ls='--')


ax.set_xscale('linear')
ax.set_xlabel('beta')
ax.set_yscale('linear')
ax.set_ylabel(r'$n_\text{eff} / n$')

ax.legend(frameon=False)

fig.tight_layout()

fig.savefig(
    f"../../../../pics/2d_example_unified_temp/neff_diag.jpg",
    bbox_inches='tight',
    pad_inches=0.05,
    dpi=300,
)

plt.show()
