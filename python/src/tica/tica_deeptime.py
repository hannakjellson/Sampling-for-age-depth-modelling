import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from deeptime.decomposition import TICA
from define_data_and_variables import get_data, get_hmc_config

def main():
    config, config_str = get_hmc_config()
    data = np.load(f"../../../output/tica/samples_{config_str}.npy")
    data -= np.mean(data, axis=(0,1))
    taus = [1,2, 3,5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    singular_values_vec = np.empty((config["nch"], len(taus), config["N"]))
    eig_vecs_vec = np.empty((config["nch"], len(taus), config["N"], config["N"]))
    for k in range(config["nch"]):
        chain_data = data[k, :, :]
        print(np.shape(data))
        for i, tau in enumerate(taus):
            tica = TICA(lagtime=tau)
            tica.fit(data)
            model = tica.fetch_model()
            singular_values = model.singular_values
            timelagged_coeffs = model.timelagged_coefficients
            print(np.shape(timelagged_coeffs))
            singular_values_vec[k, i, :] = singular_values
            eig_vecs_vec[k, i, :, :] = timelagged_coeffs

    np.save(f"../../../output/tica/eigs_{config_str}.npy", singular_values_vec)
    np.save(f"../../../output/tica/eig_vecs_{config_str}.npy", eig_vecs_vec)
    np.save(f"../../../output/tica/taus_{config_str}.npy", np.array(taus))

    


if __name__ == '__main__':
    main()