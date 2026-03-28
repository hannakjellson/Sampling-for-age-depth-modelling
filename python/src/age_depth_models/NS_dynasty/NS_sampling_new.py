import os
import dynesty
import numpy as np
import scipy
from define_data_and_variables import get_data, get_NS_config, hash_configs

config = get_NS_config()
data = get_data()

c14_depths = np.array(data["c14d"])
c14_ages = np.array(data["c14"])
c14_sigma = np.array(data["c14s"])
d18O_depths = np.array(data["d18od"])
d18O_vals = np.array(data["d18o"])
d18O_sigma = np.array(data["d18os"])
cs = np.array(data["cs"])

ref_times = np.array(data["d18ort"])
ref_vals = np.array(data["d18or"])

delta_c = data["dc"]
theta = data["th"]

flattened = np.load(r"../age_depth_model_c14_normal/output/68831df8d5/samples.npy")[0, 1000:, :]

sigma_x = np.cov(flattened, rowvar=False)
mu_x = np.mean(flattened, axis = 0)
L_w = np.linalg.cholesky(sigma_x + np.eye(data["N"]) * 1e-6)

def original_log_likelihood(sed_rates):
    cumsum = np.hstack((np.zeros(1), np.cumsum(sed_rates)))
    
    # C14 calculation
    indices_c14 = np.searchsorted(cs, c14_depths, side="right") - 1
    expected_ages = (theta - cumsum[indices_c14] * delta_c 
                        - sed_rates[indices_c14] * (c14_depths - cs[indices_c14]))
    
    l1 = np.sum(scipy.stats.norm.logpdf(c14_ages, loc=expected_ages, scale=c14_sigma))

    # D18O calculation
    indices_d18 = np.searchsorted(cs, d18O_depths, side="right") - 1
    d18_times = (theta - cumsum[indices_d18] * delta_c 
                    - sed_rates[indices_d18] * (d18O_depths - cs[indices_d18]))
    
    expected_D18O = np.interp(d18_times, ref_times, ref_vals)
    
    l2 = np.sum(scipy.stats.norm.logpdf(d18O_vals, loc=expected_D18O, scale=d18O_sigma))

    return l1 + l2
    
def loglike(sed_rates):
    if np.any(sed_rates <= 0):
        return -np.inf
    
    logL = original_log_likelihood(sed_rates)
    
    new_prior_dist = scipy.stats.multivariate_normal.logpdf(sed_rates, mean = mu_x, cov = np.cov(flattened, rowvar=False))

    orig_prior_dist = np.sum(scipy.stats.lognorm.logpdf(
        sed_rates, data["ps"], scale = np.exp(data["pm"])
    ))
    return logL + orig_prior_dist - new_prior_dist

    
def prior_transform(u):
    t = scipy.stats.norm.ppf(u)

    sed_rates = mu_x + np.dot(L_w, t)

    return sed_rates

def main():
    # n_cores = multiprocessing.cpu_count() - 1
    # with multiprocessing.Pool(processes=n_cores) as pool:
    sampler = dynesty.DynamicNestedSampler(
        loglike, 
        prior_transform, 
        data["N"],
        # pool=pool,
        # queue_size=n_cores,
    )
    sampler.run_nested(nlive_init=2000)
    
    results = sampler.results
    print(results.summary())
    new_samples = results.samples_equal()
    np.save("samples.npy", results.samples)
    np.save("true_samples.npy", new_samples)
    
    # hash = hash_configs(config, data)
    # sd_output_dir = f"output/sd_{config["sd"]}"

    # os.makedirs(f"{sd_output_dir}/{hash}", exist_ok=True)
    # np.save(results, f"{sd_output_dir}/{hash}/results.npy")
    print("Sampling complete. Results saved.")

if __name__ == "__main__":
    main()