import pymc as pm
# print(pm.__version__)
import numpy as np
import jax.numpy as jnp
import pytensor.tensor as pt
import pytensor
pytensor.config.blas__ldflags = "-llapack -lblas -lcblas"

# TODO: NOT WORKING BUT SHOULD BE POSSIBLE TO GET IT TO WORK. NOT OBVIOUS HOW TO CHANGE THINGS IN A GOOD WAY THOUGH.
# Maybe i should try this in my c-framework...
from define_data_and_variables import get_data, get_smc_config
from pytensor import shared

from pymc.sampling import jax as pmj
import arviz as az
from scipy.special import logsumexp
from pymc.vartypes import discrete_types
from pytensor.graph.replace import clone_replace
from pytensor.link.jax import JAXLinker
from pytensor.tensor.random.type import RandomGeneratorType
from pymc.pytensorf import (
    compile,
    floatX,
    join_nonshared_inputs,
    make_shared_replacements,
)
from pymc.model import modelcontext
from pymc.initial_point import make_initial_point_expression
from pymc.logprob import logp




# from inference_loop import inference_loop
# pmj._blackjax_inference_loop = inference_loop


def mean_ages(sedimentation_rates, data, data_type="c14"):
    if data_type == "c14":
        depths = data["c14d"]
    else:
        depths = data["d18od"]
    indices = pt.searchsorted(data["cs"], depths, side="right") - 1
    cumsum = pt.concatenate([[0], pt.cumsum(sedimentation_rates)])
    CS_pt = pt.constant(data["cs"])
    return (
        data["th"]
        - cumsum[indices] * data["dc"]
        - sedimentation_rates[indices] * (depths - CS_pt[indices])
    )


def interpolate(sed_rates, data):
    D18O_times = mean_ages(sed_rates, data, data_type="D18O")

    # Get indices of the left side of interpolation intervals
    indices = (
        pt.searchsorted(data["d18ort"], D18O_times, side="right") - 1
    )
    indices = pt.clip(indices, 0, len(data["d18ort"]) - 2)

    t0 = pt.constant(data["d18ort"])[indices]
    t1 = pt.constant(data["d18ort"])[indices + 1]
    v0 = pt.constant(data["d18or"])[indices]
    v1 = pt.constant(data["d18or"])[indices + 1]

    slope = (v1 - v0) / (t1 - t0)
    D18O_interp = v0 + slope * (D18O_times - t0)

    return D18O_interp


def systematic_resampling(weights, rng):
    """
    Systematic resampling.

    Parameters
    ----------
    weights :
        The weights should be probabilities and the total sum should be 1.

    Returns
    -------
    new_indices: array
        A vector of indices in the interval 0, ..., len(normalized_weights)
    """
    lnw = len(weights)
    arange = np.arange(lnw)
    uniform = (rng.random(1) + arange) / lnw

    idx = 0
    weight_accu = weights[0]
    new_indices = np.empty(lnw, dtype=int)
    for i in arange:
        while uniform[i] > weight_accu:
            idx += 1
            weight_accu += weights[idx]
        new_indices[i] = idx

    return new_indices


def _logp_forw(point, out_vars, in_vars, shared, compile_kwargs=None):
    """Compile PyTensor function of the model and the input and output variables.

    Parameters
    ----------
    out_vars : list
        Containing Distribution for the output variables
    in_vars : list
        Containing Distribution for the input variables
    shared : list
        Containing TensorVariable for depended shared data
    compile_kwargs: dict, optional
        Additional keyword arguments passed to pytensor.function
    """
    if compile_kwargs is None:
        compile_kwargs = {}

    # Replace integer inputs with rounded float inputs
    if any(var.dtype in discrete_types for var in in_vars):
        replace_int_input = {}
        new_in_vars = []
        for in_var in in_vars:
            if in_var.dtype in discrete_types:
                float_var = pt.TensorType("floatX", in_var.type.shape)(in_var.name)
                new_in_vars.append(float_var)
                replace_int_input[in_var] = pt.round(float_var).astype(in_var.dtype)
            else:
                new_in_vars.append(in_var)

        out_vars = clone_replace(out_vars, replace_int_input, rebuild_strict=False)
        in_vars = new_in_vars

    out_list, inarray0 = join_nonshared_inputs(
        point=point, outputs=out_vars, inputs=in_vars, shared_inputs=shared
    )
    f = compile([inarray0], out_list[0], **compile_kwargs)
    f.trust_input = True
    return f

class Pearson:
    def __init__(self, a):
        self.l = a.shape[0]
        self.am = a - np.sum(a, axis=0) / self.l
        self.aa = np.sum(self.am**2, axis=0) ** 0.5

    def get(self, b):
        bm = b - np.sum(b, axis=0) / self.l
        bb = np.sum(bm**2, axis=0) ** 0.5
        ab = np.sum(self.am * bm, axis=0)
        return np.abs(ab / (self.aa * bb))
    
def main():
    smc_config = get_smc_config()
    data = get_data()

    with pm.Model() as model:
        # Priors
        sedimentation_rates = pm.LogNormal(
            "sed_rates", mu=data["pm"], sigma=data["ps"], shape=data["N"], default_transform=None,
        )

        # Observations likelihood
        expected_ages = mean_ages(sedimentation_rates, data)
        c14_obs = pm.Normal(
            "c14_obs",
            mu=expected_ages,
            sigma=data["c14s"],
            observed=data["c14"],
        )

        expected_D18O = interpolate(sedimentation_rates, data)
        D18O_obs = pm.Normal(
            "d18O_obs",
            mu=expected_D18O,
            sigma=data["d18os"],
            observed=data["d18o"],
        )

    # expected_ages is symbolic
    logp_c14 = pt.sum(logp(pm.Normal.dist(mu=expected_ages, sigma=data["c14s"]), data["c14"]))
    logp_d18o = pt.sum(logp(pm.Normal.dist(mu=expected_D18O, sigma=data["d18os"]), data["d18o"]))

    # print(model.value_vars)
    # print(model.rvs_to_transforms)

    class PartialTemperedKernel(pm.smc.IMH):

        def __init__(self, *args, **kwargs):
            compile_kwargs = kwargs["compile_kwargs"] if kwargs["compile_kwargs"] is not None else {}
            compile_kwargs.update({"on_unused_input": "ignore"})

            super().__init__(*args, **kwargs)

            self.c14_logp: np.ndarray | None = None
            self.d18o_logp: np.ndarray | None = None

            initial_point = model.initial_point(random_seed=self.rng.integers(2**30))
            shared = make_shared_replacements(initial_point, self.variables, model)

            self.c14_logp_func = _logp_forw(
                initial_point, [logp_c14], self.variables, shared, compile_kwargs
            )

            self.d18o_logp_func = _logp_forw(
                initial_point, [logp_d18o], self.variables, shared, compile_kwargs
            )

        def set_rng(self, rng: np.random.Generator):
                """
                Copy compiled functions, updating their random number generators.

                This is necessary because these functions were compiled once at initialization, then pickled
                and sent to worker processes. Each worker needs its own RNG state to ensure independent sampling,
                so we replace the shared RNGs in the compiled functions with new ones created from the provided `rng`.

                This method copies the functions, so it is expensive and should only be called once per worker!
                """

                def make_rng_swaps(fn, rng):
                    shared_rngs = [
                        var for var in fn.get_shared() if isinstance(var.type, RandomGeneratorType)
                    ]
                    n_shared_rngs = len(shared_rngs)
                    if n_shared_rngs > 0 and isinstance(fn.maker.linker, JAXLinker):
                        raise NotImplementedError(
                            f"JAX rngs cannot be replaced after compilation. {self}.set_rng will fail to "
                            f"properly update random seeds between chains, resulting in non-independent "
                            f"sampling."
                        )

                    return {
                        old_shared_rng: shared(new_rng, borrow=True)
                        for old_shared_rng, new_rng in zip(
                            shared_rngs, rng.spawn(n_shared_rngs), strict=True
                        )
                    }

                self.rng = rng
                self.prior_logp_func = self.prior_logp_func.copy(
                    swap=make_rng_swaps(self.prior_logp_func, self.rng)
                )
                self.c14_logp_func = self.c14_logp_func.copy(
                    swap=make_rng_swaps(self.c14_logp_func, self.rng)
                )
                self.d18o_logp_func = self.d18o_logp_func.copy(
                    swap=make_rng_swaps(self.d18o_logp_func, self.rng)
                )
                # self.likelihood_logp_func = self.likelihood_logp_func.copy(
                #     swap=make_rng_swaps(self.likelihood_logp_func, self.rng)
                # )
                
        def setup_kernel(self):
            self.c14_logp = np.array([self.c14_logp_func(sample).item() for sample in self.tempered_posterior])
            self.d18o_logp = np.array([self.d18o_logp_func(sample).item() for sample in self.tempered_posterior])

        def update_beta_and_weights(self):
            """Calculate the next inverse temperature (beta).

            The importance weights based on two successive tempered likelihoods (i.e.
            two successive values of beta) and updates the marginal likelihood estimate.

            ESS is calculated for importance sampling. BDA 3rd ed. eq 10.4
            """
            # print(f"extra: {extra_term}")
            self.iteration += 1

            low_beta = old_beta = self.beta
            up_beta = 2.0

            rN = int(len(self.c14_logp) * self.threshold)

            while up_beta - low_beta > 1e-6:
                new_beta = (low_beta + up_beta) / 2.0
                log_weights_un = (new_beta - old_beta) * self.d18o_logp
                log_weights = log_weights_un - logsumexp(log_weights_un)
                ESS = int(np.exp(-logsumexp(log_weights * 2)))
                if ESS == rN:
                    break
                elif ESS < rN:
                    up_beta = new_beta
                else:
                    low_beta = new_beta
            if new_beta >= 1:
                new_beta = 1
                log_weights_un = (new_beta - old_beta) * self.d18o_logp
                log_weights = log_weights_un - logsumexp(log_weights_un)

            self.beta = new_beta
            self.weights = np.exp(log_weights)
            # We normalize again to correct for small numerical errors that might build up
            self.weights /= self.weights.sum()
            self.log_marginal_likelihood += logsumexp(log_weights_un) - np.log(self.draws)

        def resample(self):
            """Resample particles based on importance weights."""
            self.resampling_indexes = systematic_resampling(self.weights, self.rng)

            self.tempered_posterior = self.tempered_posterior[self.resampling_indexes]
            self.prior_logp = self.prior_logp[self.resampling_indexes]
            # self.likelihood_logp = self.likelihood_logp[self.resampling_indexes]
            self.c14_logp = self.c14_logp[self.resampling_indexes]
            self.d18o_logp = self.d18o_logp[self.resampling_indexes]


            self.tempered_posterior_logp = self.prior_logp + self.c14_logp + self.d18o_logp * self.beta

        def _reset_state(self):
            # Call the original method to reset draws, likelihoods, beta, etc.
            super()._reset_state()
            self.c14_logp = None
            self.d18o_logp = None

        def mutate(self):
            """Independent Metropolis-Hastings perturbation."""
            self.n_steps = 1
            old_corr = 2
            corr = Pearson(self.tempered_posterior)
            ac_ = []
            while True:
                log_R = np.log(self.rng.random(self.draws))
                # The proposal is independent from the current point.
                # We have to take that into account to compute the Metropolis-Hastings acceptance
                # We first compute the logp of proposing a transition to the current points.
                # This variable is updated at the end of the loop with the entries from the accepted
                # transitions, which is equivalent to recomputing it in every iteration of the loop.
                proposal = floatX(self.proposal_dist.rvs(size=self.draws, random_state=self.rng))
                proposal = proposal.reshape(len(proposal), -1)
                # To do that we compute the logp of moving to a new point
                forward_logp = self.proposal_dist.logpdf(proposal)
                # And to going back from that new point
                backward_logp = self.proposal_dist.logpdf(self.tempered_posterior)
                # ll = np.array([self.likelihood_logp_func(prop) for prop in proposal])
                c14l = np.array([self.c14_logp_func(prop).item() for prop in proposal])
                d18ol = np.array([self.d18o_logp_func(prop).item() for prop in proposal])
                pl = np.array([self.prior_logp_func(prop) for prop in proposal])
                proposal_logp = pl + c14l + d18ol * self.beta
                accepted = log_R < (
                    (proposal_logp + backward_logp) - (self.tempered_posterior_logp + forward_logp)
                )
                # print(f"acceptance: {np.sum(accepted)/len(accepted)}")

                self.tempered_posterior[accepted] = proposal[accepted]
                self.tempered_posterior_logp[accepted] = proposal_logp[accepted]
                self.prior_logp[accepted] = pl[accepted]
                # self.likelihood_logp[accepted] = ll[accepted]
                self.c14_logp[accepted] = c14l[accepted]
                self.d18o_logp[accepted] = d18ol[accepted]
                # print(accepted)
                ac_.append(accepted)
                self.n_steps += 1

                pearson_r = corr.get(self.tempered_posterior)
                if np.mean((old_corr - pearson_r) > self.correlation_threshold) > 0.9:
                    old_corr = pearson_r
                else:
                    break

            self.acc_rate = np.mean(ac_)

    chains = 5
    starting_points = np.load(r"C:\Users\hanna\Desktop\PhD\Bacon\python\src\age_depth_models\age_depth_model_c14_normal\output\68831df8d5\samples.npy")
    initvals = [{"sed_rates" : [np.array(init_val) for init_val in np.repeat(starting_points[:, 100:, :].reshape(-1, starting_points[:, 100:, :].shape[-1])[i * smc_config["nch"] : (i + 1) * smc_config["nch"]], 10, axis = 0)]} for i in range(chains)]
    with model: 
        trace = pm.smc.sample_smc(
            start = initvals,
            chains = chains,
            draws=smc_config["nch"],
            progressbar=True,
            random_seed = 32,
            kernel = PartialTemperedKernel,
        )

    trace.posterior.to_netcdf(f"samples.nc")
    # print(trace)

    print(az.summary(trace))
    # az.plot_trace(trace)
    # az.plot_posterior(trace)


if __name__ == "__main__":
    main()
