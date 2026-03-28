from functools import partial

import jax
import jax.numpy as jnp
import blackjax

from typing import NamedTuple
from blackjax.mcmc.hmc import HMCState

from define_data_and_variables import opes_config, data
from multiprocessing import shared_memory, Barrier

class BiasState(NamedTuple):
    state: HMCState
    delta_F_nominator_sum: jax.Array
    delta_F_denominator_sum: jax.Array
    delta_F: jax.Array
    bias_value: float
    count: float


def to_shared_memory(np_array):
    shm = shared_memory.SharedMemory(create=True, size=np_array.nbytes)
    shm_array = jnp.ndarray(np_array.shape, dtype=np_array.dtype, buffer=shm.buf)
    jnp.copyto(shm_array, np_array)  # copy existing data
    return shm, shm_array

def bias_potential(energy, delta_F):

    temp_term = -(opes_config["bs"] - opes_config["bs"][0]) * energy
    # jax.debug.print("energy = {}", energy)
    max_temp = jnp.max(temp_term)
    sum_for_V = jnp.sum(
            jnp.exp(temp_term - max_temp + delta_F)
        )
    return -max_temp - jnp.log(sum_for_V / opes_config["nt"])


def update_delta_F(
    energy,
    delta_F_nominator_sum,
    delta_F_denominator_sum,
    delta_F,
    potential,
):
    temp_term = (opes_config["bs"] - opes_config["bs"][0]) * energy
    delta_F_nominator_sum += jnp.exp(
        -temp_term + potential * jnp.ones(opes_config["nt"])
    )

    delta_F_denominator_sum += jnp.exp(potential) * jnp.ones(opes_config["nt"])

    delta_F = -jnp.log(
        delta_F_nominator_sum / delta_F_denominator_sum
    )

    return delta_F_nominator_sum, delta_F_denominator_sum, delta_F


# This is the replacement inference_loop
# A reference implementation can be found here:
# https://github.com/pymc-devs/pymc/blob/340e403b8813ab5f3699a476cc828cc92c4f9d50/pymc/sampling/jax.py#L250
def inference_loop(
    seed, init_position, logp_fn, draws, tune, target_accept,
    **adaptation_kwargs
):
    # Ignore passed algorithm kwarg and always use hmc (for now)
    algorithm_name = adaptation_kwargs.pop("algorithm", "nuts")
    algorithm = blackjax.hmc

    # Set up initial state, init_position is passed from outside
    # Default should be uniform
    delta_F_denominator_sum = opes_config["dfd"] * jnp.ones(opes_config["nt"])
    delta_F = opes_config["dfa"] * (opes_config["bs"] - 1)
    delta_F_nominator_sum = jnp.exp(-delta_F)*delta_F_denominator_sum

    bias_function = partial(
        bias_potential,
        delta_F=delta_F
    )

    def logp_biased(pos):
        # bias is subtracked instead of added, due to the different sign
        # of logp_fn in comparison to molecular dynamics (probability
        # distribution vs. potential energy)
        logp = logp_fn(pos)
        bias = bias_function(-logp)

        return logp - bias

    grad_fn = jax.value_and_grad(logp_biased)
    logdensity, logdensity_grad = grad_fn(init_position)

    init_state = HMCState(
        init_position,
        logdensity,
        logdensity_grad,
    )

    # Calculate initial bias value for storing
    bias_value = bias_potential(energy = -logdensity, delta_F=delta_F)

    shm_dfd, arr_dfd = to_shared_memory(delta_F_denominator_sum)
    shm_dfn, arr_dfn = to_shared_memory(delta_F_nominator_sum)
    shm_df, arr_df = to_shared_memory(delta_F)

    barrier = Barrier(opes_config["nch"])

    # Set up initial BiasState
    init_bias_state = BiasState(
        init_state,
        delta_F_nominator_sum,
        delta_F_denominator_sum,
        delta_F,
        bias_value,
        0.0,
    )

    # Pure function to perform one step of the algorithm, takes a bias state
    # and rng_keys and returns a bias_state, position and info
    def _one_step(bias_state, xs):
        _, rng_key = xs
        # unpack, done for convenience only
        state = bias_state.state
        delta_F_nominator_sum = bias_state.delta_F_nominator_sum
        delta_F_denominator_sum = bias_state.delta_F_denominator_sum
        delta_F = bias_state.delta_F

        # partial function evaluation, so that bias arguments are always the
        # same
        bias_function = partial(
            bias_potential,
            delta_F=delta_F
        )

        def logp_biased(pos):
            # bias is subtracked instead of added, due to the different sign
            # of logp_fn in comparison to molecular dynamics (probability
            # distribution vs. potential energy)
            logp = logp_fn(pos)
            bias = bias_function(-logp)

            return logp - bias

        grad_fn = jax.value_and_grad(logp_biased)
        logdensity, logdensity_grad = grad_fn(state.position) # Note that these are based on the biased log_p, so state.logdensity is energy - bias and same for state.logdensity_grad
        # jax.debug.print("logp = {}", logdensity)
        # jax.debug.breakpoint()
        state = HMCState(
            position=state.position,
            logdensity=logdensity,
            logdensity_grad=logdensity_grad
        )
        biased_kernel = algorithm(
            logp_biased,
            step_size=opes_config["hmcc"]["dt"],
            inverse_mass_matrix=jnp.ones(data["N"]),
            num_integration_steps=opes_config["hmcc"]["ndt"],
            # Uncomment for toying with integrators
            # integrator=yoshida,
        ).step
        
        for _ in range(opes_config["nhmc"]):
            # XXX update rng_key?
            rng_key, subkey = jax.random.split(rng_key)
            state, info = biased_kernel(subkey, state)
            # jax.debug.print("state= {}\n", state)

        # Prepare info and outputs
        position = state.position
        logdensity = logp_fn(position)
        potential = bias_potential(-logdensity, delta_F)

        stats = {
            "diverging": info.is_divergent,
            "energy": info.energy,
            # "tree_depth": info.num_trajectory_expansions,
            # "n_steps": info.num_integration_steps,
            "acceptance_rate": info.acceptance_rate,
            "lp": state.logdensity,
            "bias_value": potential,
            "delta_F": bias_state.delta_F,
        }

        # Update bias
        new_delta_F_nominator_sum, new_delta_F_denominator_sum, new_delta_F = \
            update_delta_F(
                -logdensity,
                delta_F_nominator_sum,
                delta_F_denominator_sum,
                delta_F,
                potential,
            )

        new_bias_state = BiasState(
            state,
            new_delta_F_nominator_sum,
            new_delta_F_denominator_sum,
            new_delta_F,
            potential,
            bias_state.count + 1,
        )

        return new_bias_state, (position, stats)

    # This is kept mostly from the reference implementation
    progress_bar = adaptation_kwargs.pop("progress_bar", False)

    keys = jax.random.split(seed, draws)
    scan_fn = blackjax.progress_bar.gen_scan_fn(draws, progress_bar)
    _, (samples, stats) = scan_fn(
        _one_step,
        init_bias_state,        # This is changed from the reference
        (jnp.arange(draws), keys),
    )

    return samples, stats
