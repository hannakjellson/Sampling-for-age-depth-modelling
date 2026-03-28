from functools import partial

import jax
import jax.numpy as jnp
import blackjax
import pytensor.tensor as pt

from typing import NamedTuple
from blackjax.mcmc.hmc import HMCState

from define_data_and_variables import opes_config, data


import sys


# Use this for toying with other integrators
# from blackjax.mcmc.integrators import yoshida

class BiasState(NamedTuple):
    state: HMCState
    delta_F_nominator_sum: jax.Array
    delta_F_denominator_sum: jax.Array
    delta_F: jax.Array
    bias_value: float
    count: float


def bias_potential(energy, delta_F):

    temp_term = (opes_config["bs"] - 1) * energy
    # jax.debug.print("energy = {}", energy)
    sum_for_V = jnp.sum(
            jnp.exp(-temp_term + delta_F)
        )
    return -jnp.log(sum_for_V / opes_config["nt"])


def update_delta_F(
    energy,
    delta_F_nominator_sum,
    delta_F_denominator_sum,
    delta_F,
    potential,
):
    # temp_term = (BETAS - BETA_0) * energy
    # delta_F_nominator_sum += jnp.exp(
    #     -temp_term + potential * jnp.ones(NUM_TEMPS)
    # )

    # delta_F_denominator_sum += jnp.exp(potential) * jnp.ones(NUM_TEMPS)

    # delta_F = -jnp.log(
    #     delta_F_nominator_sum / delta_F_denominator_sum
    # )

    # delta_F = jnp.clip(
    #     delta_F,
    #     min=None,
    #     max=DE,
    # )

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

    # jax.debug.print("init_position = {}", init_position)

    # Set up initial state, init_position is passed from outside
    # Default should be uniform
    grad_fn = jax.value_and_grad(logp_fn)
    logdensity, logdensity_grad = grad_fn(init_position)

    init_state = HMCState(
        init_position,
        logdensity,
        logdensity_grad,
    )

    # Calculate initial free energy
    (
        delta_F_nominator_sum,
        delta_F_denominator_sum,
        delta_F,
    ) = update_delta_F(
        energy=-logdensity,
        delta_F_nominator_sum=jnp.zeros(opes_config["nt"]),
        delta_F_denominator_sum=jnp.zeros(opes_config["nt"]),
        delta_F=jnp.array([   0.          -10.95075856  -21.25356605  -30.95308214  -40.09170866
                            -48.7098      -56.8455435   -64.53469873  -71.81039861  -78.70311012
                            -85.24073398  -91.44877772  -97.35055757 -102.9674177  -108.31896473
                            -113.42330292 -118.2972575  -122.95659396 -127.41623545 -131.69042265
                            -135.7927143  -139.73577571 -143.53103154 -147.18835093 -150.71589331
                            -154.12011748 -157.40587858 -160.57656501 -163.63428944 -166.58016274
                            -169.41464125 -172.13789511 -174.75013408 -177.25184628 -179.64393616
                            -181.92777425 -184.10518429 -186.17839421 -188.14997078 -190.02275013]),
        potential=0,
    )

    # Calculate initial bias value for storing
    bias_value = bias_potential(energy = -logdensity, delta_F=delta_F)

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
            # jax.debug.print("logdens = {}", logp)
            # jax.debug.print("bias = {}", bias)
            # jax.debug.print("pos = {}", pos)
            # jax.debug.breakpoint()
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
            step_size=opes_config["dt"],
            inverse_mass_matrix=jnp.ones(opes_config["N"]),
            num_integration_steps=opes_config["ndt"],
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
        # jax.debug.print("potential logdens= {}", logdensity)
        # jax.debug.print("potential = {}", potential)
        # jax.debug.print("potential position = {}", position)

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
