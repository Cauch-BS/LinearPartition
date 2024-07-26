import jax 
import typing
import jax.numpy as jnp
from jax.typing import Array, ArrayLike
from functools import partial
from . import utils

BEAM_SIZE = 100
NUM_NUC_1 = 5
NUM_NUC_2 = 25
NUM_NUC_3 = 125

######################
### Initialization ###
######################

@jax.jit
def __init__(sequence):
    sequence_length = len(sequence)
    max_pairs = sequence_length * (sequence_length - 1) // 2
    nucleotides = jnp.array(
        [utils.GET_ACGU_NUM(nuc) for nuc in sequence], dtype=jnp.int32
    )
    pair_shape = (max_pairs, 3)
    beamH = jnp.zeros(pair_shape, dtype=jnp.float64)
    beamP = jnp.zeros(pair_shape, dtype=jnp.float64)
    beamM = jnp.zeros(pair_shape, dtype=jnp.float64)
    beamM2 = jnp.zeros(pair_shape, dtype=jnp.float64)
    beamMulti = jnp.zeros(pair_shape, dtype=jnp.float64)
    beamC = jnp.zeros((sequence_length, 2), dtype = jnp.float64)
    valid_pairs = jnp.array([
       # A, G, C, U
        [0, 0, 0, 1], #A
        [0, 0, 1, 0], #G
        [0, 1, 0, 1], #C
        [1, 0, 1, 0]  #U
    ])

    return nucleotides, beamH, beamP, beamM, beamM2, beamMulti, beamC, valid_pairs

######################
###   Interface    ###
######################

def __main__(sequence):
    args = __init__(sequence)
    final_state = parse(*args)
    _, _, _, _, _, beamC = final_state
    partition_function = jnp.exp(beamC[-1].alpha)
    return partition_function, final_state

######################
### Core Functions ###
######################

@jax.jit
def quickselect_part(new_alphas: Array[jnp.float64], lower, upper) -> typing.Tuple[int, Array[jnp.float64]]:
    """
    Partition the new_alphas array into two parts. (Higher and lower than the pivot)
    Args: 
        new_alphas: An array of new_alphas
        lower: The lower bound of the array
        upper: The upper bound of the array
    Returns:
        A tuple of the split index and the partitioned new_alphas
    """
    pivot = new_alphas[upper].first
    def body_fun(state: typing.Tuple[int, int, Array[jnp.float64]]
                 ) -> typing.Tuple[typing.Tuple[int, int], 
                                   typing.Tuple[jax.Array, bool]]:
        lower, upper = state
        lower = jax.lax.while_loop(
            lambda λ: λ < upper and new_alphas[λ] < pivot,
            lambda λ: λ + 1,
            lower
        )
        upper = jax.lax.while_loop(
            lambda μ: μ > lower and new_alphas[μ] > pivot,
            lambda μ: μ - 1,
            upper
        )
        swap = lower < upper & new_alphas[lower] != new_alphas[upper]
        new_alphas_lower = jax.lax.select(swap, new_alphas[upper], new_alphas[lower])
        new_alphas_upper = jax.lax.select(swap, new_alphas[lower], new_alphas[upper])
        lower = jax.lax.select(new_alphas[lower] == new_alphas[upper], lower + 1, lower)
        return (lower, upper, new_alphas.at[lower].set(new_alphas_lower).at[upper].set(new_alphas_upper))

    lower, upper, new_alphas = jax.lax.while_loop(
        lambda state: state[0] < state[1],
        body_fun,
        (lower, upper, new_alphas)
    )

    return upper, new_alphas

@jax.jit
def quickselect(new_alphas: Array[jnp.float64], lower: int, upper: int, k: int) -> Array[jnp.float64]:
    """
    Find the k-th smallest element in new_alphas[lower:upper].
    Args:
        new_alphas: An array of new_alphas
        lower: The lower bound of the array
        upper: The upper bound of the array
        k: The k-th smallest element
    Returns:
        The k-th smallest element in new_alphas[lower:upper]
    """
    def cond_fun(state: typing.Tuple[int, int, int, jax.Array, bool]) -> bool:
        lower, upper, _, _, bool = state
        return (lower != upper) or bool
    
    def body_fun(state: typing.Tuple[int, int, int, jax.Array, bool]) -> typing.Tuple[int, int, int, jax.Array, bool]:
        lower, upper, k, new_alphas = state
        split, new_alphas = quickselect_part(new_alphas, lower, upper)
        length = split - lower + 1
        match = (length == k)
        lower = jax.lax.select(k < length, lower, jax.lax.select(match, split, split + 1))
        upper = jax.lax.select(k < length, split - 1, upper)
        k = jax.lax.select(k < length, k, k - length)
        return lower, upper, k, new_alphas, match
    
    lower, upper, _, new_alphas, _ = jax.lax.while_loop(cond_fun, body_fun, (lower, upper, k, new_alphas, False))
    return new_alphas[lower]


@jax.jit
def beam_prune(beamstep: jnp.ndarray,
               bestC: jnp.ndarray,
               beam: int) -> typing.Tuple[jnp.ndaray, float]:
    """
    Prune the beam of candidate alphas and return the best alpha.
    Args:
        beamstep:An array of shape (n, 3) where each row is [i, cand.alpha, cand.beta] The candidate alphas
        bestC: The best alpha of each positions
        beam: The beam size
    Returns:
        A tuple of the pruned beam and the best alpha. 
    """ 
    def new_alpha(item):
        iter_no, candidate_alpha = item[0], item[1]
        prev = iter_no - 1
        bestC_alpha = jax.lax.select(prev >= 0, bestC[prev], 0.0)
        return candidate_alpha + bestC_alpha
    new_alphas = jax.vmap(new_alpha)(beamstep)
    threshold = jax.lax.select(
        beam < new_alphas.shape[0],
        quickselect(new_alphas, 0, new_alphas.shape[0] - 1, new_alphas.shape[0] - beam),
        jnp.finfo(jnp.float64).min # -inf
    )
    mask = new_alphas > threshold
    beamstep = beamstep[mask]
    return beamstep, threshold

@jax.jit
def parse(sequence, beamH, beamP, beamM, beamM2, beamMulti, beamC, valid_pairs,
          type, source):
    nucleotides = jnp.array(
        [utils.GET_ACGU_NUM(nuc) for nuc in sequence], dtype=jnp.int32
    )
    sequence_length = len(sequence)
    evaluator = utils.Evaluate(type, source)
    if type == "Vienna":
        tetraloops, hexalops, triloops = evaluator.v_find_stale_hp(sequence, sequence_length)
    


