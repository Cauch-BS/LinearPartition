import jax 
import typing
import jax.numpy as jnp
from jax import Array
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
    nucleotides = jnp.array(
        [utils.GET_ACGU_NUM(nuc) for nuc in sequence], dtype=jnp.int32
    )
    State = jnp.dtype([
        ('alpha', jnp.float64),
        ('beta', jnp.float64),
        ])
    shape = (sequence_length, sequence_length, BEAM_SIZE)
    beamH = jnp.zeros(shape, dtype=jnp.float64)
    beamP = jnp.zeros(shape, dtype=jnp.float64)
    beamM = jnp.zeros(shape, dtype=jnp.float64)
    beamM2 = jnp.zeros(shape, dtype=jnp.float64)
    beamMulti = jnp.zeros(shape, dtype=jnp.float64)
    beamC = jnp.zeros((sequence_length, ), dtype = State)
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
def quickselect_part(scores: Array[jnp.float64], lower, upper) -> typing.Tuple[int, Array[jnp.float64]]:
    pivot = scores[upper]
    def body_fun(state: typing.Tuple[int, int]
                 ) -> typing.Tuple[typing.Tuple[int, int], 
                                   typing.Tuple[jax.Array, bool]]:
        lower, upper = state
        lower = jax.lax.while_loop(
            lambda λ: λ < upper and scores[λ] < pivot,
            lambda λ: λ + 1,
            lower
        )
        upper = jax.lax.while_loop(
            lambda μ: μ > lower and scores[μ] > pivot,
            lambda μ: μ - 1,
            upper
        )

        swap = lower < upper & scores[lower] != scores[upper]
        scores_lower = jax.lax.select(swap, scores[upper], scores[lower])
        scores_upper = jax.lax.select(swap, scores[lower], scores[upper])
        lower = jax.lax.select(scores[lower] == scores[upper], lower + 1, lower)
        return (lower, upper), (scores.at[lower].set(scores_lower).at[upper].set(scores_upper), swap)

    (lower, upper), (scores, _) = jax.lax.while_loop(
        lambda state: state[0] < state[1],
        body_fun,
        (lower, upper)
    )
    return upper, scores

@jax.jit
def quickselect(scores: Array, lower: int, upper: int,
                k: int) -> typing.Tuple[int, Array]:
    def else_fn(state: typing.Tuple[Array, int, int, int]
                ) -> jnp.float64:
        scores, upper, lower, k = state
        split = quickselect_part(scores, lower, upper)
        length = split - lower + 1
        return jax.lax.select(
            length == k,
            scores[split],
            jax.lax.select(
                length > k,
                quickselect(scores, lower, split - 1, k),
                quickselect(scores, split + 1, upper, k - length)
            )
        )

    result = jax.lax.select(
        lower == upper,
        scores[lower],

    )



@partial(jax.jit, static_argnums=(8,))
def parse(nucleotides, beamH, beamP, beamM, beamM2, beamMulti, beamC, valid_pairs):
    sequence_length = nucleotides.shape[0]


