# LinearPartition.py
# Refactored from LinearPartition.cpp and LinearPartition.h
# Original Author: He Zhang
# Modified by: Chaebeom Sheen
# Edited Date: 2024-06-24

#----------------------------------------------------------
from typing import Tuple, Callable
from functools import wraps
import jax
import jax.numpy as jnp 

#----------------------------------------------------------

### Decorators ###

def to_jit(f: Callable, static = (0, )) -> Callable:
    """Decorator for just-in-time compilation using JAX.
    Primarily used for optimizing methods in the BeamCKYParser class.
    Args:
        f (Callable): function to be compiled
        static (tuple): static arguments for the function (Default: (0, ))"""
    jitted = jax.jit(f, static_argnums = static)
    @wraps(f)
    def wrapper(*args, **kwargs):
        return jitted(*args, **kwargs)
    return wrapper

def memoize_for_jit(f: Callable) -> Callable:
    """Decorator for memoizing functions to be compiled using JAX.
    Args:
        f (Callable): function to be memoized"""
    memo = {}
    @wraps(f)
    def helper(*args):
        if args not in memo:
            memo[args] = f(*args)
        return memo[args]
    return helper

### End: Decorators ###


#From utils.h
def GET_ACGU_NUM(x):
    return {'A': 0, 'C': 1, 'G': 2, 'U': 3}.get(x, 4)
N_TYPE_NUC = 5
N_TYPE_NUC_DOUBLE = 25
N_TYPE_NUC_TRIPLE = 125
EXPLICIT_MAX_LENGTH = 4
SINGLE_MIN_LENGTH = 0
SINGLE_MAX_LENGTH = 30

MULTI_MAX_LENGTH = 30
HAIRPIN_MAX_LENGTH = 30
BULGE_MAX_LENGTH = 30
INTERNAL_MAX_LENGTH = 30
SYMMETRIC_MAX_LENGTH = 15
ASYMMETRY_MAX_LENGTH = 28
MAX_LOOP = 30

class Evaluate:
    def __init__(self, type = "Default", source = "Default") -> None:
        if type == "Default":
            #import constants
            from .Utils import eternafold_weight as Eterna
            from .Utils import default_weight as Default 
            if source == "Default":
                self.source = Default
            elif source == "Eterna":
                self.source = Eterna
            else:
                raise ValueError("Error: Invalid source.")
            self._HELIX_STACKING = jnp.zeros((N_TYPE_NUC, N_TYPE_NUC, N_TYPE_NUC, N_TYPE_NUC), dtype = bool)
            self._ALLOWED_PAIRS = jnp.zeros((N_TYPE_NUC, N_TYPE_NUC), dtype = bool)
            self._CACHE_SINGLE = jnp.zeros((SINGLE_MAX_LENGTH + 1, SINGLE_MAX_LENGTH + 1), dtype = float)
            self.initialize()
            self.initialize_cachesingle()

        elif type == "Vienna":
            #import parameters
            from .Utils import energy_parameters
            self.source = energy_parameters
            self.NUM_TO_NUC = jnp.array([1, 2, 3, 4, 0, -1])
            #original code equivalent
            ##define NUM_TO_NUC(x) (x==-1?-1:((x==4?0:(x+1))))
            self.NUM_TO_PAIR = jnp.array([
                [0, 0, 0, 5], #A pairs with U
                [0, 0, 1, 0], #C pairs with G
                [0, 2, 0, 3], #G pairs with C and U
                [6, 0, 4, 0]  #U pairs with A and G
            ])
            #original code equivalent
            ##define NUM_TO_PAIR(x,y) (x==0? (y==3?5:0) : (x==1? (y==2?1:0) : (x==2 ? (y==1?2:(y==3?3:0)) : (x==3 ? (y==2?4:(y==0?6:0)) : 0))))
            self.NUC_TO_PAIR = jnp.zeros((5, 5)).at[1:, 1:].set(self.NUM_TO_PAIR)
            #original_code_equvialent
            ##define NUC_TO_PAIR(x,y) (x==1? (y==4?5:0) : (x==2? (y==3?1:0) : (x==3 ? (y==2?2:(y==4?3:0)) : (x==4 ? (y==3?4:(y==1?6:0)) : 0))))

    def initialize_cachesingle(self) -> None:
        for l_1 in range(SINGLE_MIN_LENGTH, SINGLE_MAX_LENGTH + 1):
            for l_2 in range(SINGLE_MIN_LENGTH, SINGLE_MAX_LENGTH + 1):
                if l_1 == 0 and l_2 == 0:
                    continue
                elif l_1 == 0:
                    self._CACHE_SINGLE[l_1, l_2] += self.source.bulge_length[l_2]
                elif l_2 == 0:
                    self._CACHE_SINGLE[l_1, l_2] += self.source.bulge_length[l_1]
                else:
                    self._CACHE_SINGLE[l_1, l_2] += self.source.internal_length[min(l_1 + l_2, INTERNAL_MAX_LENGTH)]
                    
                    if l_1 <= EXPLICIT_MAX_LENGTH and l_2 <= EXPLICIT_MAX_LENGTH:
                        idx = l_1 * EXPLICIT_MAX_LENGTH + l_2 if l_1 <= l_2 else l_2 * EXPLICIT_MAX_LENGTH + l_1
                        self._CACHE_SINGLE[l_1, l_2] += self.source.internal_explicit[idx]

                    if l_1 == l_2:
                        self._CACHE_SINGLE[l_1, l_2] += self.source.internal_symmetric_length[min(l_1, SYMMETRIC_MAX_LENGTH)]
                    else:
                        diff = abs(l_1 - l_2)
                        self._CACHE_SINGLE[l_1, l_2] += self.source.internal_asymmetry[min(diff, ASYMMETRY_MAX_LENGTH)]

    def initialize(self) -> None:

        allowed_pairs = [
            ('A', 'U'), ('U', 'A'),
            ('C', 'G'), ('G', 'C'),
            ('G', 'U'), ('U', 'G')
        ]
        
        for a, b in allowed_pairs:
            self._ALLOWED_PAIRS[GET_ACGU_NUM(a), GET_ACGU_NUM(b)] = True

        helix_pairs = [
            ('A', 'U', 'A', 'U'), ('A', 'U', 'C', 'G'), ('A', 'U', 'G', 'C'),
            ('A', 'U', 'G', 'U'), ('A', 'U', 'U', 'A'), ('A', 'U', 'U', 'G'),
            ('C', 'G', 'A', 'U'), ('C', 'G', 'C', 'G'), ('C', 'G', 'G', 'C'),
            ('C', 'G', 'G', 'U'), ('C', 'G', 'U', 'G'), ('G', 'C', 'A', 'U'),
            ('G', 'C', 'C', 'G'), ('G', 'C', 'G', 'U'), ('G', 'C', 'U', 'G'),
            ('G', 'U', 'A', 'U'), ('G', 'U', 'G', 'U'), ('G', 'U', 'U', 'G'),
            ('U', 'A', 'A', 'U'), ('U', 'A', 'G', 'U'), ('U', 'G', 'G', 'U')
        ]
        
        for x, y, z, w in helix_pairs:
            self._HELIX_STACKING[GET_ACGU_NUM(x), GET_ACGU_NUM(y), 
                                 GET_ACGU_NUM(z), GET_ACGU_NUM(w)] = True

    ### BEGIN: nucleotide based scores ###
    
    ##the original code used the following notation 
    #nuci1 -> nucleotide after nucleotides[i]
    #nuci_1 -> nucleotide before nucleotides[i]
    #the code below does not distinguish between the two
    #if the argument comes before nuc_i, it is denoted as nuc_i_1 and refers to nucleotides[i-1]
    #if the argument comes after nuc_i, it is denoted as nuc_i_1 and refers to nucleotides[i+1]

    #parameters: nucleotides[i], nucleotides[j]
    def base_pair_score(self, nuc_i: int, nuc_j: int) -> float:
        return self.source.base_pair[nuc_j*N_TYPE_NUC + nuc_i]
    
    #parameters: nucleotides[i], nucleotides[i+1], nucleotides[j-1], nucleotides[j]
    def helix_stacking_score(self, nuc_i_0: int, nuc_i_1: int, nuc_j_1: int, nuc_j_0: int) -> float:
        return self.source.helix_stacking[nuc_i_0 * N_TYPE_NUC_TRIPLE + nuc_j_0 * N_TYPE_NUC_DOUBLE + nuc_i_1 * N_TYPE_NUC + nuc_j_1]
    
    #parameters: nucleotides[i], nucleotides[j]
    def helix_closing_score(self, nuc_i: int, nuc_j: int) -> float:
        return self.source.helix_closing[nuc_i * N_TYPE_NUC + nuc_j]
    
    #parameters: nucleotides[i], nucleotides[i+1], nucleotides[j-1], nucleotides[j]
    def terminal_mismatch_score(self, nuc_i_0: int, nuc_i_1: int, nuc_j_0 :int, nuc_j_1 :int) -> float:
        return self.source.terminal_mismatch[nuc_i_0 * N_TYPE_NUC_TRIPLE + nuc_j_0 * N_TYPE_NUC_DOUBLE + nuc_i_1 * N_TYPE_NUC + nuc_j_1]
    
    #parameters: nucleotides[i]
    def bulge_nuc_score(self, nuc_i: int) -> float:
        return self.source.bulge_0x1_nucleotides[nuc_i]

    #parameters: nucleotides[i], nucleotides[j]
    def internal_nuc_score(self, nuc_i: int, nuc_j: int) -> float:
        return self.source.internal_1x1_nucleotides[nuc_i * N_TYPE_NUC + nuc_j]
    
    #parameters: nucleotides[i], nucleotides[i+1], nucleotides[j]
    def dangle_left_score(self, nuc_i : int, nuc_i_1 : int, nuc_j : int) -> float:
        return self.source.dangle_left[nuc_i * N_TYPE_NUC_DOUBLE + nuc_j * N_TYPE_NUC + nuc_i_1]
    
    #parameters: nucleotides[i], nucleotides[j-1], nucleotides[j]
    def dangle_right_score(self, nuc_i : int, nuc_j_1 : int, nuc_j : int) -> float:
        return self.source.dangle_right[nuc_i * N_TYPE_NUC_DOUBLE + nuc_j * N_TYPE_NUC + nuc_j_1]

    ### END: nucleotide based scores ###

    ### BEGIN: length based scores ###

    #parameters: hairpin beginning at nucleotide i, ending at nucleotide j
    def hairpin_score(self, i : int, j: int) -> float:
        return self.source.hairpin_length[min(j - i - 1, HAIRPIN_MAX_LENGTH)]

    def internal_length_score(self, length: int) -> float:
        return self.source.internal_length[min(length, INTERNAL_MAX_LENGTH)]

    def internal_explicit_score(self, length1: int, length2: int) -> float:
        length1 = min(length1, EXPLICIT_MAX_LENGTH)
        length2 = min(length2, EXPLICIT_MAX_LENGTH)
        if length1 > length2: 
            length1, length2 = length2, length1
        return self.source.internal_explicit(length1 * N_TYPE_NUC + length2)

    def intenal_symmetric_score(self, length: int) -> float:
        return self.source.internal_symmetric_length[min(length, SYMMETRIC_MAX_LENGTH)]

    def internal_asymmetry_score(self, length1: int, length2: int) -> float:
        diff = abs(length1 - length2)
        return self.source.internal_asymmetry[min(diff, ASYMMETRY_MAX_LENGTH)]

    def bulge_length_score(self, length: int) -> float:
        return self.source.bulge_length[min(length, BULGE_MAX_LENGTH)]

    def hairpin_at_least_score(self, length: int) -> float:
        return self.source.hairpin_length_at_least[min(length, HAIRPIN_MAX_LENGTH)]

    def buldge_at_least_score(self, length: int) -> float:
        return self.source.bulge_length_at_least[min(length, BULGE_MAX_LENGTH)]

    def internal_length_at_least_score(self, length: int) -> float:
        return self.source.internal_length_at_least[min(length, INTERNAL_MAX_LENGTH)]
    
    ### END: length based scores ###

    ###Begin: Junction Scores (Based on Nucleotide and Length based Scores) ###
    
    #parameters: i, j, nucleotides[i], nucleotides[i+1], nucleotides[j-1], nucleotides[j], length of the junction
    def score_junction_A(self, i: int, j: int, nuc_i_0: int, nuc_i_1: int, nuc_j_1: int, nuc_j_0: int, length: int) -> float:
        """The function calculates the score of a junction with nucleotides i, i+1, j-1, j
        Args:
            i (int): start index of the junction
            j (int): end index of the junction
            nuc_i_0 (int): nucleotide at index i
            nuc_i_1 (int): nucleotide at index i+1
            nuc_j_1 (int): nucleotide at index j-1
            nuc_j_0 (int): nucleotide at index j
            length (int): length of the junction
        """
        dangle_left = self.dangle_left_score(nuc_i_0, nuc_i_1, nuc_j_0) if i < length - 1 else 0.0
        dangle_right = self.dangle_right_score(nuc_i_0, nuc_j_1, nuc_j_0) if j > 0  else 0.0
        return self.helix_closing_score(nuc_i_0, nuc_j_0) + dangle_left + dangle_right
    
    #parameters: nucleotides[i], nucleotides[i+1], nucleotides[j-1], nucleotides[j]
    def score_junction_B(self, i: int, j: int, nuc_i_0: int, nuc_i_1: int, nuc_j_1: int, nuc_j_0: int) -> float:
        """The function calculates the score of a junction with nucleotides i, i+1, j-1, j
        Args:
            i (int): start index of the junction
            j (int): end index of the junction
            nuc_i_0 (int): nucleotide at index i
            nuc_i_1 (int): nucleotide at index i+1
            nuc_j_1 (int): nucleotide at index j-1
            nuc_j_0 (int): nucleotide at index j
        """
        #i and j are the indices of the junction but do not seem to appear in the function
        #these arguments remain as they are present in the original code for readability
        return self.helix_closing_score(nuc_i_0, nuc_j_0) + self.terminal_mismatch_score(nuc_i_0, nuc_i_1, nuc_j_0, nuc_j_1)
    
    def score_hairpin_length(self, length: int) -> float:
        return self.source.hairpin_length(min(length, HAIRPIN_MAX_LENGTH))
    
    def score_helix(self, nuc_i_0: int, nuc_i_1: int, nuc_j_1: int, nuc_j_0: int) -> float:
        return self.helix_stacking_score(nuc_i_0, nuc_i_1, nuc_j_1, nuc_j_0) + self.base_pair_score(nuc_i_1, nuc_j_1)
    
    def score_single_nuc(self, i: int, j: int, p: int, q: int, nuc_p: int, nuc_q: int) -> float:
        l_1, l_2 = p - i - 1, j - q - 1
        if l_1 == 0 and l_2 == 1:
            return self.bulge_nuc_score(nuc_q)
        elif l_1 == 1 and l_2 == 0:
            return self.bulge_nuc_score(nuc_p)
        elif l_1 == 1 and l_2 == 1:
            return self.internal_nuc_score(nuc_p, nuc_q)
        else:
            return 0.0
    
    def score_single(self, i: int, j: int, p: int, q: int, length: int,
                     nuc_i: int, nuc_i_1: int, nuc_j_1: int, nuc_j: int,
                     nuc_p_1: int, nuc_p: int, nuc_q: int, nuc_q_1: int) -> float:
        
        l_1, l_2 = p - i - 1, j - q - 1

        return sum(
            self._CACHE_SINGLE[l_1, l_2],
            self.base_pair_score(nuc_p, nuc_q),
            self.score_junction_B(i, j, nuc_i, nuc_i_1, nuc_j_1, nuc_j),
            self.score_junction_B(p, q, nuc_p, nuc_p_1, nuc_q_1, nuc_q),
            self.score_single_nuc(i, j, p, q, nuc_p, nuc_q)
        )
    
    def score_single_without_junction_B(self, i: int, j: int, p: int, q: int,
                                        nuc_p_1: int, nuc_p: int, nuc_q: int, nuc_q_1: int) -> float:
        l_1, l_2 = p - i - 1, j - q - 1
        return sum(
            self._CACHE_SINGLE[l_1, l_2],
            self.base_pair_score(nuc_p, nuc_q),
            self.score_single_nuc(i, j, p, q, nuc_p_1, nuc_q_1)
        )


    def score_multi(self, i: int, j: int, 
                    nuc_i: int, nuc_i_1: int, nuc_j_1: int, nuc_j: int,
                    length: int) -> float:
        return sum(self.score_junction_A(i, j, nuc_i, nuc_i_1, nuc_j_1, nuc_j, length),
                   self.source.multi_base,
                   self.source.multi_paired)

    def score_multi_unpaired(self, i: int, j: int) -> float:
        return (j - i + 1)*self.source.multi_unpaired 
    
    def score_M1(self, i: int, j: int, k: int, 
                 nuc_i_1: int, nuc_i: int, nuc_k: int, nuc_k_1: int, 
                 length: int) -> float:
        return sum(
            self.score_junction_A(k, i, nuc_k, nuc_k_1, nuc_i_1, nuc_i, length),
            self.score_multi_unpaired(k + 1, j),
            self.base_pair_score(nuc_i, nuc_k),
            self.source.multi_paired
        )

    def score_external_paired(self, i: int, j: int, 
                              nuc_i: int, nuc_i_1: int, nuc_j_1: int, nuc_j: int,
                              length: int) -> float:
        return sum(
            self.score_junction_A(i, j, nuc_i, nuc_i_1, nuc_j_1, nuc_j, length),
            self.source.external_paired
        )

    def score_external_unpaired(self, i: int, j: int):
        return (j - i + 1) * self.source.external_unpaired
    

    ### END: Junction Scores ###

    ###----------------------------------------------------------###

    #from utility_v.h

    ###BEGIN: Hairpin, Single Nucleotide Scores ###
    
    def v_find_stable_hp(self, seq: str, seq_length: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if_tetraloops = jnp.full(max(seq_length - 5, 0), -1)
        if_triloops= jnp.full(max(seq_length - 4, 0), -1)
        if_hexaloops = jnp.full(max(seq_length - 7, 0), -1)
        for i in range(0, seq_length - 5):
            if seq[i] == 'C' and seq[i + 5] == 'G':
                loop = seq[i: i + 6]
                if loop in self.source.tetraloops:
                    if_tetraloops[i] = list(self.source.tetraloops.keys()).index(loop)
        for i in range(0, seq_length - 4):
            if {seq[i], seq[i + 4]} != {'G', 'C'}:
                loop = seq[i: i + 5]
                if loop in self.source.triloops:
                    if_triloops[i] = list(self.source.triloops.keys()).index(loop)
        for i in range(0, seq_length - 7):
            if seq[i] == 'A' and seq[i + 7] == 'U':
                loop = seq[i: i + 8]
                if loop in self.source.hexaloops:
                    if_hexaloops[i] = list(self.source.hexaloops.keys()).index(loop)
        return if_tetraloops, if_hexaloops, if_triloops
    
    def v_hairpin_score(self, i: int, j: int,
                        nuc_i: int, nuc_i_1: int, nuc_1_j: int, nuc_j: int,
                        tetra_hex_tri_index = -1) -> int:
        '''Returns the score of a hairpin loop.
        Configuration Diagram
        5'- i i_1 ..(hairpin).. 1_j j -3'
        Original Code Equivalnt: v_score_hairpin''' 
        size = j - i -1
        typ = self.NUM_TO_PAIR[nuc_i, nuc_j]
        #self.NUM_TO_PAIR = jnp.array([
        #        [0, 0, 0, 5], #A pairs with U
        #        [0, 0, 1, 0], #C pairs with G
        #        [0, 2, 0, 3], #G pairs with C and U
        #        [6, 0, 4, 0]])  #U pairs with A and G
        pre = self.NUM_TO_NUC[nuc_i_1]
        #self.NUM_TO_NUC = [1, 2, 3, 4, 0, -1]
        post = self.NUM_TO_NUC[nuc_1_j]

        energy = self.source.hairpins[size] if size <= HAIRPIN_MAX_LENGTH else self.source.hairpins[HAIRPIN_MAX_LENGTH] + int(
            self.source.LOG_MULT * jnp.log(size / HAIRPIN_MAX_LENGTH)
        )

        if size < 3: 
            return energy

        if size == 4 and tetra_hex_tri_index > -1:
            return list(self.source.tetraloops.values())[tetra_hex_tri_index]
        elif size == 6 and tetra_hex_tri_index > -1:
            return list(self.source.hexaloops.values())[tetra_hex_tri_index]
        elif size == 3:
            if tetra_hex_tri_index > -1:
                return list(self.source.triloops.values())[tetra_hex_tri_index]
            else:
                energy += self.source.TerminalU if typ > 2 else 0 #typ > 2 if [AU, UA, GU, UG]
                return energy
    
        energy += self.source.hairpin_match[typ][pre][post]

        return energy

    def v_score_single(self, i: int, j: int, p: int, q: int,
                       nuc_i: int, nuc_i_1: int, nuc_1_j: int, nuc_j: int,
                       nuc_1_p: int, nuc_p: int, nuc_q: int, nuc_q_1: int) -> int:
        """
        Returns the score of a single nucleotide loop.
        Configuration Diagram
        5'- i i_1 .... 1_p p -- ⌉
        3'- j 1_j .... q_1 q -- ⌋ 
        Original Code Equivalent: v_score_single"""

        pre_up = self.NUM_TO_NUC[nuc_i_1]
        post_low = self.NUM_TO_NUC[nuc_1_j]
        post_up = self.NUM_TO_NUC[nuc_1_p]
        pre_low = self.NUM_TO_NUC[nuc_q_1]
        typ_end = self.NUM_TO_PAIR[nuc_i, nuc_j]
        typ_mid = self.NUM_TO_PAIR[nuc_p, nuc_q]
        n_up = p - i - 1
        n_low = j - q - 1
        n_long, n_short = n_up, n_low if n_up > n_low else n_low, n_up

        if n_long == 0:
            #both n_up and n_low are 0
            #Diagram
            #5'-ip-3'
            #3'-jq-5'
            #this constitutes a stack
            return self.source.stacks[typ_end, typ_mid]
        
        if n_short == 0:
            #one of the loops is 0
            #Diagram
            #5'-i....p-3'
            #...|../
            #3'-jq-5'
            #this constitutes a bulge
            energy = self.source.hairpins[n_long] if n_long <= MAX_LOOP else self.source.hairpins[MAX_LOOP] + int(
                self.source.LOG_MULT * jnp.log(n_long / MAX_LOOP)
            )
            if n_long == 1: #this is nearly a stack, stack energy should be added
                energy += self.source.stacks[typ_end, typ_mid]
            else:
                if typ_end > 2: 
                    energy += self.source.TerminalU
                if typ_mid > 2: 
                    energy += self.source.TerminalU

            return energy
        
        else: 
            #we have an internal loop
            if n_short == 1: #this constitutes a 1*n loop 
                if n_long == 1: #this is a 1*1 loop
                    energy = self.source.in_loop_1x1[typ_end, typ_mid, pre_up, post_low]
                    return energy
                if n_long == 2: #this is a 2 * 1 loop
                    if n_up == 1:
                        #n_up is shorter, n_low is longer
                        energy = self.source.in_loop_2x1[typ_end, typ_mid, pre_up, pre_low, post_low]
                    else: 
                        #n_low is shorter, n_up is longer
                        energy = self.source.in_loop_2x1[typ_mid, typ_end, pre_low, pre_up, post_up]

                    return energy
                else:
                    #this is a 1 * n loop
                    energy = self.source.internal_loop[n_long + 1] if n_long < MAX_LOOP else self.source.internal_loop[MAX_LOOP] + int(
                        self.source.LOG_MULT * jnp.log(n_long / MAX_LOOP)
                    )
                    energy += min(self.source.MAX_INTERNAL, (n_long - n_short)*self.source.INTERNAL_MULT)

                    energy += self.source.internal_1xn_correction[
                                typ_end, pre_up, post_low
                                ] + self.source.internal_1xn_correction[
                                    typ_mid, pre_low, post_up
                                ]
                    
                    return energy
                
            elif n_short == 2: #this constitutes a 2 * n loop (n > 2)
                if n_long == 2: #this is a 2 * 2 loop
                    energy = self.source.in_loop_2x2[typ_end, typ_mid, pre_up, post_up, pre_low, post_low]
                    return energy
                elif n_long == 3: #this is a 2 * 3 loop
                    energy = self.source.internal_loop[3 + 2] + self.source.INTERNAL_MULT * (3 - 2)
                    energy += self.source.internal_2x3_correction[
                                typ_end, pre_up, post_low
                                ]+ self.source.internal_2x3_correction[
                                    typ_mid, pre_low, post_up
                                ]
                    return energy
                
                #else: continue to the general case
            
            #general case (no else here!)

            n_tot = n_long + n_short
            energy = self.source.internal_loop[n_tot] if n_tot < MAX_LOOP else self.source.internal_loop[MAX_LOOP] + int(
                self.source.LOG_MULT * jnp.log(n_tot / MAX_LOOP)
            )

            energy += min(self.source.MAX_INTERNAL, (n_long - n_short)*self.source.INTERNAL_MULT)

            energy += self.source.internal_match[
                        typ_end, pre_up, post_low
                        ] + self.source.internal_match[
                            typ_mid, pre_low, post_up
                        ]
        return energy
    ###END: Hairpin, Single Nucleotide Scores ###

    ###BEGIN: Multi-Branch Loop Scores ###

    def v_score_multi_paired(self, typ: int, pre: int, post: int,
                           dangle_mode: int) -> int: 
        """Original Code Equvialent: E_MLstem
        """
        energy = 0
        if dangle_mode != 0:
            if pre >= 0 and post >= 0: #if both are one of ACGUN
                energy += self.source.multiloop_match[typ, pre, post]
            elif pre >= 0: #if post is out of range
                energy += self.source.dangle5[typ, pre]
            elif post >= 0: #if pre is out of range
                energy += self.source.dangle3[typ, post]
        
        if typ > 2: #if typ is one of AU, UA, GU, UG (i.e. if typ contians uracil)
            energy += self.source.TerminalU
        
        energy += self.source.MULTLOOP_IN

        return energy

    def v_score_mult_1nuc(self, i: int, j: int, k: int,
                        nuc_1_i: int, nuc_i: int, nuc_k: int, nuc_k_1: int,
                        length: int, dangle_mode: int) -> int: 
        typ = self.NUM_TO_PAIR[nuc_i, nuc_k]
        pre = self.NUM_TO_NUC[nuc_1_i]
        post = self.NUM_TO_NUC[nuc_k_1]
        
        energy = self.v_score_multi_stem(typ, pre, post, dangle_mode)

        return energy
        
    def v_score_multi_unpaired(self, i: int, j: int) -> int: 
        return 0 
    
    def v_score_multi(self, i: int, j: int, 
                      nuc_i: int, nuc_i_1: int, nuc_1_j: int, nuc_j: int, 
                      length: int, dangle_mode: int) -> int:
        typ = self.NUM_TO_PAIR[nuc_j, nuc_i]
        pre = self.NUM_TO_NUC[nuc_i_1]
        post = self.NUM_TO_NUC[nuc_1_j]

        energy = self.v_score_multi_paired(
            typ, pre, post, dangle_mode
            ) + self.source.MULTLOOP_CLS 
        
        return energy

    ###END: Multi-Branch Loop Scores ###

    ###BEGIN: External Loop Scores ###

    def v_score_external_paired(self, i: int, j: int, 
                                nuc_1_i: int, nuc_i: int, nuc_j: int, nuc_j_1: int,
                                length: int, dangle_mode: int) -> int:
        typ = self.NUM_TO_PAIR[nuc_i, nuc_j]
        prev = self.NUM_TO_NUC[nuc_1_i]
        post = self.NUM_TO_NUC[nuc_j_1]
        energy = 0
        if dangle_mode != 0:
            if prev >= 0 and post >= 0:
                energy += self.source.exterior_match[typ, prev, post]
            elif prev >= 0:
                energy += self.source.dangle5[typ, prev]
            elif post >= 0:
                energy += self.source.dangle3[typ, post]

        if typ > 2:
            energy += self.source.TerminalU
        return energy
        
    def v_score_external_unpaired(self, i: int, j: int) -> int:
        return 0

    ###END: External Loop Scores ###
    

