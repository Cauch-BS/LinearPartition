# LinearPartition.py
# Refactored from LinearPartition.cpp and LinearPartition.h
# Original Author: He Zhang
# Modified by: Chaebeom Sheen
# Edited Date: 2024-06-24

#----------------------------------------------------------
from typing import Any, Tuple, Callable
from collections import namedtuple
from functools import wraps
import math
import jax
import jax.numpy as jnp 
import time
import scipy.constants as const

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

#----------------------------------------------------------
#import constants
from .Utils import eternafold_weight as Eterna
from .Utils import default_weight as Default 

#----------------------------------------------------------

#From utils.h
GET_ACGU_NUM = lambda x: {'A': 0, 'C': 1, 'G': 2, 'U': 3}.get(x, 4)
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

_HELIX_STACKING = jnp.zeros((N_TYPE_NUC, N_TYPE_NUC, N_TYPE_NUC, N_TYPE_NUC), dtype = bool)
_ALLOWED_PAIRS = jnp.zeros((N_TYPE_NUC, N_TYPE_NUC), dtype = bool)
_CACHE_SINGLE = jnp.zeros((SINGLE_MAX_LENGTH + 1, SINGLE_MAX_LENGTH + 1), dtype = float)

class Evaluate:
    def __init__(self, source = "Default") -> None:
        if source == "Default":
            self.source = Default
        elif source == "Eterna":
            self.source = Eterna
        else:
            raise ValueError("Error: Invalid source.")
        self.initialize()
        self.initialize_cachesingle()
        

    def initialize_cachesingle(self) -> None:
        global _CACHE_SINGLE

        for l_1 in range(SINGLE_MIN_LENGTH, SINGLE_MAX_LENGTH + 1):
            for l_2 in range(SINGLE_MIN_LENGTH, SINGLE_MAX_LENGTH + 1):
                if l_1 == 0 and l_2 == 0:
                    continue
                elif l_1 == 0:
                    _CACHE_SINGLE[l_1, l_2] += self.source.bulge_length[l_2]
                elif l_2 == 0:
                    _CACHE_SINGLE[l_1, l_2] += self.source.bulge_length[l_1]
                else:
                    _CACHE_SINGLE[l_1, l_2] += self.source.internal_length[min(l_1 + l_2, INTERNAL_MAX_LENGTH)]
                    
                    if l_1 <= EXPLICIT_MAX_LENGTH and l_2 <= EXPLICIT_MAX_LENGTH:
                        idx = l_1 * EXPLICIT_MAX_LENGTH + l_2 if l_1 <= l_2 else l_2 * EXPLICIT_MAX_LENGTH + l_1
                        _CACHE_SINGLE[l_1, l_2] += self.source.internal_explicit[idx]

                    if l_1 == l_2:
                        _CACHE_SINGLE[l_1, l_2] += self.source.internal_symmetric_length[min(l_1, SYMMETRIC_MAX_LENGTH)]
                    else:
                        diff = abs(l_1 - l_2)
                        _CACHE_SINGLE[l_1, l_2] += self.source.internal_asymmetry[min(diff, ASYMMETRY_MAX_LENGTH)]

    def initialize(self) -> None:
        global _ALLOWED_PAIRS, _HELIX_STACKING

        allowed_pairs = [
            ('A', 'U'), ('U', 'A'),
            ('C', 'G'), ('G', 'C'),
            ('G', 'U'), ('U', 'G')
        ]
        
        for a, b in allowed_pairs:
            _ALLOWED_PAIRS[GET_ACGU_NUM(a), GET_ACGU_NUM(b)] = True

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
            _HELIX_STACKING[GET_ACGU_NUM(x), GET_ACGU_NUM(y), GET_ACGU_NUM(z), GET_ACGU_NUM(w)] = True

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
        if length1 > length2: length1, length2 = length2, length1
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
            _CACHE_SINGLE[l_1, l_2],
            self.base_pair_score(nuc_p, nuc_q),
            self.score_junction_B(i, j, nuc_i, nuc_i_1, nuc_j_1, nuc_j),
            self.score_junction_B(p, q, nuc_p, nuc_p_1, nuc_q_1, nuc_q),
            self.score_single_nuc(i, j, p, q, nuc_p, nuc_q)
        )
    
    def score_single_without_junction_B(self, i: int, j: int, p: int, q: int,
                                        nuc_p_1: int, nuc_p: int, nuc_q: int, nuc_q_1: int) -> float:
        l_1, l_2 = p - i - 1, j - q - 1
        return sum(
            _CACHE_SINGLE[l_1, l_2],
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

#----------------------------------------------------------

#From LinearPartition.h

StandTemp = 298.15 #Standard temperature in Kelvin
kB = const.Boltzmann #Boltzmann constant (in units of J/K)
β = 1.0 / (kB * StandTemp) #thermodynamic beta (e.g. thermodynamic coldness) in units of 1/J

def hash_pair(p: Tuple[Any, Any]) -> int:
    hash1 = hash(p[0])
    hash2 = hash(p[1])
    return hash1 ^ hash2

def comp(left: list, right: list) -> bool:
    if left[0] == right[0]:
        return left[1] < right[1]
    return left[0] < right[0]

#From CONTRAfold
#cubic polynomial approximations for log(e^x + 1)
def fast_log_exp_plus_one(x):
    #error tolerance: 0.00000705
    if x < 0.661537:
        return (((-0.0065591595 * x + 0.1276442762) * x + 0.4996554598) * x + 0.6931542306)
    elif x < 1.63202:
        return (((-0.0155157557 * x + 0.1446775699) * x + 0.4882939746) * x + 0.6958092989)
    elif x < 2.49126:
        return (((-0.0128909247 * x + 0.1301028251) * x + 0.5150398748) * x + 0.6795585882)
    elif x < 3.37925:
        return (((-0.0072142647 * x + 0.0877540853) * x + 0.6208708362) * x + 0.5909675829)
    elif x < 4.42617:
        return (((-0.0031455354 * x + 0.0467229449) * x + 0.7592532310) * x + 0.4348794399)
    elif x < 5.78907:
        return (((-0.0010110698 * x + 0.0185943421) * x + 0.8831730747) * x + 0.2523695427)
    elif x < 7.81627:
        return (((-0.0001962780 * x + 0.0046084408) * x + 0.9634431978) * x + 0.0983148903)
    else:
        return (((-0.0000113994 * x + 0.0003734731) * x + 0.9959107193) * x + 0.0149855051)

#cubic polynomial approximations for log(e^x + e^y)
def fast_log_plus_equals(x, y):
    if x < y:
        x, y = y, x
    if y > -2e20 / 2 and x - y < 11.8624794162:
        x = fast_log_exp_plus_one(x - y) + y

    return x

#cubic polynomial approximations
def fast_exp(x):
    #error tolerance: 0.0000496
    if x < -9.91152:
        return 0
    elif x < -5.86228:
        return (((0.0000803850 * x + 0.0021627428) * x + 0.0194708555) * x + 0.0588080014)
    elif x < -3.83966:
        return (((0.0013889414 * x + 0.0244676474) * x + 0.1471290604) * x + 0.3042757740)
    elif x < -2.4915:
        return (((0.0072335607 * x + 0.0906002677) * x + 0.3983111356) * x + 0.6245959221)
    elif x < -1.48054:
        return (((0.0232410351 * x + 0.2085645908) * x + 0.6906367911) * x + 0.8682322329)
    elif x < -0.672505:
        return (((0.0573782771 * x + 0.3580258429) * x + 0.9121133217) * x + 0.9793091728)
    elif x < 0:
        return (((0.1199175927 * x + 0.4815668234) * x + 0.9975991939) * x + 0.9999505077)
    elif x < 46.052:
        return math.exp(x)
    else:
        return float(10 ** 20)

#----------------------------------------------------------
#From LinearPartition.cpp

#----------------------------------------------------------
#From LinearPartition.h

#beginning the BeamCYKParser class

#TODO: USE VECTORIZED OPTIONS TO SIMPLIFY (CURRENTLY USING FOR LOOPS FOR CLARITY)

State = namedtuple('State', ['alpha', 'beta'], 
                   defaults = (float('-inf'), float('-inf')))

class BeamCKYParser:
    def __init__(self, seq: str, beam_size: int = 100, no_sharp_turn: bool = True, is_verbose: bool = False, 
               bpp_file: str = '', 
               float_only: bool = False, bpp_cutoff: float = 0.0, forest_file: str ="", mea_: bool = False, 
               gamma: float = 3.0, mea_file_index: str = "",
               bpseq :bool = False, threshknot_ :bool = False, threshknot_threshold: float = 0.3,
               threshknot_file_index: str = "",
               source :str = "Default",
               shape_file_path ="", is_fasta=False, dangle_mode: int = 1) -> None:
        self.seq = seq
        self.seq_length = len(seq)

        self.beam = beam_size
        self.no_sharp_turn = no_sharp_turn
        self.is_verbose = is_verbose
        self.bpp_file = bpp_file
        self.float_only = float_only
        self.bpp_cutoff = bpp_cutoff
        self.forest_file = forest_file
        self.mea_ = mea_
        self.gamma = gamma
        self.mea_file_index = mea_file_index
        self.bpseq = bpseq
        self.threshknot_ = threshknot_
        self.threshknot_threshold = threshknot_threshold
        self.threshknot_file_index = threshknot_file_index
        self.source = source
        self.shape_file_path = shape_file_path
        self.is_fasta = is_fasta
        self.dangle_mode = dangle_mode
 
        self.bestH = [{i: State() for i in range(self.seq_length)} for _ in range(self.seq_length)]
        self.bestP = [{i: State() for i in range(self.seq_length)} for _ in range(self.seq_length)]
        self.Pij = jnp.zeros((self.seq_length + 1, self.seq_length + 1))
        self.bestM2 = [{i: State() for i in range(self.seq_length)} for _ in range(self.seq_length)]
        self.bestMulti = [{i: State() for i in range(self.seq_length)} for _ in range(self.seq_length)]
        self.bestM = [{i: State() for i in range(self.seq_length)} for _ in range(self.seq_length)]
        self.if_tetraloops = []
        self.if_hexaloops = []
        self.if_triloops = []
        self.bestC = [State() for _ in range(self.seq_length)]
        self.nucs = [GET_ACGU_NUM[nuc] for nuc in self.seq]

#----------------------------------------------------------
#From bpp.cpp

    ### Begin: Output functions ###

    #prints bpp to output_file
    def output_to_file(self, output_file: str, turn: int = 3) -> None:
        """Output base pairing probability matrix to file.
        Args:
            output_file (str): output file name
            turn (int): "turn" is the minimal distance at which base pairing can occur.
            if self.no_sharp_turn (bool), no sharp turn is allowed (default turn = 3)"""
        
        if output_file:
            self.output_file = output_file
            print("Outputting base pairing probability matrix to file: ", self.output_file)
            self.writable_file = open(self.output_file, "w")
            if not self.writable_file:
                raise Exception("Error: cannot open file ", output_file)
            
            turn = turn if self.no_sharp_turn else 0

            for i in range(1, self.seq_length + 1):
                for j in range(i + turn + 1, self.seq_length + 1):
                    self.writable_file.write(f"{i} {j} {self.Pij[i, j]:.5f}\n")
            
            self.writable_file.write("end of probability matrix \n")
            self.writable_file.close()
            print("Done!")
        else:
            raise ValueError("Error: output file not specified.")
    
    #prints MEA (maximum expected accuracy) base pair sequence to output_file
    def output_MEA_bpseq(self, output_file:str, pairs: dict[int, int], seq: str) -> None:
        """Output MEA (maximum expected accurate) base pairs in bpseq format to file.
        If file is not specified, print to stdout."""
        if output_file:
            self.output_file = output_file
            print("Outputting base pairs in bpseq format to {output_file}...")
            self.writable_file = open(self.output_file, "w")
            if not self.writable_file:
                raise Exception(f"Error: cannot open file {self.output_file}")
            for i in range(1, self.seq_length + 1):
                j = pairs[i] if i < len(pairs) else 0
                nucleotide = seq[i - 1]
                self.writable_file.write(f"{i} {nucleotide} {j}\n")
            self.writable_file.write("end of bpseq file\n")
            self.writable_file.close()
            print("Done!")

        else:
            for i in range(1, self.seq_length + 1):
                j = pairs[i] if i < len(pairs) else 0
                nucleotide = seq[i - 1]
                print(f"{i} {nucleotide} {j}")

    ### Main Functions (only consists of Pure Functions (i.e. no prints or reference to global values)) ###
    
    #modifies the base pairing matrix self.Pij
    def calc_pair_prob(self, viterbi: State) -> None:
        """Calculate base pairing probability matrix.
        Args:
            viterbi (State): Viterbi state of the sequence"""
        for j in range(self.seq_length):
            item = self.bestP[j] #bestP[j] is a list of tuples
            i = item[0]
            state = State(item[1])
            temp_prob_inside = state.alpha + state.beta - viterbi.alpha #viterbi.alpha is the best score
            if temp_prob_inside > -9.91152:
                prob = fast_exp(temp_prob_inside)
                if prob > 1.0: prob = 1.0
                if prob < self.bpp_cutoff: prob = 0.0
                self.Pij = self.Pij.at[i + 1, j + 1].set(prob)

    #returns the secondary structure of the sequence between i and j
    def back_trace(self, i: int, j: int, back_pointer: jnp.ndarray) -> str:
        """Recursively backtraces the secondary structure of the sequence between i and j.
        Args:
            i (int): start index
            j (int): end index
            back_pointer (jnp.ndarray): back pointer matrix (-1: unpaired, 0: no pointer, k: paired with k)"""
        if i > j:
            return ""
            
        if back_pointer[i, j] == -1:
            if i == j: 
                return "."
            else:
                return "." + self.back_trace(i + 1, j, back_pointer)
        
        elif back_pointer[i, j] != 0:
            k = back_pointer[i, j]
            assert (k + 1 > 0) and (k + 1 < self.seq_length + 1), "Error: back_pointer out of range."
            if k == j: temp = ""
            else:
                temp = self.back_trace(k + 1, j, back_pointer)
            
            return "(" + self.back_trace(i +1, k - 1, back_pointer) + ")" + temp

    #converts secondary structure ..((..)).. to dictionary of pairs
    #e.g. ..((..)).. -> {3: 8, 4: 7}
    def get_pairs(self, structure) -> dict[int, int]:
        """Convert secondary structure to dictionary of pairs.
        Args:
            structure (str): secondary structure string (e.g. ..((..))..)"""
        pairs = {}
        stack = []
        for i in range(len(structure)):
            if structure[i] == "(":
                stack.append(i)
            elif structure[i] == ")":
                j = stack.pop()
                pairs[j] = i
                pairs[i] = j
        return pairs
    
 
    def thresh_knot(self) -> None:
        """Find pairs with probability greater than the threshold value.
        Threshold value is set by self.threshknot_threshold.
        Pairs are stored in self.pairs."""

        row_prob = jnp.zeros(self.seq_length + 1)
        prob_list = jnp.zeros((self.seq_length + 1, self.seq_length + 1))

        for i in range(1, self.seq_length + 1):
            for j in range(i + 1, self.seq_length + 1):
                prob = self.Pij[i, j]
                if prob >= self.threshknot_threshold:
                    prob_list = prob_list.at[i, j].set(prob)
                    row_prob = row_prob.at[j].set(max(row_prob[i], prob))
                    row_prob = row_prob.at[j].set(max(row_prob[j], prob))

        pairs = dict()
        visited = set()

        for i in range(1, self.seq_length + 1):
            for j in range(i + 1, self.seq_length + 1):
                prob = prob_list[i, j]
                if (prob == row_prob[i]) and (prob == row_prob[j]): #if prob is the maximum
                    if (i in visited) or (j in visited):
                        continue
                    visited.add(i)
                    visited.add(j)
                    pairs[i] = j
                    pairs[j] = i
        
        self.pairs = pairs
    
    def pair_prob_mea(self) -> None:
        """Calculate the maximum expected accuracy (MEA) base pair probability from Pij."""
        probs = jnp.delete(jnp.delete(self.Pij, 0, 0), 0, 1)
        partitions = 1 - jnp.sum(probs, axis = 0) - jnp.sum(probs, axis = 1) + jnp.diag(probs) 
        #partitions[i] = 1 - sum(probs[i, :]) - sum(probs[:, i]) + probs[i, i]

        paired = jax.vmap(lambda row: jnp.nonzero(row)[0])( #sorts the indices of non-zero elements in each row
            probs > 0 #boolean mask to extract non-zero probabilities
        )

        partition_slices = jnp.zeros((self.seq_length, self.seq_length))
        back_pointer = jnp.zeros((self.seq_length, self.seq_length)) 
        #back_pointer[i, j] = k means i and j are paired (-1: unpaired, 0: no pointer)
        partition_slices = partition_slices.at[jnp.diag_indices(self.seq_length)].set(partitions)
        back_pointer = back_pointer.at[jnp.diag_indices(self.seq_length)].set(-1)
        #initial values for partition_slices and back_pointer

        #filling the partition_slices and back_pointer matrices
        #uses MacCaskill's Dynamic Algorithm to fill the initial values
        for k in range(1, self.seq_length):
            #k represents the distance between i and j
            #thus this loop iterates over the "k"-th diagonal of the matrix (excluding the main diagonal)
            for i in range(self.seq_length - k):
                j = i + k
                partition_slices.at[i, j].set(partitions[i] + partition_slices[i+1, j]) #dynamic partition calculation
                back_pointer = back_pointer.at[i, j].set(-1)
                for i_ in paired[i]:
                    if i_ > j: break
                    elif i_ < j: temp_part_i_j = partition_slices[i_ + 1, j]
                    else: temp_part_i_j = 0
                    temp_prob = 2 * self.gamma * probs[i, i_] + partition_slices[i + 1][i_ -1] + temp_part_i_j
                    if temp_prob > partition_slices[i][j]:
                        partition_slices[i][j] = temp_prob
                        back_pointer[i][j] = i_
        #back trace to find the secondary structure
        self.structure = self.back_trace_opt(0, self.seq_length - 1, back_pointer)

    def outside(self, next_pair: list[int]) -> None:
        evaluate = Evaluate(self.source)
        #initialize the bestC array
        self.bestC[self.seq_length - 1].beta = 0.0

        for j in range(self.seq_length - 1, 0, -1):
            nuc_j = self.nucs[j]
            nuc_j_1 = self.nucs[j + 1] if j + 1 < self.seq_length else -1

            beamstepH = self.bestH[j]
            beamstepMulti = self.bestMulti[j]
            beamstepP = self.bestP[j]
            beamstepM2 = self.bestM2[j]
            beamstepM = self.bestM[j]
            beamstepC = self.bestC[j]
        
            #beam search on C
            #C <- C + U(=unpaired)
            if j < self.seq_length - 1:
                new_score = evaluate.score_external_unpaired(j + 1, j + 1)
                beamstepC.beta = fast_log_plus_equals(
                    beamstepC.beta, 
                    self.bestC[j + 1].beta + new_score
                )
            
            #beam search on M
            for i, state in beamstepM.items():
                if j < self.seq_length - 1:
                    new_score = evaluate.score_multi_unpaired(j + 1, j + 1)
                    state.beta = fast_log_plus_equals(
                        state.beta,
                        self.bestM[j+1][i].beta + new_score
                    )
            
            #beam search on M2
            for i, state in beamstepM2.items():
                # if multi-loop
                for p in range(i - 1, max(i - SINGLE_MAX_LENGTH - 1, 0), -1):
                    nuc_p = self.nucs[p]
                    q = next_pair[nuc_p][j]
                    if q == -1:
                        break
                    elif (q != -1) and (i - 1  - p <= SINGLE_MAX_LENGTH):
                        new_score = sum(evaluate.score_multi_unpaired(p + 1, i - 1), 
                                        evaluate.score_multi_unpaired(j + 1, q - 1)
                        )
                        state.beta = fast_log_plus_equals(
                            state.beta,
                            self.bestMulti[q][p].beta + new_score
                        )

                # M2 <- M
                state.beta = fast_log_plus_equals(
                    state.beta,
                    beamstepM[i].beta
                )

            #beam search on P
            for i, state in beamstepP.items():
                nuc_i_1 = self.nucs[i - 1] if i - 1 >= 0 else -1
                nuc_i = self.nucs[i]
                #1. P <- P + U
                if (i > 0) and (j < self.seq_length - 1):
                    precomputed = evaluate.score_junction_B(j, i, nuc_j, nuc_j_1, nuc_i_1, nuc_i)

                    for p in range(i -1, max(i - SINGLE_MAX_LENGTH - 1, 0)):
                        nuc_p = self.nucs[p]
                        nuc_p_1 = self.nucs[p + 1]
                        q = next_pair[nuc_p][j]

                        while q != -1 and (i - p) + (q - j) - 2 <= SINGLE_MAX_LENGTH:
                            nuc_q = self.nucs[q]
                            nuc_q_1 = self.nucs[q - 1]

                            if p == i-1 and q == j + 1:
                                #calculate score for helix
                                new_score = evaluate.score_helix(
                                    nuc_p, nuc_p_1, nuc_q_1, nuc_q
                                )
                                state.beta = fast_log_plus_equals(
                                    state.beta,
                                    self.bestP[q][p].beta + new_score
                                )

                                #TODO: Add Suppot for Vaxpress (#ifdef lpv)

                            else:
                                #calculate score for single branch
                                new_score = sum(
                                    evaluate.score_junction_B(
                                    p, q, nuc_p, nuc_p_1, nuc_q_1, nuc_q
                                    ),
                                    precomputed,
                                    evaluate.score_single_without_junction_B(
                                        p, q, i, j, nuc_i_1, nuc_i, nuc_j, nuc_j_1
                                    )
                                )
                                
                                state.beta = fast_log_plus_equals(
                                    state.beta,
                                    self.bestP[q][p].beta + new_score
                                )
                            
                            q = next_pair[nuc_p][q]
                
                #2. P <- M
                if (i > 0) and (j < self.seq_length - 1):
                    new_score = evaluate.score_M1(
                        i, j, j, nuc_i_1, nuc_i, nuc_j, nuc_j_1, self.seq_length
                    )
                    
                    state.beta = fast_log_plus_equals(
                        state.beta,
                        beamstepM[i].beta + new_score
                    )

                #3. P <- M2 and M <- M2
                k = i - 1
                if (k > 0) and (self.bestM[k]):
                    new_score = evaluate.score_M1(
                        i, j, j, nuc_i_1, nuc_i, nuc_j, nuc_j_1, self.seq_length
                    )
                    m1_alpha = new_score
                    m1_plus_P_alpha = state.alpha + m1_alpha

                    for m, m_state in self.bestM[k].items():
                        state.beta = fast_log_plus_equals(
                            state.beta,
                            beamstepM2[m].beta + m_state.alpha + m1_alpha
                        )
                        m_state.beta = fast_log_plus_equals(
                            m_state.beta,
                            beamstepM2[m].beta + m1_plus_P_alpha
                        )

                #4. C <-  P + C
                k = i - 1
                if k > 0:
                    nuc_k = nuc_i_1
                    nuc_k_1 = nuc_i

                    new_score = evaluate.score_external_paired(
                        k + 1, j, nuc_k, nuc_k_1, nuc_j, nuc_j_1, self.seq_length
                    )

                    external_paired_alpha_plus_beamstepC_beta = beamstepC.beta + new_score

                    self.bestC[k].beta = fast_log_plus_equals(
                        self.bestC[k].beta,
                        state.alpha + external_paired_alpha_plus_beamstepC_beta
                    )
                    state.beta = fast_log_plus_equals(
                        state.beta,
                        self.bestC[k].alpha + external_paired_alpha_plus_beamstepC_beta
                    )

                    new_new_score = evaluate.score_external_paired(
                        0, j, -1, self.nucs[0], nuc_j, nuc_j_1, self.seq_length)
                    
                    state.beta = fast_log_plus_equals(
                        state.beta,
                        beamstepC.beta + new_new_score
                    )

            #beam search on Multi
            
            for i, state in beamstepMulti.items():
                nuc_i = self.nucs[i]
                nuc_i_1 = self.nucs[i + 1]
                jnext = next_pair[nuc_i][j]

                ## 1. extend (i, j) to (i, jnext)
                if jnext != -1:
                    new_score = evaluate.score_multi_unpaired(j, jnext -1)
                    state.beta = fast_log_plus_equals(
                        state.beta,
                        self.bestMulti[jnext][i].beta + new_score
                    )
                
                ## 2. Generate P (i, j)
                new_score = evaluate.score_multi(i, j, 
                                        nuc_i, nuc_i_1,
                                        self.nucs[j-1], nuc_j, 
                                        self.seq_length)
                state.beta = fast_log_plus_equals(
                    state.beta,
                    beamstepP[i].beta + new_score
                )
#----------------------------------------------------------

#From LinearPartition.cpp
