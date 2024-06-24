# LinearPartition.py
# Refactored from LinearPartition.cpp and LinearPartition.h
# Original Author: He Zhang
# Modified by: Chaebeom Sheen
# Edited Date: 2024-06-24

#----------------------------------------------------------

#From LinearPartition.h

from typing import Any, Tuple
from collections import defaultdict
import math
import jax
import jax.numpy as jnp 

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

#beginning the BeamCYKParser class

class State:
    def __init__(self, alpha = float('-inf'), beta = float('-inf')):
        self.alpha = alpha
        self.beta = beta

class BeamCKYParser:
    def __init__(self, seq: str, beam_size: int = 100, no_sharp_turn: bool = True, is_verbose: bool = False, 
               bpp_file: str = '', bpp_file_index: str = '',
               float_only: bool = False, bpp_cutoff: float = 0.0, forest_file: str ="", mea_: bool = False, 
               gamma: float = 3.0, mea_file_index: str = "",
               bpseq :bool = False, threshknot_ :bool = False, threshknot_threshold: float = 0.3,
               threshknot_file_index: str = "",
               shape_file_path ="", is_fasta=False, dangle_mode: int = 1) -> None:
        self.seq = seq
        self.beam = beam_size
        self.no_sharp_turn = no_sharp_turn
        self.is_verbose = is_verbose
        self.bpp_file = bpp_file
        self.bpp_file_index = bpp_file_index
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
        self.shape_file_path = shape_file_path
        self.is_fasta = is_fasta
        self.dangle_mode = dangle_mode

        self.seq_length = len(seq)
        self.bestH = {}
        self.bestP = {}
        self.Pij = jnp.zeros((self.seq_length + 1, self.seq_length + 1))
        self.bestM2 = {}
        self.bestMulti = {}
        self.bestM = {}
        self.if_tetraloops = []
        self.if_hexaloops = []
        self.if_triloops = []
        self.bestC = None
        self.nucs = []
        self.SHAPE_data = []
        self.pseudo_energy_stack = []

    def output_to_file(self, output_file: str) -> None:
            if output_file:
                self.output_file = output_file
                print("Outputting base pairing probability matrix to file: ", self.output_file)
                self.writable_file = open(self.outpu_file, "w")
                if not self.writable_file:
                    raise Exception("Error: cannot open file ", output_file)
                
                turn = 3 if self.no_sharp_turn else 0

                for i in range(1, self.seq_length + 1):
                    for j in range(i + turn + 1, self.seq_length + 1):
                        self.writable_file.write(f"{i} {j} {self.Pij[i, j]:.5f}\n")
                
                self.writable_file.write("end of probability matrix \n")
                self.writable_file.close()
                print("Done!")
            else:
                raise ValueError("Error: output file not specified.")
    
    def output_MEA_bpseq(self, output_file:str, pairs: dict[int, int], seq: str) -> None:
        if output_file:
            self.output_file = output_file
            print("Outputting base pairs in bpseq format to {output_file}...")
            self.writable_file = open(self.output_file, "w")
            if not self.writable_file:
                raise Exception(f"Error: cannot open file {self.output_file}")
            for i in range(1, self.seq_length + 1):
                j = pairs [i] if i < len(pairs) else 0
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
    
    def calc_pair_prob(self, viterbi: State) -> None:
        for j in range(self.seq_length):
            item = self.bestP[j]
            i = item[0]
            state = State(item[1])
            temp_prob_inside = state.alpha + state.beta - viterbi.alpha
            if temp_prob_inside > -9.91152:
                prob = fast_exp(temp_prob_inside)
                if prob > 1.0: prob = 1.0
                if prob < self.bpp_cutoff: prob = 0.0
                self.Pij = self.Pij.at[i + 1, j + 1].set(prob)
        
        if self.bpp_file:
            self.output_to_file(self.bpp_file)
        
        elif self.bpp_file_index:
            self.output_to_file(self.bpp_file_index)

    calc_pair_prob_opt = jax.jit(calc_pair_prob)

    def back_trace(self, i: int, j: int, back_pointer: jnp.ndarray) -> str:
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
    
    def get_pairs(self, structure) -> dict[int, int]:
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






