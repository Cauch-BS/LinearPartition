


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

def quickselect_partition(scores: list[(float, int)], lower: int, upper: int) -> int:
    """
    Partitions the scores list using the Quickselect algorithm.
    Returns the index of the pivot, given as scores[upper][0].
    """
    pivot = scores[upper][0]

    while lower < upper:
        while scores[lower][0] < pivot:
            lower += 1
        while scores[upper][0] > pivot:
            upper -= 1
        if scores[lower][0] == scores[upper][0]:
            lower += 1
        elif lower < upper:
            scores[lower], scores[upper] = scores[upper], scores[lower]

    return upper

def quickselect(scores: list[(float, int)], 
                lower: int, 
                upper: int, 
                k: int) -> float:
    """
    Selects the kth smallest element from the given list of scores using the quickselect algorithm.
    
    Args:
        scores (list[(float, int)]): The list of scores, where each score is a tuple of a float value and an integer index.
        lower (int): The lower index of the sublist to consider.
        upper (int): The upper index of the sublist to consider.
        k (int): The index of the desired element in the sorted list.
    
    Returns:
        float: The value of the kth smallest element.
    """
    
    if lower == upper: 
        return scores[lower][0]
    
    split = quickselect_partition(scores, lower, upper)
    
    if k == split: 
        return scores[k][0]
    elif k < split: 
        return quickselect(scores, lower, split - 1, k)
    else: 
        return quickselect(scores, split + 1, upper, k)

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
        self.scores = []

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
                if prob > 1.0: 
                    prob = 1.0
                if prob < self.bpp_cutoff: 
                    prob = 0.0
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
            if k == j: 
                temp = ""
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
                    if i_ > j: 
                        break
                    elif i_ < j: 
                        temp_part_i_j = partition_slices[i_ + 1, j]
                    else: 
                        temp_part_i_j = 0
                    temp_prob = 2 * self.gamma * probs[i, i_] + partition_slices[i + 1][i_ -1] + temp_part_i_j
                    if temp_prob > partition_slices[i][j]:
                        partition_slices[i][j] = temp_prob
                        back_pointer[i][j] = i_
        #back trace to find the secondary structure
        self.structure = self.back_trace_opt(0, self.seq_length - 1, back_pointer)
    #there are multiple if loops in the original code that are not present in the current implementation
    #ifdef lpv is not supported in the current implementation
    #lpv = "Linear Partition Vienna", i.e. Linear Partition with Vienna Implementation
    def outside(self, next_pair: list[int]) -> None:
        evaluate = Evaluate(self.source)
        #initialize the bestC array
        self.bestC[self.seq_length - 1].beta = 0.0

        for j in range(self.seq_length - 1, 0, -1):
            nuc_j = self.nucs[j]
            nuc_j_1 = self.nucs[j + 1] if j + 1 < self.seq_length else -1

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

                                #TODO: Add Suppot for VienanaRNA (#ifdef lpv)

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
    def beam_prune(
            self,
            beamstep: dict[int, State], 
    ):
        self.scores.clear()

        for i, state in beamstep.items():
            k = i - 1
            new_score = self.bestC[k].alpha if k >= 0 else 0 
            new_alpha = state.alpha + new_score
            self.scores.append((new_alpha, i))
        
        if len(self.scores) <= self.beam: 
            return float('-inf')

        threshold = quickselect(self.scores, 0, len(self.scores) - 1, 
                                len(self.scores) - self.beam
                                )
        
        for p in self.scores:
            if p[0] < threshold:
                del beamstep[p[1]]
        
        return threshold
    
    def parse(self) -> float:
        evaluate = Evaluate(self.source)
        next_pair = [[]]
        for nuc_i in range(N_TYPE_NUC):
            #initialize next pair
            next_pair[nuc_i].extend([-1] * self.seq_length)
            next = -1
            for j in range(self.seq_length - 1, -1, -1):
                next_pair[nuc_i][j] = next
                if evaluate: 
                    pass


        
    