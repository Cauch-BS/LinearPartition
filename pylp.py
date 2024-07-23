import numpy as np
from collections import namedtuple

NOTON = 5
NOTOND = 25
NOTONT = 125
_allowed_pairs = None # Initialized in initialize()

# LinearPartition.h:20
kT = 61.63207755
NEG_INF = -2e20

# LinearPartition.h:28
pf_type = np.float32

# energy_parameter.h:11
VIE_INF = 10000000

# LinearPartition.h:35
value_type = np.int64
VALUE_MIN = np.finfo(np.float64).min

# LinearPartition.h:67
State = np.dtype([
    ('alpha', np.float64),
    ('beta', np.float64),
])

# energy_parameter.h:29
Triloops = (
    "CAACG "
    "GUUAC "
).split()
Triloop37 = (680, 690)

# energy_parameter.h:35
Tetraloops = (
    "CAACGG "
    "CCAAGG "
    "CCACGG "
    "CCCAGG "
    "CCGAGG "
    "CCGCGG "
    "CCUAGG "
    "CCUCGG "
    "CUAAGG "
    "CUACGG "
    "CUCAGG "
    "CUCCGG "
    "CUGCGG "
    "CUUAGG "
    "CUUCGG "
    "CUUUGG "
).split()
Tetraloo37 = (550, 330, 370, 340, 350, 360, 370, 250, 360, 280, 370, 270, 280, 350, 370, 370)

# energy_parameter.h:56
Hexaloops = (
    "ACAGUACU "
    "ACAGUGAU "
    "ACAGUGCU "
    "ACAGUGUU "
).split()
Hexaloop37 = (280, 360, 290, 180)

# energy_parameter.h:75
hairpin37 = (
    VIE_INF, VIE_INF, VIE_INF, 540, 560, 570, 540, 600, 550, 640,
    650, 660, 670, 680, 690, 690, 700, 710, 710, 720, 720, 730,
    730, 740, 740, 750, 750, 750, 760, 760, 770)

mismatchH37 = [
[[   VIE_INF,   VIE_INF,   VIE_INF,   VIE_INF,   VIE_INF],
 [   VIE_INF,   VIE_INF,   VIE_INF,   VIE_INF,   VIE_INF],
 [   VIE_INF,   VIE_INF,   VIE_INF,   VIE_INF,   VIE_INF],
 [   VIE_INF,   VIE_INF,   VIE_INF,   VIE_INF,   VIE_INF],
 [   VIE_INF,   VIE_INF,   VIE_INF,   VIE_INF,   VIE_INF]],
# CG..
[[   -80,  -100,  -110,  -100,   -80],
 [  -140,  -150,  -150,  -140,  -150],
 [   -80,  -100,  -110,  -100,   -80],
 [  -150,  -230,  -150,  -240,  -150],
 [  -100,  -100,  -140,  -100,  -210]],
# GC..
[[   -50,  -110,   -70,  -110,   -50],
 [  -110,  -110,  -150,  -130,  -150],
 [   -50,  -110,   -70,  -110,   -50],
 [  -150,  -250,  -150,  -220,  -150],
 [  -100,  -110,  -100,  -110,  -160]],
[[    20,    20,   -20,   -10,   -20],
 [    20,    20,   -50,   -30,   -50],
 [   -10,   -10,   -20,   -10,   -20],
 [   -50,  -100,   -50,  -110,   -50],
 [   -10,   -10,   -30,   -10,  -100]],
[[     0,   -20,   -10,   -20,     0],
 [   -30,   -50,   -30,   -60,   -30],
 [     0,   -20,   -10,   -20,     0],
 [   -30,   -90,   -30,  -110,   -30],
 [   -10,   -20,   -10,   -20,   -90]],
[[   -10,   -10,   -20,   -10,   -20],
 [   -30,   -30,   -50,   -30,   -50],
 [   -10,   -10,   -20,   -10,   -20],
 [   -50,  -120,   -50,  -110,   -50],
 [   -10,   -10,   -30,   -10,  -120]],
[[     0,   -20,   -10,   -20,     0],
 [   -30,   -50,   -30,   -50,   -30],
 [     0,   -20,   -10,   -20,     0],
 [   -30,  -150,   -30,  -150,   -30],
 [   -10,   -20,   -10,   -20,   -90]],
[[    20,    20,   -10,   -10,     0],
 [    20,    20,   -30,   -30,   -30],
 [     0,   -10,   -10,   -10,     0],
 [   -30,   -90,   -30,  -110,   -30],
 [   -10,   -10,   -10,   -10,   -90]]]

def GET_ACGU_NUM(c):
    return {'A': 0, 'C': 1, 'G': 2, 'U': 3}.get(c, 4)

# utility_v.h:12
def num_to_nuc(x):
    if x == -1:
        return -1
    else:
        return 0 if x == 4 else (x + 1)

# utility_v.h:13
def num_to_pair(x, y):
    if x == 0:
        return 5 if y == 3 else 0
    elif x == 1:
        return 1 if y == 2 else 0
    elif x == 2:
        if y == 1:
            return 2
        elif y == 3:
            return 3
        else:
            return 0
    elif x == 3:
        if y == 2:
            return 4
        elif y == 0:
            return 6
        else:
            return 0
    else:
        return 0

# utility_v.h:14
def nuc_to_pair(x, y):
    if x == 1:
        return 5 if y == 4 else 0
    elif x == 2:
        return 1 if y == 3 else 0
    elif x == 3:
        if y == 2:
            return 2
        elif y == 4:
            return 3
        else:
            return 0
    elif x == 4:
        if y == 3:
            return 4
        elif y == 1:
            return 6
        else:
            return 0
    else:
        return 0

def initialize():
    global _allowed_pairs

    _allowed_pairs = np.zeros((NOTON, NOTON), dtype=bool)
    _allowed_pairs[GET_ACGU_NUM('A'), GET_ACGU_NUM('U')] = True
    _allowed_pairs[GET_ACGU_NUM('U'), GET_ACGU_NUM('A')] = True
    _allowed_pairs[GET_ACGU_NUM('C'), GET_ACGU_NUM('G')] = True
    _allowed_pairs[GET_ACGU_NUM('G'), GET_ACGU_NUM('C')] = True
    _allowed_pairs[GET_ACGU_NUM('G'), GET_ACGU_NUM('U')] = True
    _allowed_pairs[GET_ACGU_NUM('U'), GET_ACGU_NUM('G')] = True

# utility_v.h:42
def v_init_tetra_hex_tri(seq, seq_length):
    global if_tetraloops, if_triloops, if_hexaloops

    if_tetraloops = -np.ones(max(0, seq_length - 5), dtype=np.int32)
    if_triloops = -np.ones(max(0, seq_length - 4), dtype=np.int32)
    if_hexaloops = -np.ones(max(0, seq_length - 7), dtype=np.int32)

    # TetraLoops
    for i in range(seq_length - 5):
        if not (seq[i] == 'C' and seq[i + 5] == 'G'):
            continue
        hexseq = seq[i:i + 6]
        if hexseq in Tetraloops:
            if_tetraloops[i] = Tetraloops.index(hexseq)

    # Triloops
    for i in range(seq_length - 4):
        if not ((seq[i] == 'C' and seq[i + 4] == 'G') or (seq[i] == 'G' and seq[i + 4] == 'C')):
            continue
        pentaseq = seq[i:i + 5]
        if pentaseq in Triloops:
            if_triloops[i] = Triloops.index(pentaseq)

    # Hexaloops
    for i in range(seq_length - 7):
        if not (seq[i] == 'A' and seq[i + 7] == 'U'):
            continue
        octaseq = seq[i:i + 8]
        if octaseq in Hexaloops:
            if_hexaloops[i] = Hexaloops.index(octaseq)

    return if_tetraloops, if_triloops, if_hexaloops

# utility_v.h:76
def v_score_hairpin(i, j, nuci, nuci1, nucj_1, nucj, tetra_hex_tri_index=-1):
    size = j - i - 1
    type = num_to_pair(nuci, nucj)
    si1 = num_to_nuc(nuci1)
    sj1 = num_to_nuc(nucj_1)

    if size <= 30:
        energy = hairpin37[size]
    else:
        energy = hairpin37[30] + int(lxc37 * math.log(size / 30.0))

    if size < 3:
        return energy  # should only be the case when folding alignments

    if size == 4 and tetra_hex_tri_index > -1:
        return Tetraloop37[tetra_hex_tri_index]
    elif size == 6 and tetra_hex_tri_index > -1:
        return Hexaloop37[tetra_hex_tri_index]
    elif size == 3:
        if (tetra_hex_tri_index > -1):
            return Triloop37[tetra_hex_tri_index]
        return energy + (TerminalAU37 if type > 2 else 0)

    energy += mismatchH37[type][si1][sj1]

    return energy


# LinearPartition.h:176
def Fast_LogExpPlusOne(x):
    # Define the boundaries and polynomial coefficients
    boundaries = [0.6615367791, 1.6320158198, 2.4912588184, 3.3792499610,
                  4.4261691294, 5.7890710412, 7.8162726752, 11.8624794162]
    coefficients = [
        (-0.0065591595, 0.1276442762, 0.4996554598, 0.6931542306),
        (-0.0155157557, 0.1446775699, 0.4882939746, 0.6958092989),
        (-0.0128909247, 0.1301028251, 0.5150398748, 0.6795585882),
        (-0.0072142647, 0.0877540853, 0.6208708362, 0.5909675829),
        (-0.0031455354, 0.0467229449, 0.7592532310, 0.4348794399),
        (-0.0010110698, 0.0185943421, 0.8831730747, 0.2523695427),
        (-0.0001962780, 0.0046084408, 0.9634431978, 0.0983148903),
        (-0.0000113994, 0.0003734731, 0.9959107193, 0.0149855051)
    ]

    assert 0.0 <= x <= 11.8624794162, "Argument out-of-range."

    # Determine which set of coefficients to use based on the value of x
    if x < boundaries[3]:
        if x < boundaries[1]:
            if x < boundaries[0]:
                coeff = coefficients[0]
            else:
                coeff = coefficients[1]
        else:
            if x < boundaries[2]:
                coeff = coefficients[2]
            else:
                coeff = coefficients[3]
    else:
        if x < boundaries[5]:
            if x < boundaries[4]:
                coeff = coefficients[4]
            else:
                coeff = coefficients[5]
        else:
            if x < boundaries[6]:
                coeff = coefficients[6]
            else:
                coeff = coefficients[7]

    # Calculate the polynomial
    return ((coeff[0] * x + coeff[1]) * x + coeff[2]) * x + coeff[3]

# LinearPartition.h:213
def Fast_LogPlusEquals(x, y):
    if x < y:
        x, y = y, x
    if y > NEG_INF / 2 and x - y < 11.8624794162:
        x = Fast_LogExpPlusOne(x - y) + y
    return x

def parse(seq):
    # LinearPartition.cpp:590
    beam = 100
    no_sharp_turn = True


    # LinearPartition.cpp:73
    seq_length = len(seq)

    # LinearPartition.cpp:76
    bestC = np.full(seq_length, VALUE_MIN, dtype=State)
    bestH = [{} for i in range(seq_length)]
    bestP = [{} for i in range(seq_length)]
    bestM = [{} for i in range(seq_length)]
    bestM2 = [{} for i in range(seq_length)]
    bestMulti = [{} for i in range(seq_length)]

    scores = [[pf_type(), int()] for i in range(seq_length)]

    # LinearPartition.cpp:106
    nucs = [GET_ACGU_NUM(seq[i]) for i in range(seq_length)]

    next_pair = [[-1 for _ in range(seq_length)] for _ in range(NOTON)]
    for nuci in range(NOTON):
        next = -1
        for j in range(seq_length-1, -1, -1):
            next_pair[nuci][j] = next
            if _allowed_pairs[nuci, nucs[j]]:
                next = j

    v_init_tetra_hex_tri(seq, seq_length)

    if seq_length > 0:
        bestC[0]['alpha'] = 0.0
    if seq_length > 1:
        bestC[1]['alpha'] = 0.0

    # LinearPartition.cpp:136
    newscore = 0
    for j in range(seq_length):
        nucj = nucs[j]
        nucj1 = nucs[j + 1] if (j + 1) < seq_length else -1

        beamstepH = bestH[j]
        beamstepMulti = bestMulti[j]
        beamstepP = bestP[j]
        beamstepM2 = bestM2[j]
        beamstepM = bestM[j]
        beamstepC = bestC[j]

        # beam of H
        if beam > 0 and len(beamstepH) > beam:
            beam_prune(beamstepH)

        # for nucj put H(j, j_next) into H[j_next]
        jnext = next_pair[nucj][j]
        if no_sharp_turn:
            while jnext - j < 4 and jnext != -1:
                jnext = next_pair[nucj][jnext]
        if jnext != -1:
            nucjnext = nucs[jnext]
            nucjnext_1 = nucs[jnext - 1] if (jnext - 1) > -1 else -1
            tetra_hex_tri = -1
            if jnext - j - 1 == 4:
                tetra_hex_tri = if_tetraloops[j]
            elif jnext - j - 1 == 6:
                tetra_hex_tri = if_hexaloops[j]
            elif jnext - j - 1 == 3:
                tetra_hex_tri = if_triloops[j]
            newscore = -v_score_hairpin(j, jnext, nucj, nucj1, nucjnext_1, nucjnext, tetra_hex_tri)
            bestH[jnext][j]['alpha'] = Fast_LogPlusEquals(bestH[jnext][j]['alpha'], newscore / kT)

        # for every state h in H[j]
        # 1. extend h(i, j) to h(i, jnext)
        # 2. generate p(i, j)
        for i, state in beamstepH.items():
            nuci = nucs[i]
            jnext = next_pair[nuci][j]
            if jnext != -1:
                nuci1 = nucs[i + 1] if (i + 1) < seq_length else -1
                nucjnext = nucs[jnext]
                nucjnext_1 = nucs[jnext - 1] if (jnext - 1) > -1 else -1

                # 1. extend h(i, j) to h(i, jnext)
                tetra_hex_tri = -1
                if jnext - i - 1 == 4:
                    tetra_hex_tri = if_tetraloops[i]
                elif jnext - i - 1 == 6:
                    tetra_hex_tri = if_hexaloops[i]
                elif jnext - i - 1 == 3:
                    tetra_hex_tri = if_triloops[i]
                newscore = -v_score_hairpin(i, jnext, nuci, nuci1, nucjnext_1, nucjnext, tetra_hex_tri)
                bestH[jnext][i]['alpha'] = Fast_LogPlusEquals(bestH[jnext][i]['alpha'], newscore / kT)

            # 2. generate p(i, j)
            beamstepP[i]['alpha'] = Fast_LogPlusEquals(beamstepP[i]['alpha'], state['alpha'])

        if j == 0:
            continue

        # beam of Multi
        if beam > 0 and len(beamstepMulti) > beam:
            beam_prune(beamstepMulti)

        for i, state in beamstepMulti.items():
            nuci = nucs[i]
            nuci1 = nucs[i + 1]
            jnext = next_pair[nuci][j]

            # 1. extend (i, j) to (i, jnext)
            if jnext != -1:
                bestMulti[jnext][i]['alpha'] = Fast_LogPlusEquals(bestMulti[jnext][i]['alpha'], state['alpha'])

            # 2. generate P (i, j)
            newscore = -v_score_multi(i, j, nuci, nuci1, nucs[j - 1], nucj, seq_length, dangle_mode)
            beamstepP[i]['alpha'] = Fast_LogPlusEquals(beamstepP[i]['alpha'], state['alpha'] + newscore / kT)

        # beam of P
        if beam > 0 and len(beamstepP) > beam:
            beam_prune(beamstepP)

        for i, state in beamstepP.items():
            nuci = nucs[i]
            nuci_1 = nucs[i - 1] if i - 1 > -1 else -1

            # 1. generate new helix / single_branch
            if i > 0 and j < seq_length - 1:
                for p in range(i - 1, max(i - SINGLE_MAX_LEN, 0) - 1, -1):
                    nucp = nucs[p]
                    nucp1 = nucs[p + 1]
                    q = next_pair[nucp][j]
                    while q != -1 and ((i - p) + (q - j) - 2 <= SINGLE_MAX_LEN):
                        nucq = nucs[q]
                        nucq_1 = nucs[q - 1]

                        if p == i - 1 and q == j + 1:
                            # helix
                            newscore = -v_score_single(p, q, i, j, nucp, nucp1, nucq_1, nucq, nuci_1, nuci, nucj, nucj1)
                            if use_shape:
                                newscore += -(pseudo_energy_stack[p] + pseudo_energy_stack[i] + pseudo_energy_stack[j] + pseudo_energy_stack[q])
                            bestP[q][p]['alpha'] = Fast_LogPlusEquals(bestP[q][p]['alpha'], state['alpha'] + newscore / kT)
                        else:
                            # single branch
                            newscore = -v_score_single(p, q, i, j, nucp, nucp1, nucq_1, nucq, nuci_1, nuci, nucj, nucj1)
                            bestP[q][p]['alpha'] = Fast_LogPlusEquals(bestP[q][p]['alpha'], state['alpha'] + newscore / kT)
                        q = next_pair[nucp][q]

            # 2. M = P
            if i > 0 and j < seq_length - 1:
                newscore = -v_score_M1(i, j, j, nuci_1, nuci, nucj, nucj1, seq_length, dangle_mode)
                beamstepM[i]['alpha'] = Fast_LogPlusEquals(beamstepM[i]['alpha'], state['alpha'] + newscore / kT)

            # 3. M2 = M + P
            k = i - 1
            if k > 0 and bestM[k]:
                newscore = -v_score_M1(i, j, j, nuci_1, nuci, nucj, nucj1, seq_length, dangle_mode)
                m1_alpha = state.alpha + newscore / kT
                for newi, m_state in bestM[k].items():
                    beamstepM2[newi]['alpha'] = Fast_LogPlusEquals(beamstepM2[newi]['alpha'], m_state['alpha'] + m1_alpha)

            # 4. C = C + P
            k = i - 1
            if k >= 0:
                prefix_C = bestC[k]
                nuck = nuci_1
                nuck1 = nuci
                newscore = -v_score_external_paired(k + 1, j, nuck, nuck1, nucj, nucj1, seq_length, dangle_mode)
                beamstepC['alpha'] = Fast_LogPlusEquals(beamstepC['alpha'], prefix_C['alpha'] + state['alpha'] + newscore / kT)
            else:
                newscore = -v_score_external_paired(0, j, -1, nucs[0], nucj, nucj1, seq_length, dangle_mode)
                beamstepC['alpha'] = Fast_LogPlusEquals(beamstepC['alpha'], state['alpha'] + newscore / kT)

        # beam of M2
        if beam > 0 and len(beamstepM2) > beam:
            beam_prune(beamstepM2)

        for i, state in beamstepM2.items():
            # 1. multi-loop
            for p in range(i - 1, max(i - SINGLE_MAX_LEN, 0) - 1, -1):
                nucp = nucs[p]
                q = next_pair[nucp][j]
                if q != -1 and ((i - p - 1) <= SINGLE_MAX_LEN):
                    bestMulti[q][p]['alpha'] = Fast_LogPlusEquals(bestMulti[q][p]['alpha'], state['alpha'])

            # 2. M = M2
            beamstepM[i]['alpha'] = Fast_LogPlusEquals(beamstepM[i]['alpha'], state['alpha'])

        # beam of M
        if beam > 0 and len(beamstepM) > beam:
            beam_prune(beamstepM)

        for i, state in beamstepM.items():
            if j < seq_length - 1:
                bestM[j + 1][i]['alpha'] = Fast_LogPlusEquals(bestM[j + 1][i]['alpha'], state['alpha'])

        # beam of C
        if j < seq_length - 1:
            bestC[j + 1]['alpha'] = Fast_LogPlusEquals(bestC[j + 1]['alpha'], beamstepC['alpha'])

    return next_pair

if __name__ == '__main__':
    initialize()
    print(parse('AAGCCCCGCUUU'))
