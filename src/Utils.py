import numpy as np

# provides feature functions for Linear partition
# Original Author: Kai Zhao, Dezhong Deng
# Modified By: Chaebeom Sheen
# Updated: 2024-06-22


ACGU_NUM = list(enumerate('ACGU'))
GET_ACGU_NUM = ACGU_NUM.__getitem__

TETRALOOPS = ["CAACGG", "CCAAGG", "CCACGG", "CCCAGG", "CCGAGG",
              "CCGCGG", "CCUAGG", "CCUCGG", "CUAAGG", "CUACGG",
              "CUCAGG", "CUCCGG", "CUGCGG", "CUUAGG", "CUUCGG",
              "CUUUGG"]
HEXALOOPS = ["ACAGUACU", "ACAGUGAU", "ACAGUGCU", "ACAGUGUU"]
TRILOOPS = ["CAACG", "GUUAC"]

def find_special_hairpins(seq):
    if_tetraloops = np.full(max(len(seq) - 5, 0), -1, dtype=np.int32)
    if_hexaloops = np.full(max(len(seq) - 7, 0), -1, dtype=np.int32)
    if_triloops = np.full(max(len(seq) - 7, 0), -1, dtype=np.int32)

    for i in range(len(seq) - 5):
        if seq[i] == 'C' and seq[i + 5] == 'G':
            ts = seq[i:i + 6]
            if ts in TETRALOOPS:
                if_tetraloops[i] = TETRALOOPS.index(ts)
    
    for i in range(len(seq) - 7):
        if seq[i] == 'A' and seq[i + 7] == 'U':
            ts = seq[i:i + 8]
            if ts in HEXALOOPS:
                if_hexaloops[i] = HEXALOOPS.index(ts)
    
    for i in range(len(seq) - 4):
        if seq[i] == 'C' and seq[i + 4] == 'G':
            ts = seq[i:i + 5]
            if ts in TRILOOPS:
                if_triloops[i] = TRILOOPS.index(ts)

    return if_tetraloops, if_hexaloops, if_triloops
