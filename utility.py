import random
import numpy as np
import copy
import warnings
from Bio import SeqIO


def fetch_sequences_from_fasta(path, length):
    promoter_seg = set()
    for seq_record in SeqIO.parse(path, "fasta"):
        sequence = seq_record.seq
        l = len(sequence)
        if (l == length):
            promoter_seg.add(sequence)
        else:
            warnings.warn(f"The sequence whose identifier(ID) is {seq_record.id} has skipped since its length is {l} that is not equal to {length}.")
    return np.array(list(promoter_seg))

def construct_non_promoter_sequences(pos_seg, n_sub=20, n_subst=12):
    if type(pos_seg) is not np.ndarray:
        pos_seg = np.array(pos_seg)
    shape = pos_seg.shape
    length = shape[1] / n_sub 
    if length.is_integer():
        subs = []
        length = int(length)
        p_seg = copy.deepcopy(pos_seg)
        for n, seg in enumerate(p_seg):
            for i in range(0, shape[1], length):
                sub = seg[i:i+length]
                subs.append(sub)
            rand = random.sample(range(n*n_sub, n*n_sub+n_sub), n_subst)
            for r in rand:
                subs[r] = random.choices(['A', 'G', 'T', 'C'] , k=length)
    else:
        raise Exception(f'The length of DNA sequence is {shape[1]}. It cannot divide into {n_sub} subsequence')
    return np.array(subs).reshape((*shape))

def onehot_dna_sequences(sequences):
    mapping = {"A": 0, 'a': 0, "C": 1, 'c': 1, "G": 2, 'g': 2, "T": 3, 't': 3}
    input_features = []
    for sequence in sequences:
        try:
            mapped_bases = [mapping[base] for base in sequence]
            input_features.append(np.eye(4)[mapped_bases])
        except:
            warnings.warn(f'A sequence is skipped since it contains a base that is not among valid bases [A, C, G, T]')
    return np.array(input_features)