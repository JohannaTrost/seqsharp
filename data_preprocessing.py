from Bio import SeqIO
import numpy as np
import time
import itertools

""" amino acids 
alanine       : A |  glutamine     : Q | leucine       : L | serine    : S |
arginine      : R |  glutamic acid : E | lysine        : K | threonine : T |
asparagine    : N |  glycine       : G | methionine    : M | tryptophan: W |
aspartic acid : D |  histidine     : H | phenylalanine : F | tyrosine  : Y |
cysteine      : C |  isoleucine    : I | proline       : P | valine    : V |
unknown : X 
gap : - 
"""
amino_acids = np.array(
    ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X', '-'])
aligned_seq_records = []  # aligned sequences including meta-information
seqs = []  # one-hot encoded sequences
for seq_record in SeqIO.parse("align.pl", "fasta"):
    aligned_seq_records.append(seq_record)
    seqs.append([(amino_acids == aa).astype(int) for aa in seq_record.seq])
seqs = np.array(seqs)

# ms_start = time.time()
nb_seqs = len(seqs)
seq_pairs_diff = []
seq_pairs_sum = []
aa_prop_no_pair = []  # amino acid proportions at each position for all sequences excluding the pair
sum_all_seqs = np.sum(seqs, axis=0)
for i in range(nb_seqs):
    for j in range(i + 1, nb_seqs):  # loops for pairs
        sums = seqs[i, :, :] + seqs[j, :, :]
        aa_prop_no_pair.append((sum_all_seqs - sums) / nb_seqs)
        seq_pairs_diff.append(seqs[i, :, :] - seqs[j, :, :])
        seq_pairs_sum.append(sums)
# print(time.time()-ms_start)

"""
Vectorized Solution (about 10% slower)
ms_start = time.time()
nb_seqs = len(seqs)
ij = np.asarray(list(itertools.combinations(range(seqs.shape[0]), 2)))
aa_prop_no_pair = seqs.sum(axis=0)[np.newaxis, :, :].repeat(ij.shape[0], 0)
p1 = seqs[ij[:, 0], :, :]
p2 = seqs[ij[:, 1], :, :]
seq_pairs_sum = p1 + p2
aa_prop_no_pair -= seq_pairs_sum
aa_prop_no_pair = aa_prop_no_pair / nb_seqs
seq_pairs_diff = p1 - p2
print(time.time()-ms_start)
"""

# # # # # little test # # # # #
print("\n************************************* MINI DEMO *************************************\n\n"
      "Seq 1 : {}\n"
      "Seq 17 : {}\n"
      "Seq 1 last amino acid :   {} -> {}\n"
      "Seq 17 last amino acid :   {} -> {}\n"
      "Amino acid order : {}\n"
      "Difference (seq1,seq17) (last amino acid) :   {}\n"
      "Sum (seq1,seq17) (last amino acid) :   {}\n\n"
      "Shape of representations :\n"
      "\tNb of pairs: {} -> ({}^2-{}) / 2,\n"
      "\t(Aligned) seq length: {},\n "
      "\tNb of aa (plus unknown, gaps): {}\n\n"
      "Proportions of amino acids of remaining seqs at every position (for pair (seq1,seq17)) : \n{}\n"
      "Number of amino acids of remaining seqs at every position (for pair (seq1,seq17)) :   \n{}\n"
      "Number of amino acids of all seq (to compare) : \n{}"
      .format(repr(aligned_seq_records[0].seq), repr(aligned_seq_records[16].seq), aligned_seq_records[0][-1],
              seqs[0][-1], aligned_seq_records[16][-1], seqs[16][-1], list(amino_acids), seq_pairs_diff[15][-1],
              seq_pairs_sum[15][-1], np.asarray(seq_pairs_diff).shape[0], nb_seqs, nb_seqs,
              np.asarray(seq_pairs_diff).shape[1], np.asarray(seq_pairs_diff).shape[2], np.asarray(aa_prop_no_pair[15]),
              np.asarray(aa_prop_no_pair[15])*nb_seqs, sum_all_seqs))
