import numpy as np

from hierarchical_EM import multi_dens


def expected_aa_freqs(data, profs, pro_w, cl_w=None):
    # \sum_cl P(aln | cl)  \sum_prof weight_prof \times prof
    # For iEM,  \sum_prof weight_prof \times prof

    n_alns, n_clusters, n_aas = len(data), len(cl_w), 20
    ax_s = 1

    weighted_profs = pro_w @ profs

    log_aln_cl = np.zeros((n_alns, n_clusters))

    p_sites_profs = multi_dens(data, profs)
    for aln in range(n_alns):
        # -------- lk on alignment level
        log_aln_cl[aln] = np.sum(np.log(np.dot(pro_w, p_sites_profs[aln].T)),
                                 axis=ax_s)

    c = np.abs(np.max(log_aln_cl)) # to scale log TODO check if overall max or on !cl! axis
    log_aln_cl_scaled = log_aln_cl + c
    lk_aln_cl = np.exp(log_aln_cl_scaled)

    lk_aln_cl_rep = np.repeat(lk_aln_cl[:, :, np.newaxis],
                               n_aas, axis=2) # add aa axis

    expct_aa_freqs = np.zeros((n_alns, n_aas))
    for aln in range(n_alns):
        expct_aa_freqs[aln] = np.sum(lk_aln_cl_rep[aln] * weighted_profs, axis=0)

    log_expct_aa_freqs = np.log(expct_aa_freqs) - c

    expct_aa_freqs = expct_aa_freqs / np.repeat(np.sum(expct_aa_freqs, axis=1)[:, np.newaxis], n_aas, axis=1)
    # expct_aa_freqs = np.exp(log_expct_aa_freqs)

    return expct_aa_freqs
