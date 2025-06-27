import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import pearsonr
from rdkit import DataStructs
import seaborn as sns


def cont_tanimoto_minmax(x, y):
    num = np.sum(np.minimum(x, y))
    denom = np.sum(np.maximum(x, y)) + 1e-8
    return 1.0 - (num / denom)


def run_pairwise_analysis(ngf_embeddings, ecfp_fps, sample_size=2000, seed=42):
    """
    Compute pairwise distances between all pairs of NGF and ECFP representations.
    Use all distances to compute Pearson r, but only sample a subset for plotting.

    Args:
        ngf_embeddings (np.ndarray): shape [N, D]
        ecfp_fps (List[ExplicitBitVect]): RDKit bit vectors
        sample_size (int): number of pairs to sample for plotting
        seed (int): RNG seed for reproducibility

    Returns:
        ecfp_dists_all, ngf_dists_all (np.ndarray): distances over all pairs
        ecfp_dists_sampled, ngf_dists_sampled (np.ndarray): sampled distances for plotting
        r (float): Pearson correlation over all pairs
    """
    N = len(ngf_embeddings)
    all_pairs = list(combinations(range(N), 2))

    # Compute all distances
    ecfp_dists_all = np.empty(len(all_pairs))
    ngf_dists_all = np.empty(len(all_pairs))

    for idx, (i, j) in enumerate(all_pairs):
        ecfp_dists_all[idx] = cont_tanimoto_minmax(
            np.array(ecfp_fps[i], dtype=np.float32),
            np.array(ecfp_fps[j], dtype=np.float32)
        )
        ngf_dists_all[idx] = cont_tanimoto_minmax(ngf_embeddings[i], ngf_embeddings[j])

    # Compute Pearson r over all pairs
    r, _ = pearsonr(ecfp_dists_all, ngf_dists_all)

    # Sample subset for plotting
    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(len(all_pairs), size=min(sample_size, len(all_pairs)), replace=False)
    ecfp_dists_sampled = ecfp_dists_all[sampled_indices]
    ngf_dists_sampled = ngf_dists_all[sampled_indices]

    return ecfp_dists_all, ngf_dists_all, ecfp_dists_sampled, ngf_dists_sampled, r



def plot_pairwise_distances(ecfp_sampled, ngf_sampled, r, title='NGF vs ECFP Distances'):
    plt.figure(figsize=(5, 5))
    plt.scatter(
        ecfp_sampled,
        ngf_sampled,
        s=20,
        alpha=0.4,
        edgecolors='black',
        linewidths=0.2,
        facecolor='C0'
    )
    plt.xlabel("Circular fingerprint distances")
    plt.ylabel("Neural fingerprint distances")
    plt.xlim(0.5, 1.0)
    plt.ylim(0.5, 1.0)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.title(f"{title}\n$r={r:.3f}$", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_distance_distribution(ecfp_all, ngf_all):
    plt.figure(figsize=(6,4))
    plt.hist(ecfp_all,  bins=50, alpha=0.6, label='ECFP')
    plt.hist(ngf_all, bins=50, alpha=0.6, label='NGF')
    plt.xlabel("Min/max Tanimoto distance")
    plt.ylabel("Count")
    plt.title("Distance distributions ECFP vs. NGFs")
    plt.legend()
    plt.tight_layout()
    plt.show()
    

def plot_bit_usage_comparison(fp_ecfp: np.ndarray, emb_ngf: np.ndarray):
    # Sum activations per bit/dimension across all molecules
    bit_usage_ecfp = np.sum(fp_ecfp, axis=0)
    bit_usage_ngf  = np.sum(emb_ngf, axis=0)
    D = bit_usage_ecfp.shape[0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), sharey=False)
    
    ax1.bar(np.arange(D), bit_usage_ecfp, width=1.0)
    ax1.set_title("ECFP Bit Usage")
    ax1.set_xlabel("Bit Index")
    ax1.set_ylabel("Activation Count (sum over molecules)")
    
    ax2.bar(np.arange(D), bit_usage_ngf, width=1.0)
    ax2.set_title("NGF Embedding Usage")
    ax2.set_xlabel("Dimension Index")
    ax2.set_ylabel("Activation Sum (continuous)")
    
    plt.tight_layout()
    plt.show()


    