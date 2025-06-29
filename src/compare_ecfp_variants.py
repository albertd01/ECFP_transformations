import numpy as np
from rdkit import Chem
from utils.dataset_utils import DuvenaudDataset
from utils.ecfp_utils import compute_ecfp_bit_vectors, compute_algorithm1_fps, compute_ecfp_count_vectors
from itertools import combinations
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def cont_tanimoto_minmax(x, y):
    num = np.sum(np.minimum(x, y))
    denom = np.sum(np.maximum(x, y)) + 1e-8
    return 1.0 - (num / denom)

def plot_pairwise_distances(ecfp_sampled, ngf_sampled, r, algo1 = "", algo2 = ""):
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
    plt.xlabel(algo2 +" distances")
    plt.ylabel(algo1 +" distances")
    plt.xlim(0.5, 1.0)
    plt.ylim(0.5, 1.0)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.title(f"{algo1} vs. {algo2}\n$r={r:.3f}$", fontsize=14)
    plt.tight_layout()
    plt.show()

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

dataset_name = "ESOL" 
dataset = DuvenaudDataset(dataset_name)
dataset.process()
smiles_list = dataset.smiles_list

# Compute fingerprints for all implementations
fps_dict = {
    "rdkit_binary": compute_ecfp_bit_vectors(smiles_list, radius=2, nBits=2048),
    "rdkit_count": compute_ecfp_count_vectors(smiles_list, radius=2, nBits=2048),
    "algorithm1": compute_algorithm1_fps(smiles_list, radius=2, nBits=2048)
}

implementations = list(fps_dict.keys())
for i, impl_a in enumerate(implementations):
    for j, impl_b in enumerate(implementations):
        if i >= j:
            continue
        impl_a_dists_all, impl_b_dists_all, impl_a_dists_sampled, impl_b_dists_sampled, r = run_pairwise_analysis(fps_dict[impl_a], fps_dict[impl_b])
        plot_pairwise_distances(impl_a_dists_sampled, impl_b_dists_sampled, r, implementations[i], implementations[j])


