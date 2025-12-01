

import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr
from typing import Tuple, Dict, Optional
from dataset_utils import ECFPDataset


def compute_pairwise_distances(
    X: torch.Tensor,
    metric: str = "euclidean",
    sample_size: Optional[int] = None,
    sample_indices: Optional[torch.Tensor] = None
) -> np.ndarray:
    """
    Compute pairwise distances between all samples.

    Args:
        X: Tensor of shape [N, D] containing N samples
        metric: Distance metric - "euclidean", "cosine", "manhattan", "tanimoto", or "continuous_tanimoto"
        sample_size: If provided, randomly sample this many samples (for efficiency)
        sample_indices: If provided, use these specific indices (overrides sample_size)

    Returns:
        Distance matrix of shape [N, N] (or [sample_size, sample_size])
    """
    if sample_indices is not None:
        # Use provided indices (for consistent sampling across datasets)
        X = X[sample_indices]
    elif sample_size is not None and sample_size < X.shape[0]:
        # Random sample for efficiency
        indices = torch.randperm(X.shape[0])[:sample_size]
        X = X[indices]

    X = X.float()
    N = X.shape[0]

    if metric == "euclidean":
        # Efficient computation: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i^T*x_j
        XX = (X * X).sum(dim=1, keepdim=True)  # [N, 1]
        distances = XX + XX.T - 2 * X @ X.T    # [N, N]
        distances = torch.sqrt(torch.clamp(distances, min=0.0))

    elif metric == "cosine":
        # Cosine distance = 1 - cosine similarity
        X_norm = X / (torch.linalg.norm(X, dim=1, keepdim=True) + 1e-8)
        similarities = X_norm @ X_norm.T
        distances = 1.0 - similarities

    elif metric == "manhattan":
        # L1 distance
        distances = torch.zeros(N, N)
        for i in range(N):
            distances[i] = torch.abs(X - X[i]).sum(dim=1)

    elif metric == "tanimoto":
        # Continuous Tanimoto distance (from Duvenaud et al. 2015)
        # Tanimoto similarity: s(a,b) = (a�b) / (||a||� + ||b||� - a�b)
        # Tanimoto distance: d(a,b) = 1 - s(a,b)

        # Compute dot products: [N, N]
        dot_products = X @ X.T

        # Compute squared norms: [N, 1]
        squared_norms = (X * X).sum(dim=1, keepdim=True)  # [N, 1]

        # Denominator: ||a||� + ||b||� - a�b
        denominators = squared_norms + squared_norms.T - dot_products

        # Tanimoto similarity
        similarities = dot_products / (denominators + 1e-8)

        # Tanimoto distance
        distances = 1.0 - similarities

    elif metric == "continuous_tanimoto":
        # Element-wise min/max Tanimoto from Duvenaud et al. 2015 (Neural Graph Fingerprints)
        # Tanimoto similarity: s(a,b) = sum(min(a,b)) / sum(max(a,b))
        # Tanimoto distance: d(a,b) = 1 - s(a,b)
        # This is the "continuous generalization" used in the NGF paper

        distances = torch.zeros(N, N, device=X.device)
        for i in range(N):
            for j in range(i, N):
                # Element-wise min and max
                min_vals = torch.minimum(X[i], X[j])
                max_vals = torch.maximum(X[i], X[j])

                # Compute similarity
                numerator = min_vals.sum()
                denominator = max_vals.sum() + 1e-8
                similarity = numerator / denominator

                # Convert to distance
                distance = 1.0 - similarity

                # Symmetric matrix
                distances[i, j] = distance
                distances[j, i] = distance

    else:
        raise ValueError(f"Unknown metric: {metric}")

    return distances.cpu().numpy()


def distance_correlation(
    dist1: np.ndarray,
    dist2: np.ndarray,
    method: str = "spearman"
) -> Tuple[float, float]:
    """
    Compute correlation between two distance matrices.

    Args:
        dist1: First distance matrix [N, N]
        dist2: Second distance matrix [N, N]
        method: "spearman" or "pearson"

    Returns:
        (correlation, p_value)
    """
    # Extract upper triangle (excluding diagonal) to avoid redundancy
    N = dist1.shape[0]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)

    vec1 = dist1[mask].flatten()
    vec2 = dist2[mask].flatten()

    if method == "spearman":
        corr, pval = spearmanr(vec1, vec2)
    elif method == "pearson":
        corr, pval = pearsonr(vec1, vec2)
    else:
        raise ValueError(f"Unknown method: {method}")

    return corr, pval


def analyze_distance_preservation(
    original_dataset: ECFPDataset,
    transformed_dataset: ECFPDataset,
    metrics: list = ["tanimoto", "continuous_tanimoto", "euclidean", "cosine"],
    correlation_method: str = "spearman",
    sample_size: int = 500
) -> Dict[str, Dict[str, float]]:
    """
    Analyze how well a transformation preserves pairwise distance structure.

    For each specified metric, this function:
    1. Computes pairwise distances on the ORIGINAL dataset using that metric
    2. Computes pairwise distances on the TRANSFORMED dataset using the SAME metric
    3. Correlates the two distance matrices to measure preservation

    This tells you how well the transformation preserves the distance structure.
    A correlation of 1.0 means perfect preservation, 0.0 means no correlation.

    Args:
        original_dataset: Dataset with original ECFPs
        transformed_dataset: Dataset with transformed ECFPs
        metrics: List of distance metrics to use (same metric applied to both datasets)
                 Options:
                 - "tanimoto": Dot-product based Tanimoto (efficient, good for binary)
                 - "continuous_tanimoto": Min/max based Tanimoto (NGF paper version)
                 - "euclidean": Standard L2 distance
                 - "cosine": Cosine distance
                 - "manhattan": L1 distance
        correlation_method: "spearman" (rank-based) or "pearson" (linear)
        sample_size: Number of samples to use (for efficiency with large datasets)

    Returns:
        Dictionary mapping metric names to correlation results.
        Each entry contains: correlation, p_value, correlation_method

    Example:
        For metric="tanimoto":
        - dist_original[i,j] = tanimoto_distance(original_ecfp[i], original_ecfp[j])
        - dist_transformed[i,j] = tanimoto_distance(transformed_ecfp[i], transformed_ecfp[j])
        - correlation = spearman(dist_original.flatten(), dist_transformed.flatten())
    """
    # Stack all features into tensors
    X_orig = original_dataset.features  # [N, D]
    X_trans = transformed_dataset.features  # [N, D]

    # Generate sample indices ONCE to ensure we compare the same molecules
    sample_indices = None
    if sample_size is not None and sample_size < X_orig.shape[0]:
        sample_indices = torch.randperm(X_orig.shape[0])[:sample_size]
        print(f"Using {sample_size} randomly sampled molecules (same samples for both datasets)")

    results = {}

    for metric in metrics:
        print(f"\nAnalyzing {metric} distance preservation...")

        # Compute pairwise distance matrices using the SAME metric AND SAME samples
        print(f"  Computing {metric} distances on original dataset...")
        dist_orig = compute_pairwise_distances(
            X_orig, metric=metric, sample_indices=sample_indices
        )

        print(f"  Computing {metric} distances on transformed dataset...")
        dist_trans = compute_pairwise_distances(
            X_trans, metric=metric, sample_indices=sample_indices
        )

        # Correlate: how similar are the distance structures?
        corr, pval = distance_correlation(dist_orig, dist_trans, method=correlation_method)

        results[metric] = {
            "correlation": corr,
            "p_value": pval,
            "correlation_method": correlation_method
        }

        print(f"  {correlation_method.capitalize()} correlation: {corr:.4f} (p={pval:.2e})")
        if corr > 0.9:
            print(f"  -> Excellent preservation!")
        elif corr > 0.7:
            print(f"  -> Good preservation")
        elif corr > 0.5:
            print(f"  -> Moderate preservation")
        else:
            print(f"  -> Poor preservation")

    return results


def compute_sparsity_metrics(X: torch.Tensor) -> Dict[str, float]:
    """
    Compute various sparsity and density metrics for a feature matrix.

    Args:
        X: Tensor of shape [N, D]

    Returns:
        Dictionary with sparsity metrics
    """
    X_np = X.cpu().numpy()

    # Proportion of zero/near-zero values
    zero_fraction = (np.abs(X_np) < 1e-6).mean()

    # Mean number of non-zero features per sample
    nnz_per_sample = (np.abs(X_np) > 1e-6).sum(axis=1).mean()

    # Gini coefficient (measure of sparsity)
    # Higher Gini = more sparse
    def gini(x):
        x_sorted = np.sort(np.abs(x.flatten()))
        n = len(x_sorted)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * x_sorted)) / (n * np.sum(x_sorted)) - (n + 1) / n

    gini_coeff = gini(X_np)

    # Coefficient of variation (std/mean)
    cv = X_np.std() / (np.abs(X_np.mean()) + 1e-8)

    # Value distribution stats
    mean_val = X_np.mean()
    std_val = X_np.std()
    min_val = X_np.min()
    max_val = X_np.max()

    return {
        "zero_fraction": zero_fraction,
        "nonzero_per_sample": nnz_per_sample,
        "gini_coefficient": gini_coeff,
        "coefficient_of_variation": cv,
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
    }


def compare_sparsity(
    original_dataset: ECFPDataset,
    transformed_dataset: ECFPDataset
) -> Dict[str, Dict[str, float]]:
    """
    Compare sparsity metrics between original and transformed datasets.

    Returns:
        Dictionary with "original" and "transformed" keys
    """
    print("\n=== Sparsity Analysis ===")

    metrics_orig = compute_sparsity_metrics(original_dataset.features)
    metrics_trans = compute_sparsity_metrics(transformed_dataset.features)

    print("\nOriginal ECFP:")
    for k, v in metrics_orig.items():
        print(f"  {k}: {v:.4f}")

    print("\nTransformed ECFP:")
    for k, v in metrics_trans.items():
        print(f"  {k}: {v:.4f}")

    print("\nChanges:")
    for k in metrics_orig:
        change = metrics_trans[k] - metrics_orig[k]
        pct_change = (change / (abs(metrics_orig[k]) + 1e-8)) * 100
        print(f"  Delta {k}: {change:+.4f} ({pct_change:+.1f}%)")

    return {
        "original": metrics_orig,
        "transformed": metrics_trans
    }


if __name__ == "__main__":
    # Example usage
    from dataset_utils import ECFPDataset
    from transforms import GaussianNoise, L2Normalization, Compose

    print("Loading ESOL dataset...")
    original = ECFPDataset(
        name="esol",
        split_type="random",
        target_index=0,
        n_bits=2048,
        radius=2,
        use_count=False
    )

    # Test with Gaussian noise
    print("\n=== Testing GaussianNoise transformation ===")
    transformed = ECFPDataset(
        name="esol",
        split_type="random",
        target_index=0,
        n_bits=2048,
        radius=2,
        use_count=False
    )

    noise_transform = GaussianNoise(sigma=0.1, seed=42)
    transformed.apply_transform(noise_transform)

    # Distance preservation analysis (including Tanimoto)
    results = analyze_distance_preservation(
        original, transformed,
        metrics=["euclidean", "cosine", "tanimoto"],
        sample_size=300
    )

    # Sparsity comparison
    sparsity_results = compare_sparsity(original, transformed)
