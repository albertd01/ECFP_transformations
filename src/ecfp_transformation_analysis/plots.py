import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict
import seaborn as sns

def norms(Z): return np.linalg.norm(Z, axis=1)
def cos_self(X, Xp): return (X*Xp).sum(1) / (norms(X)*norms(Xp) + 1e-9)


def plot_norms(pre_transform, post_transform):
    """Original norm comparison plots."""
    plt.figure(); plt.hist(norms(pre_transform), bins=50, alpha=.6, label='orig'); plt.hist(norms(post_transform), bins=50, alpha=.6, label='trans'); plt.title('L2 norms'); plt.legend()
    plt.figure(); plt.hist(cos_self(pre_transform, post_transform), bins=50); plt.title('cos(x, x\')')
    plt.figure(); plt.hist(norms(post_transform - pre_transform), bins=50); plt.title('||x\' - x||')
    plt.show()


def plot_distance_correlation(
    dist_orig: np.ndarray,
    dist_trans: np.ndarray,
    metric_name: str = "Tanimoto",
    sample_size: int = 1000,
    save_path: Optional[str] = None
):
    """
    Scatter plot comparing original vs transformed pairwise distances.

    Args:
        dist_orig: Original distance matrix [N, N]
        dist_trans: Transformed distance matrix [N, N]
        metric_name: Name of the distance metric for labeling
        sample_size: Number of points to plot (for efficiency)
        save_path: If provided, save figure to this path
    """
    # Extract upper triangle
    N = dist_orig.shape[0]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    orig_vec = dist_orig[mask].flatten()
    trans_vec = dist_trans[mask].flatten()

    # Sample for efficiency
    if len(orig_vec) > sample_size:
        idx = np.random.choice(len(orig_vec), sample_size, replace=False)
        orig_vec = orig_vec[idx]
        trans_vec = trans_vec[idx]

    # Scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(orig_vec, trans_vec, alpha=0.3, s=1)
    ax.plot([orig_vec.min(), orig_vec.max()],
            [orig_vec.min(), orig_vec.max()],
            'r--', label='y=x')
    ax.set_xlabel(f'Original {metric_name} Distance')
    ax.set_ylabel(f'Transformed {metric_name} Distance')
    ax.set_title(f'{metric_name} Distance Preservation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_value_distributions(
    X_orig: np.ndarray,
    X_trans: np.ndarray,
    transform_name: str = "Transformation",
    save_path: Optional[str] = None
):
    """
    Compare value distributions before and after transformation.

    Args:
        X_orig: Original features [N, D]
        X_trans: Transformed features [N, D]
        transform_name: Name of transformation for labeling
        save_path: If provided, save figure to this path
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Flatten for distribution analysis
    orig_vals = X_orig.flatten()
    trans_vals = X_trans.flatten()

    # 1. Histograms
    axes[0, 0].hist(orig_vals, bins=100, alpha=0.6, label='Original', density=True)
    axes[0, 0].hist(trans_vals, bins=100, alpha=0.6, label='Transformed', density=True)
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Value Distribution')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')

    # 2. Log-scale histogram (for sparse data)
    axes[0, 1].hist(np.log10(np.abs(orig_vals) + 1e-10), bins=100, alpha=0.6, label='Original', density=True)
    axes[0, 1].hist(np.log10(np.abs(trans_vals) + 1e-10), bins=100, alpha=0.6, label='Transformed', density=True)
    axes[0, 1].set_xlabel('log10(|Value|)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Log-Scale Value Distribution')
    axes[0, 1].legend()

    # 3. Sparsity per sample
    orig_nnz = (np.abs(X_orig) > 1e-6).sum(axis=1)
    trans_nnz = (np.abs(X_trans) > 1e-6).sum(axis=1)
    axes[1, 0].hist(orig_nnz, bins=50, alpha=0.6, label='Original')
    axes[1, 0].hist(trans_nnz, bins=50, alpha=0.6, label='Transformed')
    axes[1, 0].set_xlabel('Number of Non-Zero Features')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Sparsity Distribution')
    axes[1, 0].legend()

    # 4. L2 norms
    orig_norms = np.linalg.norm(X_orig, axis=1)
    trans_norms = np.linalg.norm(X_trans, axis=1)
    axes[1, 1].hist(orig_norms, bins=50, alpha=0.6, label='Original')
    axes[1, 1].hist(trans_norms, bins=50, alpha=0.6, label='Transformed')
    axes[1, 1].set_xlabel('L2 Norm')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Norm Distribution')
    axes[1, 1].legend()

    plt.suptitle(f'Distribution Analysis: {transform_name}')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_ablation_results(
    results_list: List[Dict],
    metric_key: str = "val_rmse",
    save_path: Optional[str] = None
):
    """
    Create a bar plot comparing downstream performance across transformations.

    Args:
        results_list: List of result dictionaries from ablation study
        metric_key: Which metric to plot (e.g., "val_rmse", "val_roc_auc")
        save_path: If provided, save figure to this path
    """
    transform_names = []
    metric_values = []

    for result in results_list:
        transform_names.append(result["transform_name"])
        perf = result["downstream_performance"].get(metric_key)
        if perf is not None:
            metric_values.append(perf)
        else:
            metric_values.append(np.nan)

    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(transform_names))

    bars = ax.bar(x, metric_values, alpha=0.7)

    # Color baseline differently
    for i, name in enumerate(transform_names):
        if name == "baseline":
            bars[i].set_color('red')
            bars[i].set_alpha(0.5)

    ax.set_xlabel('Transformation')
    ax.set_ylabel(metric_key)
    ax.set_title(f'Downstream Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(transform_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add horizontal line for baseline
    baseline_val = None
    for i, name in enumerate(transform_names):
        if name == "baseline":
            baseline_val = metric_values[i]
            break
    if baseline_val is not None:
        ax.axhline(y=baseline_val, color='red', linestyle='--', alpha=0.5, label='Baseline')
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_distance_preservation_heatmap(
    results_list: List[Dict],
    save_path: Optional[str] = None
):
    """
    Create a heatmap showing distance correlation for different transformations.

    Args:
        results_list: List of result dictionaries from ablation study
        save_path: If provided, save figure to this path
    """
    transform_names = []
    metrics = ["tanimoto", "euclidean", "cosine"]
    correlation_matrix = []

    for result in results_list:
        if result["distance_preservation"] is None:
            continue

        transform_names.append(result["transform_name"])
        row = []
        for metric in metrics:
            corr = result["distance_preservation"][metric]["correlation"]
            row.append(corr)
        correlation_matrix.append(row)

    if not correlation_matrix:
        print("No distance preservation data to plot")
        return

    correlation_matrix = np.array(correlation_matrix)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, len(transform_names) * 0.5 + 2))
    im = ax.imshow(correlation_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(transform_names)))
    ax.set_xticklabels(metrics)
    ax.set_yticklabels(transform_names)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Spearman Correlation', rotation=270, labelpad=15)

    # Add text annotations
    for i in range(len(transform_names)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{correlation_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)

    ax.set_title('Distance Preservation (Spearman Correlation)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_comprehensive_comparison(
    original_dataset,
    transformed_dataset,
    transform_name: str = "Transformation",
    save_dir: Optional[str] = None
):
    """
    Generate all comparison plots for a single transformation.

    Args:
        original_dataset: ECFPDataset with original features
        transformed_dataset: ECFPDataset with transformed features
        transform_name: Name of the transformation
        save_dir: Directory to save plots (if None, just display)
    """
    X_orig = original_dataset.features.cpu().numpy()
    X_trans = transformed_dataset.features.cpu().numpy()

    # Value distributions
    save_path = f"{save_dir}/{transform_name}_distributions.png" if save_dir else None
    plot_value_distributions(X_orig, X_trans, transform_name, save_path)

    # Classic norm plots
    plot_norms(X_orig, X_trans)