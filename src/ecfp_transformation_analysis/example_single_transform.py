#!/usr/bin/env python3
"""
Quick start example: Test a single transformation.

This script demonstrates how to:
1. Load a dataset with ECFPs
2. Apply a transformation (e.g., GaussianNoise)
3. Evaluate downstream task performance
4. Analyze distance preservation (including Tanimoto)
5. Compare sparsity metrics
6. Visualize the effects

Usage:
    python example_single_transform.py
"""

import torch
import numpy as np
from dataset_utils import ECFPDataset
from downstream import run_downstream_task
from transforms import (
    GaussianNoise, L2Normalization, Standardization,
    NonlinearActivation, SparseToDense
)
from pairwise_distances import (
    analyze_distance_preservation,
    compare_sparsity
)
from plots import plot_value_distributions

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("="*80)
print("ECFP Transformation Analysis - Single Transform Example")
print("="*80)

# ============================================================================
# 1. Load Original Dataset
# ============================================================================
print("\n[1/6] Loading original ESOL dataset...")
original_dataset = ECFPDataset(
    name="esol",
    split_type="random",
    target_index=0,
    n_bits=2048,
    radius=2,
    use_count=False  # Binary fingerprints
)
print(f"  Dataset size: {len(original_dataset.features)} molecules")
print(f"  Feature dimension: {original_dataset.features.shape[1]}")

# ============================================================================
# 2. Create Transformed Dataset
# ============================================================================
print("\n[2/6] Creating transformed dataset with GaussianNoise...")

# Choose a transformation
# Options:
# - GaussianNoise(sigma=0.1): Adds Gaussian noise for continuity
# - L2Normalization(): Normalizes to unit norm
# - Standardization(): Z-score normalization (needs fit())
# - NonlinearActivation(activation="tanh"): Apply smooth activation
# - SparseToDense(temperature=1.0): Converts to dense distribution

transform = GaussianNoise(sigma=0.1, seed=42)
# Uncomment to try other transformations:
# transform = L2Normalization()
# transform = NonlinearActivation(activation="tanh", scale=1.0)
# transform = SparseToDense(temperature=1.0, mode="softmax")

# Create transformed dataset
transformed_dataset = ECFPDataset(
    name="esol",
    split_type="random",
    target_index=0,
    n_bits=2048,
    radius=2,
    use_count=False
)

# Apply transformation
transformed_dataset.apply_transform(transform)
print(f"  Transformation applied: {transform.__class__.__name__}")

# ============================================================================
# 3. Evaluate Downstream Task Performance
# ============================================================================
print("\n[3/6] Training MLP on original ECFP...")
original_result = run_downstream_task(
    original_dataset,
    task_type="regression",
    hidden_dim=128,
    epochs=100,
    lr=1e-3,
    device="cpu"
)
print(f"  Original - Val RMSE: {original_result['val_rmse']:.4f}, Test RMSE: {original_result['test_rmse']:.4f}")

print("\n[4/6] Training MLP on transformed ECFP...")
transformed_result = run_downstream_task(
    transformed_dataset,
    task_type="regression",
    hidden_dim=128,
    epochs=100,
    lr=1e-3,
    device="cpu"
)
print(f"  Transformed - Val RMSE: {transformed_result['val_rmse']:.4f}, Test RMSE: {transformed_result['test_rmse']:.4f}")

# ============================================================================
# 4. Analyze Distance Preservation (Including Tanimoto)
# ============================================================================
print("\n[5/6] Analyzing pairwise distance preservation...")
distance_results = analyze_distance_preservation(
    original_dataset,
    transformed_dataset,
    metrics=["tanimoto", "euclidean", "cosine"],
    correlation_method="spearman",
    sample_size=500  # Use 500 samples for efficiency
)

# ============================================================================
# 5. Compare Sparsity and Density
# ============================================================================
print("\n[6/6] Comparing sparsity metrics...")
sparsity_results = compare_sparsity(original_dataset, transformed_dataset)

# ============================================================================
# 6. Visualize Results
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nTransformation: {transform.__class__.__name__}")

print("\nDownstream Performance:")
print(f"  Original    - Val: {original_result['val_rmse']:.4f}, Test: {original_result['test_rmse']:.4f}")
print(f"  Transformed - Val: {transformed_result['val_rmse']:.4f}, Test: {transformed_result['test_rmse']:.4f}")
delta = transformed_result['val_rmse'] - original_result['val_rmse']
print(f"  Change: {delta:+.4f} ({'worse' if delta > 0 else 'better'})")

print("\nDistance Preservation (Spearman Correlation):")
for metric, results in distance_results.items():
    print(f"  {metric.capitalize()}: {results['correlation']:.4f}")

print("\nSparsity Changes:")
orig_sparsity = sparsity_results['original']['zero_fraction']
trans_sparsity = sparsity_results['transformed']['zero_fraction']
print(f"  Zero fraction: {orig_sparsity:.4f} → {trans_sparsity:.4f}")
orig_nnz = sparsity_results['original']['nonzero_per_sample']
trans_nnz = sparsity_results['transformed']['nonzero_per_sample']
print(f"  Non-zero per sample: {orig_nnz:.1f} → {trans_nnz:.1f}")

print("\n" + "="*80)
print("Generating visualization...")
print("="*80)

# Create visualization
plot_value_distributions(
    original_dataset.features.cpu().numpy(),
    transformed_dataset.features.cpu().numpy(),
    transform_name=transform.__class__.__name__
)

print("\nDone!")
