#!/usr/bin/env python3
"""
Test script for Block Radius Linear Mixing transformation.

This demonstrates the complete workflow:
1. Generate multi-radius (delta) ECFPs
2. Apply block-wise linear mixing with nonlinearity
3. Evaluate downstream task performance
4. Analyze distance preservation
"""

import torch
import numpy as np

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

from dataset_utils import ECFPDataset
from downstream import run_downstream_task
from transforms import BlockRadiusLinearMixing
from pairwise_distances import analyze_distance_preservation, compare_sparsity

print("="*80)
print("Block Radius Linear Mixing - GNN-Inspired ECFP Transformation")
print("="*80)

# Step 1: Create multi-radius dataset
print("\n[1/6] Creating multi-radius ECFP dataset...")
print("  - Using radius=2 (3 blocks: r=0, r=1, r=2)")
print("  - Each block gets 512 bits")
print("  - Total dimension: 3 x 512 = 1536")

multi_radius_dataset = ECFPDataset(
    name="esol",
    split_type="random",
    target_index=0,
    radius=2,  # Will have blocks for radius 0, 1, 2
    n_bits=1536,  # Total bits across all radii
    use_count=False,
    multi_radius=True,  # Enable radius-delta fingerprints
    n_bits_per_radius=512  # 512 bits per radius block
)

print(f"  ✓ Dataset created: {len(multi_radius_dataset.features)} molecules")
print(f"  ✓ Feature dimension: {multi_radius_dataset.features.shape[1]}")
print(f"  ✓ Radius blocks: {multi_radius_dataset.radius_schema.blocks}")

# Step 2: Baseline performance (no transformation)
print("\n[2/6] Training MLP on original multi-radius ECFPs...")
baseline_result = run_downstream_task(
    multi_radius_dataset,
    task_type="regression",
    hidden_dim=128,
    epochs=100,
    lr=1e-3,
    device="cpu"
)

print(f"\n  Baseline Performance:")
print(f"    Val RMSE:  {baseline_result['val']:.4f}")
print(f"    Test RMSE: {baseline_result['test']:.4f}")

# Step 3: Create block radius mixing transformation
print("\n[3/6] Creating Block Radius Linear Mixing transformation...")
print("  - Applying orthogonal linear map per radius block")
print("  - Using ReLU nonlinearity")
print("  - L2 normalization after concatenation")

block_mixing = BlockRadiusLinearMixing(
    radius_blocks=multi_radius_dataset.radius_schema.blocks,
    nonlinearity="relu",
    seed=42
)

# Step 4: Apply transformation
print("\n[4/6] Applying transformation...")
transformed_dataset = ECFPDataset(
    name="esol",
    split_type="random",
    target_index=0,
    radius=2,
    n_bits=1536,
    use_count=False,
    multi_radius=True,
    n_bits_per_radius=512
)
transformed_dataset.apply_transform(block_mixing)
print("  ✓ Transformation applied")

# Step 5: Evaluate transformed dataset
print("\n[5/6] Training MLP on transformed ECFPs...")
transformed_result = run_downstream_task(
    transformed_dataset,
    task_type="regression",
    hidden_dim=128,
    epochs=100,
    lr=1e-3,
    device="cpu"
)

print(f"\n  Transformed Performance:")
print(f"    Val RMSE:  {transformed_result['val']:.4f}")
print(f"    Test RMSE: {transformed_result['test']:.4f}")

print(f"\n  Change: {transformed_result['val'] - baseline_result['val']:+.4f}")

# Step 6: Distance preservation analysis
print("\n[6/6] Analyzing distance preservation...")
distance_results = analyze_distance_preservation(
    multi_radius_dataset,
    transformed_dataset,
    metrics=["tanimoto", "continuous_tanimoto", "euclidean", "cosine"],
    correlation_method="spearman",
    sample_size=500
)

# Sparsity analysis
print("\n--- Sparsity Analysis ---")
sparsity_results = compare_sparsity(multi_radius_dataset, transformed_dataset)

# Final summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nDownstream Performance:")
print(f"  Original:    Val RMSE = {baseline_result['val']:.4f}")
print(f"  Transformed: Val RMSE = {transformed_result['val']:.4f}")
print(f"  Change: {transformed_result['val'] - baseline_result['val']:+.4f}")

print(f"\nDistance Preservation (Spearman Correlation):")
for metric, results in distance_results.items():
    print(f"  {metric:20s}: {results['correlation']:.4f}")

print(f"\nSparsity Changes:")
orig_zero_frac = sparsity_results['original']['zero_fraction']
trans_zero_frac = sparsity_results['transformed']['zero_fraction']
print(f"  Zero fraction: {orig_zero_frac:.4f} → {trans_zero_frac:.4f}")
print(f"  Change: {trans_zero_frac - orig_zero_frac:+.4f}")

print("\n" + "="*80)
print("Key Insights:")
print("- Block Radius Mixing mimics GNN layer-wise processing")
print("- Each radius (r=0, r=1, r=2) gets its own orthogonal transformation")
print("- Nonlinearity + L2 norm makes it more GNN-like")
print("- Preserves hierarchical structure from ECFP generation")
print("="*80)
