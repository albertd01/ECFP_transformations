#!/usr/bin/env python3
"""
Verify that dot-product Tanimoto preserves rotation while continuous Tanimoto does not.
"""

import torch
import numpy as np

def generate_random_rotation(dim: int, seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(dim, dim, generator=g)
    Q, R = torch.linalg.qr(A)
    d = torch.sign(torch.diagonal(R))
    Q = Q * d
    return Q

def dot_product_tanimoto_distance(x, y):
    """Tanimoto distance: d = 1 - (x·y) / (||x||² + ||y||² - x·y)"""
    dot = (x * y).sum()
    norm_x = (x ** 2).sum()
    norm_y = (y ** 2).sum()
    similarity = dot / (norm_x + norm_y - dot + 1e-8)
    return 1.0 - similarity

def continuous_tanimoto_distance(x, y):
    """Continuous Tanimoto: d = 1 - sum(min(x,y)) / sum(max(x,y))"""
    min_vals = torch.minimum(x, y)
    max_vals = torch.maximum(x, y)
    similarity = min_vals.sum() / (max_vals.sum() + 1e-8)
    return 1.0 - similarity

# Test with binary vectors
print("="*80)
print("Testing Tanimoto Variants Under Rotation")
print("="*80)

dim = 20
n_samples = 5

# Generate binary vectors
torch.manual_seed(42)
X = torch.randint(0, 2, (n_samples, dim)).float()

print(f"\nTesting with {n_samples} binary vectors of dimension {dim}")

# Generate rotation
Q = generate_random_rotation(dim, seed=42)
X_rot = X @ Q.T

print(f"\n{'Pair':<8} {'Dot-Prod Original':<20} {'Dot-Prod Rotated':<20} {'Diff':<15}")
print("-" * 70)

for i in range(min(3, n_samples)):
    for j in range(i+1, min(3, n_samples)):
        dist_orig = dot_product_tanimoto_distance(X[i], X[j])
        dist_rot = dot_product_tanimoto_distance(X_rot[i], X_rot[j])
        diff = abs(dist_orig - dist_rot)
        print(f"({i},{j})    {dist_orig:.6f}           {dist_rot:.6f}           {diff:.2e}")

print(f"\n{'Pair':<8} {'Cont. Original':<20} {'Cont. Rotated':<20} {'Diff':<15}")
print("-" * 70)

for i in range(min(3, n_samples)):
    for j in range(i+1, min(3, n_samples)):
        dist_orig = continuous_tanimoto_distance(X[i], X[j])
        dist_rot = continuous_tanimoto_distance(X_rot[i], X_rot[j])
        diff = abs(dist_orig - dist_rot)
        print(f"({i},{j})    {dist_orig:.6f}           {dist_rot:.6f}           {diff:.2e}")

print("\n" + "="*80)
print("Conclusion:")
print("- Dot-product Tanimoto: Preserved under rotation (diff ≈ 0)")
print("- Continuous Tanimoto: NOT preserved under rotation (large diff)")
print("="*80)

# Now test permutation
print("\n" + "="*80)
print("Testing Tanimoto Variants Under Permutation")
print("="*80)

perm = torch.randperm(dim, generator=torch.Generator().manual_seed(42))
X_perm = X[:, perm]

print(f"\n{'Pair':<8} {'Dot-Prod Original':<20} {'Dot-Prod Permuted':<20} {'Diff':<15}")
print("-" * 70)

for i in range(min(3, n_samples)):
    for j in range(i+1, min(3, n_samples)):
        dist_orig = dot_product_tanimoto_distance(X[i], X[j])
        dist_perm = dot_product_tanimoto_distance(X_perm[i], X_perm[j])
        diff = abs(dist_orig - dist_perm)
        print(f"({i},{j})    {dist_orig:.6f}           {dist_perm:.6f}           {diff:.2e}")

print(f"\n{'Pair':<8} {'Cont. Original':<20} {'Cont. Permuted':<20} {'Diff':<15}")
print("-" * 70)

for i in range(min(3, n_samples)):
    for j in range(i+1, min(3, n_samples)):
        dist_orig = continuous_tanimoto_distance(X[i], X[j])
        dist_perm = continuous_tanimoto_distance(X_perm[i], X_perm[j])
        diff = abs(dist_orig - dist_perm)
        print(f"({i},{j})    {dist_orig:.6f}           {dist_perm:.6f}           {diff:.2e}")

print("\n" + "="*80)
print("Conclusion:")
print("- Dot-product Tanimoto: Preserved under permutation (diff ≈ 0)")
print("- Continuous Tanimoto: Preserved under permutation (diff ≈ 0)")
print("="*80)
