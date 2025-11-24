#!/usr/bin/env python3
"""
Compare the two Tanimoto variants and their behavior under transformations.
"""

import torch
import numpy as np

def dot_product_tanimoto(x, y):
    """Tanimoto based on dot products: s(x,y) = (x·y) / (||x||² + ||y||² - x·y)"""
    dot = (x * y).sum()
    norm_x = (x ** 2).sum()
    norm_y = (y ** 2).sum()
    return dot / (norm_x + norm_y - dot + 1e-8)

def continuous_tanimoto(x, y):
    """Continuous Tanimoto from NGF paper: s(x,y) = sum(min(x,y)) / sum(max(x,y))"""
    min_vals = torch.minimum(x, y)
    max_vals = torch.maximum(x, y)
    return min_vals.sum() / (max_vals.sum() + 1e-8)

# Test vectors
x = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
y = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0])

print("="*80)
print("Comparing Tanimoto Variants")
print("="*80)

print(f"\nx = {x.tolist()}")
print(f"y = {y.tolist()}")

sim_dot = dot_product_tanimoto(x, y)
sim_cont = continuous_tanimoto(x, y)

print(f"\nDot-product Tanimoto similarity: {sim_dot:.4f}")
print(f"Continuous Tanimoto similarity:  {sim_cont:.4f}")

# Test under rotation
print("\n" + "="*80)
print("Testing under ROTATION (should preserve dot-product version)")
print("="*80)

dim = 5
g = torch.Generator().manual_seed(42)
A = torch.randn(dim, dim, generator=g)
Q, R = torch.linalg.qr(A)

x_rot = Q @ x
y_rot = Q @ y

sim_dot_rot = dot_product_tanimoto(x_rot, y_rot)
sim_cont_rot = continuous_tanimoto(x_rot, y_rot)

print(f"\nAfter rotation:")
print(f"Dot-product Tanimoto: {sim_dot:.4f} -> {sim_dot_rot:.4f} (diff: {abs(sim_dot - sim_dot_rot):.2e})")
print(f"Continuous Tanimoto:  {sim_cont:.4f} -> {sim_cont_rot:.4f} (diff: {abs(sim_cont - sim_cont_rot):.2e})")

# Test under translation
print("\n" + "="*80)
print("Testing under TRANSLATION (neither preserves)")
print("="*80)

b = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
x_trans = x + b
y_trans = y + b

sim_dot_trans = dot_product_tanimoto(x_trans, y_trans)
sim_cont_trans = continuous_tanimoto(x_trans, y_trans)

print(f"\nAfter translation by {b[0]:.1f}:")
print(f"Dot-product Tanimoto: {sim_dot:.4f} -> {sim_dot_trans:.4f} (diff: {abs(sim_dot - sim_dot_trans):.2e})")
print(f"Continuous Tanimoto:  {sim_cont:.4f} -> {sim_cont_trans:.4f} (diff: {abs(sim_cont - sim_cont_trans):.2e})")

# Test with negative values
print("\n" + "="*80)
print("Testing with NEGATIVE values (continuous_tanimoto uses min/max)")
print("="*80)

x_neg = torch.tensor([1.0, -1.0, 2.0, -0.5, 0.0])
y_neg = torch.tensor([0.5, -2.0, 1.0, -0.5, 1.0])

print(f"\nx = {x_neg.tolist()}")
print(f"y = {y_neg.tolist()}")

sim_dot_neg = dot_product_tanimoto(x_neg, y_neg)
sim_cont_neg = continuous_tanimoto(x_neg, y_neg)

print(f"\nDot-product Tanimoto: {sim_dot_neg:.4f}")
print(f"Continuous Tanimoto:  {sim_cont_neg:.4f}")

print("\n" + "="*80)
print("Summary:")
print("- Dot-product Tanimoto: Preserved by rotation, uses inner products")
print("- Continuous Tanimoto: Uses element-wise min/max, from NGF paper")
print("- For binary ECFPs, both should give similar results")
print("- For continuous vectors, they can differ significantly")
print("="*80)
