#!/usr/bin/env python3
"""
Test that rotation actually preserves the mathematical properties we expect.
"""

import torch
import numpy as np

def generate_random_rotation(dim: int, seed: int = 42) -> torch.Tensor:
    """Generate a random orthogonal rotation matrix via QR decomposition."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(dim, dim, generator=g)
    Q, R = torch.linalg.qr(A)
    d = torch.sign(torch.diagonal(R))
    Q = Q * d
    return Q


# Test with a small example
dim = 10
Q = generate_random_rotation(dim)

print("="*80)
print("Testing Rotation Properties")
print("="*80)

# Check if Q is orthogonal: Q^T Q = I
QTQ = Q.T @ Q
I = torch.eye(dim)
orthogonality_error = torch.norm(QTQ - I)
print(f"\n1. Orthogonality test: ||Q^T Q - I|| = {orthogonality_error:.2e}")
if orthogonality_error < 1e-6:
    print("   ✓ Q is orthogonal")
else:
    print("   ✗ Q is NOT orthogonal!")

# Create some test vectors (binary-like)
torch.manual_seed(42)
n_samples = 5
X = torch.randint(0, 2, (n_samples, dim)).float()  # Binary vectors

# Apply rotation
X_rot = X @ Q.T  # Each row x_i -> x_i @ Q^T = Q @ x_i^T transposed

print(f"\n2. Testing on {n_samples} binary vectors...")

# Check dot products are preserved
print("\n   Dot products:")
for i in range(min(3, n_samples)):
    for j in range(i+1, min(3, n_samples)):
        dot_orig = (X[i] @ X[j]).item()
        dot_rot = (X_rot[i] @ X_rot[j]).item()
        diff = abs(dot_orig - dot_rot)
        print(f"     pair ({i},{j}): {dot_orig:.4f} -> {dot_rot:.4f}, diff={diff:.2e}")

# Check norms are preserved
print("\n   L2 Norms:")
for i in range(min(3, n_samples)):
    norm_orig = torch.norm(X[i]).item()
    norm_rot = torch.norm(X_rot[i]).item()
    diff = abs(norm_orig - norm_rot)
    print(f"     vector {i}: {norm_orig:.4f} -> {norm_rot:.4f}, diff={diff:.2e}")

# Check Tanimoto distances
print("\n   Tanimoto Distances:")
for i in range(min(3, n_samples)):
    for j in range(i+1, min(3, n_samples)):
        # Original
        dot_orig = X[i] @ X[j]
        norm_i_orig = torch.sum(X[i]**2)
        norm_j_orig = torch.sum(X[j]**2)
        tanimoto_orig = dot_orig / (norm_i_orig + norm_j_orig - dot_orig + 1e-8)

        # Rotated
        dot_rot = X_rot[i] @ X_rot[j]
        norm_i_rot = torch.sum(X_rot[i]**2)
        norm_j_rot = torch.sum(X_rot[j]**2)
        tanimoto_rot = dot_rot / (norm_i_rot + norm_j_rot - dot_rot + 1e-8)

        diff = abs(tanimoto_orig - tanimoto_rot)
        print(f"     pair ({i},{j}): {tanimoto_orig:.6f} -> {tanimoto_rot:.6f}, diff={diff:.2e}")

# Check Cosine distances
print("\n   Cosine Similarities:")
for i in range(min(3, n_samples)):
    for j in range(i+1, min(3, n_samples)):
        # Original
        dot_orig = X[i] @ X[j]
        norm_i_orig = torch.norm(X[i])
        norm_j_orig = torch.norm(X[j])
        cos_orig = dot_orig / (norm_i_orig * norm_j_orig + 1e-8)

        # Rotated
        dot_rot = X_rot[i] @ X_rot[j]
        norm_i_rot = torch.norm(X_rot[i])
        norm_j_rot = torch.norm(X_rot[j])
        cos_rot = dot_rot / (norm_i_rot * norm_j_rot + 1e-8)

        diff = abs(cos_orig - cos_rot)
        print(f"     pair ({i},{j}): {cos_orig:.6f} -> {cos_rot:.6f}, diff={diff:.2e}")

print("\n" + "="*80)
print("If all differences are < 1e-6, rotation is working correctly!")
print("="*80)
