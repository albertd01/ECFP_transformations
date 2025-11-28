#!/usr/bin/env python3
"""
Systematic ablation study script for ECFP transformations.

This script evaluates individual transformations to understand their effects on:
1. Downstream task performance (RMSE/ROC-AUC)
2. Pairwise distance preservation (Tanimoto, Euclidean, Cosine correlations)
3. Sparsity and density metrics

Usage:
    python ablation_study.py --dataset esol --task regression
"""

import argparse
import json
from pathlib import Path
import torch
import numpy as np
from typing import Dict, List, Any

from dataset_utils import ECFPDataset
from downstream import run_downstream_task
from transforms import (
    GaussianNoise, L2Normalization, Standardization, NonlinearActivation,
    SparseToDense, AdaptiveScaling, Rotation, Permutation, RandomGaussianProjection,
    Scaling, Translation, Shear, Reflection
)
from pairwise_distances import analyze_distance_preservation, compare_sparsity


def convert_to_serializable(obj):
    """
    Recursively convert numpy/torch types to native Python types for JSON serialization.
    """
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj



def run_transformation_experiment(
    transform_name: str,
    transform_obj,
    dataset_name: str,
    task_type: str,
    n_bits: int = 2048,
    radius: int = 2,
    use_count: bool = False,
    hidden_dim: int = 128,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
    distance_sample_size: int = 500
) -> Dict[str, Any]:
    """
    Run a complete experiment for a single transformation.

    Returns dictionary with:
        - downstream_performance: RMSE or ROC-AUC
        - distance_correlations: Tanimoto, Euclidean, Cosine
        - sparsity_metrics: Before and after transformation
    """
    print(f"\n{'='*80}")
    print(f"Experiment: {transform_name}")
    print(f"{'='*80}\n")

    # Load original dataset
    print("Loading original dataset...")
    original_dataset = ECFPDataset(
        name=dataset_name,
        split_type="random",
        target_index=0,
        n_bits=n_bits,
        radius=radius,
        use_count=use_count
    )

    # Load transformed dataset
    print("Loading transformed dataset...")
    transformed_dataset = ECFPDataset(
        name=dataset_name,
        split_type="random",
        target_index=0,
        n_bits=n_bits,
        radius=radius,
        use_count=use_count
    )

    # Apply transformation
    print(f"Applying transformation: {transform_name}")
    transformed_dataset.apply_transform(transform_obj)

    # 1. Downstream task performance
    print("\n--- Downstream Task Performance ---")
    downstream_result = run_downstream_task(
        transformed_dataset,
        task_type=task_type,
        hidden_dim=hidden_dim,
        epochs=epochs,
        lr=lr,
        device=device
    )
    print(f"Result: {downstream_result}")

    # 2. Distance preservation analysis
    print("\n--- Distance Preservation Analysis ---")
    distance_results = analyze_distance_preservation(
        original_dataset,
        transformed_dataset,
        metrics=["tanimoto", "euclidean", "cosine", "continuous_tanimoto"],
        correlation_method="spearman",
        sample_size=distance_sample_size
    )

    # 3. Sparsity analysis
    sparsity_results = compare_sparsity(original_dataset, transformed_dataset)

    # Compile results
    results = {
        "transform_name": transform_name,
        "dataset": dataset_name,
        "task_type": task_type,
        "downstream_performance": downstream_result,
        "distance_preservation": distance_results,
        "sparsity_metrics": sparsity_results
    }

    return results


def generate_random_rotation(dim: int, seed: int = 42) -> torch.Tensor:
    """Generate a random orthogonal rotation matrix via QR decomposition."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(dim, dim, generator=g)
    Q, R = torch.linalg.qr(A)
    # Ensure determinant is +1 (proper rotation)
    d = torch.sign(torch.diagonal(R))
    Q = Q * d
    return Q


def main():
    parser = argparse.ArgumentParser(description="ECFP Transformation Ablation Study")
    parser.add_argument("--dataset", type=str, default="esol",
                        help="Dataset name (esol, bace, lipo, etc.)")
    parser.add_argument("--task", type=str, default="regression",
                        choices=["regression", "classification"],
                        help="Task type")
    parser.add_argument("--n_bits", type=int, default=2048,
                        help="ECFP fingerprint size")
    parser.add_argument("--radius", type=int, default=2,
                        help="ECFP radius")
    parser.add_argument("--use_count", action="store_true",
                        help="Use count fingerprints instead of binary")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu or cuda)")
    parser.add_argument("--output", type=str, default="ablation_results.json",
                        help="Output JSON file")
    parser.add_argument("--transformations", type=str, nargs="+",
                        default=["all"],
                        help="Which transformations to test (all, gaussian_noise, l2_norm, etc.)")

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate affine transformation parameters
    dim = args.n_bits

    # Random rotation matrix
    Q_rotation = generate_random_rotation(dim, seed=42)

    # Random permutation
    perm = torch.randperm(dim, generator=torch.Generator().manual_seed(42))

    # Random scaling (uniform between 0.5 and 2.0)
    g = torch.Generator().manual_seed(42)
    scales_uniform = torch.rand(dim, generator=g) * 1.5 + 0.5  # [0.5, 2.0]

    # Random scaling (Gaussian around 1.0)
    scales_gaussian = torch.randn(dim, generator=torch.Generator().manual_seed(43)) * 0.3 + 1.0

    # Random translation (small values)
    translation_small = torch.randn(dim, generator=torch.Generator().manual_seed(44)) * 0.1
    translation_medium = torch.randn(dim, generator=torch.Generator().manual_seed(45)) * 0.5

    # Shear matrices
    # Simple shear: identity matrix with small off-diagonal elements
    shear_matrix_small = torch.eye(dim)
    # Add shear in first 10 dimensions (shear factor 0.1)
    for i in range(min(10, dim-1)):
        shear_matrix_small[i, i+1] = 0.1

    shear_matrix_medium = torch.eye(dim)
    # Add shear in first 10 dimensions (shear factor 0.3)
    for i in range(min(10, dim-1)):
        shear_matrix_medium[i, i+1] = 0.3

    # Reflection matrices
    # 1. Reflection through origin: -I
    reflection_origin = -torch.eye(dim)

    # 2. Reflection across a random hyperplane: R = I - 2*(v⊗v) where v is unit normal
    v = torch.randn(dim, generator=torch.Generator().manual_seed(46))
    v = v / torch.linalg.norm(v)  # Normalize to unit vector
    reflection_hyperplane = torch.eye(dim) - 2 * torch.outer(v, v)

    # Define transformations to test
    transformations = {
        "baseline": None,  # No transformation

        # === Affine Transformations ===
        "rotation": Rotation(Q_rotation),
        "permutation": Permutation(perm),
        "scaling_uniform": Scaling(scales_uniform),
        "scaling_gaussian": Scaling(scales_gaussian),
        "translation_small": Translation(translation_small),
        "translation_medium": Translation(translation_medium),
        "shear_small": Shear(shear_matrix_small),
        "shear_medium": Shear(shear_matrix_medium),
        "reflection_origin": Reflection(reflection_origin),
        "reflection_hyperplane": Reflection(reflection_hyperplane),

        # === Continuity Transformations ===
        "gaussian_noise_0.05": GaussianNoise(sigma=0.05, seed=42),
        "gaussian_noise_0.1": GaussianNoise(sigma=0.1, seed=42),
        "gaussian_noise_0.2": GaussianNoise(sigma=0.2, seed=42),

        # === Normalization Transformations ===
        "l2_normalization": L2Normalization(),
        "standardization": Standardization(),

        # === Nonlinear Transformations ===
        "tanh": NonlinearActivation(activation="tanh", scale=1.0),
        "sigmoid": NonlinearActivation(activation="sigmoid", scale=1.0),
        "softplus": NonlinearActivation(activation="softplus", scale=1.0),

        # === Density Transformations ===
        "sparse_to_dense_t0.5": SparseToDense(temperature=0.5, mode="softmax"),
        "sparse_to_dense_t1.0": SparseToDense(temperature=1.0, mode="softmax"),
        "sparse_to_dense_t2.0": SparseToDense(temperature=2.0, mode="softmax"),

        # === Adaptive Transformations ===
        "adaptive_minmax": AdaptiveScaling(method="minmax"),
        "adaptive_robust": AdaptiveScaling(method="robust"),
    }

    # Filter transformations if specified
    if "all" not in args.transformations:
        transformations = {k: v for k, v in transformations.items()
                          if k in args.transformations or k == "baseline"}

    # Run experiments
    all_results = []

    for transform_name, transform_obj in transformations.items():
        if transform_obj is None:
            # Baseline: no transformation
            print(f"\n{'='*80}")
            print(f"Baseline (No Transformation)")
            print(f"{'='*80}\n")

            original_dataset = ECFPDataset(
                name=args.dataset,
                split_type="random",
                target_index=0,
                n_bits=args.n_bits,
                radius=args.radius,
                use_count=args.use_count
            )

            downstream_result = run_downstream_task(
                original_dataset,
                task_type=args.task,
                hidden_dim=128,
                epochs=args.epochs,
                lr=1e-3,
                device=args.device
            )

            results = {
                "transform_name": "baseline",
                "dataset": args.dataset,
                "task_type": args.task,
                "downstream_performance": downstream_result,
                "distance_preservation": None,
                "sparsity_metrics": None
            }

        else:
            results = run_transformation_experiment(
                transform_name=transform_name,
                transform_obj=transform_obj,
                dataset_name=args.dataset,
                task_type=args.task,
                n_bits=args.n_bits,
                radius=args.radius,
                use_count=args.use_count,
                hidden_dim=128,
                epochs=args.epochs,
                lr=1e-3,
                device=args.device,
                distance_sample_size=500
            )

        all_results.append(results)

    # Save results (convert numpy/torch types to native Python types)
    output_path = Path(args.output)
    serializable_results = convert_to_serializable(all_results)
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}\n")

    # Print summary
    print("\n=== SUMMARY ===\n")
    metric_name = "val_rmse" if args.task == "regression" else "val_roc_auc"

    print(f"{'Transformation':<30} {metric_name:>15}")
    print("-" * 50)

    for result in all_results:
        name = result["transform_name"]
        perf = result["downstream_performance"].get("metric", "N/A") + " " + str(result["downstream_performance"].get("val", "N/A"))
        if isinstance(perf, float):
            print(f"{name:<30} {perf:>15.4f}")
        else:
            print(f"{name:<30} {str(perf):>15}")


if __name__ == "__main__":
    main()
