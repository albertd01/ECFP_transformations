import numpy as np
import matplotlib.pyplot as plt
def compute_sparsity(fingerprints, threshold=1e-4):
    """
    Compute sparsity stats for a matrix of fingerprints.

    Args:
        fingerprints (np.ndarray): shape [N, D]
        threshold (float): below which a value is considered "inactive"

    Returns:
        dict: {
            "mean_active_bits": float,
            "mean_sparsity": float,
            "bit_usage": np.ndarray of shape [D]
        }
    """
    binarized = (np.abs(fingerprints) > threshold).astype(np.int32)  # shape [N, D]
    active_per_sample = np.sum(binarized, axis=1)
    sparsity_per_sample = 1.0 - (active_per_sample / fingerprints.shape[1])
    bit_usage = np.sum(binarized, axis=0)  # how many samples activate each bit

    return {
        "mean_active_bits": np.mean(active_per_sample),
        "mean_sparsity": np.mean(sparsity_per_sample),
        "bit_usage": bit_usage
    }


