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


def plot_bit_usage_cdf(fp_ecfp: np.ndarray, emb_ngf: np.ndarray, threshold=1e-6):
    """
    Plot the CDF of bit-usage (activation counts) for ECFP vs NGF.
    fp_ecfp: [N, D] binary (0/1) or count ECFP fingerprints
    emb_ngf: [N, D] continuous NGF embeddings
    threshold: for NGF, treat values > threshold as 'active'
    """
    # Compute usage counts
    usage_ecfp = np.sum(fp_ecfp > 0, axis=0)        # counts of molecules per bit
    usage_ngf  = np.sum(emb_ngf  > threshold, axis=0)

    # Sort and normalize to CDF
    def make_cdf(arr):
        s = np.sort(arr)
        return s, np.arange(1, len(s)+1) / len(s)

    s_ecfp, c_ecfp = make_cdf(usage_ecfp)
    s_ngf,  c_ngf  = make_cdf(usage_ngf)

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(s_ecfp, c_ecfp, label="ECFP", color="C0")
    plt.plot(s_ngf,  c_ngf,  label="NGF",  color="C1")
    plt.xlabel("Bit Usage Count")
    plt.ylabel("Empirical CDF")
    plt.title("Bit-Usage CDF: ECFP vs NGF")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()