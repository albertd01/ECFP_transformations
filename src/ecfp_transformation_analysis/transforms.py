import math, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Protocol


class BaseTransform(Protocol):
    def fit(self, X: torch.Tensor) -> "BaseTransform": ...
    def __call__(self, x: torch.Tensor, idx: Optional[int] = None) -> torch.Tensor: ...
    def to(self, device: torch.device) -> "BaseTransform": ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...

class StatelessTransform:
    def fit(self, X): return self
    def to(self, device): return self
    def state_dict(self): return {}
    def load_state_dict(self, state): pass

class Compose:
    def __init__(self, transforms: Sequence[BaseTransform]): self.ts = list(transforms)
    def fit(self, X: torch.Tensor): 
        for t in self.ts: t.fit(X); 
        return self
    def __call__(self, x: torch.Tensor, idx: Optional[int] = None) -> torch.Tensor:
        for t in self.ts: x = t(x, idx)
        return x
    def to(self, device): 
        for t in self.ts: t.to(device)
        return self
    def state_dict(self): 
        return {i: t.state_dict() for i, t in enumerate(self.ts)}
    def load_state_dict(self, sd): 
        for i, t in enumerate(self.ts): t.load_state_dict(sd[i])


class Rotation(StatelessTransform):
    def __init__(self, Q: torch.Tensor):  # Q: [D,D] orthogonal
        self.Q = Q  # store on CPU; move in .to()
    def to(self, device): self.Q = self.Q.to(device); return self
    def __call__(self, x, idx=None): return self.Q @ x

class Permutation(StatelessTransform):
    def __init__(self, perm: torch.Tensor):  # [D] long
        self.perm = perm
    def to(self, device): self.perm = self.perm.to(device); return self
    def __call__(self, x, idx=None): return x[self.perm]

class Scaling(StatelessTransform):
    def __init__(self, scales: torch.Tensor): self.s = scales
    def to(self, device): self.s = self.s.to(device); return self
    def __call__(self, x, idx=None): return x * self.s

class Translation(StatelessTransform):
    def __init__(self, bias: torch.Tensor): self.b = bias
    def to(self, device): self.b = self.b.to(device); return self
    def __call__(self, x, idx=None): return x + self.b

class Shear(StatelessTransform):
    def __init__(self, shear_matrix: torch.Tensor):
        self.S = shear_matrix

    def to(self, device):
        self.S = self.S.to(device)
        return self

    def __call__(self, x, idx=None):
        return self.S @ x

class Reflection(StatelessTransform):
    def __init__(self, reflection_matrix: torch.Tensor):
        self.R = reflection_matrix

    def to(self, device):
        self.R = self.R.to(device)
        return self

    def __call__(self, x, idx=None):
        return self.R @ x

class DropoutDeterministic(StatelessTransform):
    """Same mask for all samples (useful for ablations)."""
    def __init__(self, keep: float, dim: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.mask = (torch.rand(dim, generator=g) < keep).float()
    def to(self, device): self.mask = self.mask.to(device); return self
    def __call__(self, x, idx=None): return x * self.mask

class RandomGaussianProjection(BaseTransform):
    """Fit: sample A once; Apply: x' = A x (dense)."""
    def __init__(self, out_dim: int, seed: int = 0):
        self.out_dim = out_dim
        self.seed = seed
        self.A = None
    def fit(self, X):
        D = X.shape[1]
        g = torch.Generator().manual_seed(self.seed)
        A = torch.randn(self.out_dim, D, generator=g) / math.sqrt(self.out_dim)
        # Optional: orthogonalize rows
        self.A, _ = torch.linalg.qr(A.T)  # [D,out] orthonormal cols
        self.A = self.A.T  # [out,D]
        return self
    def to(self, device):
        if self.A is not None: self.A = self.A.to(device)
        return self
    def __call__(self, x, idx=None): return self.A @ x
    def state_dict(self): return {"A": self.A}
    def load_state_dict(self, sd): self.A = sd["A"]


# ============================================================================
# Transformations for Density and Continuity
# ============================================================================

class GaussianNoise(StatelessTransform):
    def __init__(self, sigma: float = 0.1, seed: int = 0):
        self.sigma = sigma
        self.seed = seed
        self.rng = torch.Generator().manual_seed(seed)

    def __call__(self, x, idx=None):
        if idx is not None:
            g = torch.Generator().manual_seed(self.seed + idx)
        else:
            g = self.rng
        noise = torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype) * self.sigma
        return x + noise


class L2Normalization(StatelessTransform):
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, x, idx=None):
        norm = torch.linalg.norm(x) + self.eps
        return x / norm


class Standardization(BaseTransform):
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = X.mean(dim=0)
        self.std = X.std(dim=0) + self.eps
        return self

    def to(self, device):
        if self.mean is not None:
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        return self

    def __call__(self, x, idx=None):
        return (x - self.mean) / self.std

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, sd):
        self.mean = sd["mean"]
        self.std = sd["std"]


class NonlinearActivation(StatelessTransform):
    """
    Applies smooth non-linear transformations to create continuity.
    Supports: tanh, sigmoid, softplus, gelu.
    """
    def __init__(self, activation: str = "tanh", scale: float = 1.0):
        self.activation = activation.lower()
        self.scale = scale

        if self.activation not in ["tanh", "sigmoid", "softplus", "gelu"]:
            raise ValueError(f"Unknown activation: {activation}")

    def __call__(self, x, idx=None):
        x_scaled = x * self.scale

        if self.activation == "tanh":
            return torch.tanh(x_scaled)
        elif self.activation == "sigmoid":
            return torch.sigmoid(x_scaled)
        elif self.activation == "softplus":
            return torch.nn.functional.softplus(x_scaled)
        elif self.activation == "gelu":
            return torch.nn.functional.gelu(x_scaled)


class SparseToDense(StatelessTransform):
    """
    Converts sparse binary vectors to dense continuous representations.
    Uses temperature-scaled softmax: x'ᵢ = exp(xᵢ/T) / (Σⱼ exp(xⱼ/T) + ε).
    Lower temperature → more peaked (closer to original).
    Higher temperature → more uniform (denser).
    """
    def __init__(self, temperature: float = 1.0, mode: str = "softmax"):
        """
        Args:
            temperature: Controls density (higher = more dense/uniform)
            mode: "softmax" or "normalize" (simple normalization to sum to 1)
        """
        self.temperature = temperature
        self.mode = mode

    def __call__(self, x, idx=None):
        if self.mode == "softmax":
            # Apply softmax with temperature
            x_temp = x / self.temperature
            return torch.softmax(x_temp, dim=0)
        elif self.mode == "normalize":
            # Simple normalization to [0, 1] and sum to 1
            x_pos = torch.relu(x)  # Ensure positive
            total = x_pos.sum() + 1e-8
            return x_pos / total
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class AdaptiveScaling(BaseTransform):
    """
    Feature-wise learned scaling and shifting: x' = scale * x + shift.
    Similar to batch normalization but with parameters learned from data.
    Stateful: computes scale and shift from training data statistics.
    """
    def __init__(self, method: str = "minmax"):
        """
        Args:
            method: "minmax" (scale to [0,1]) or "robust" (using median/IQR)
        """
        self.method = method
        self.scale = None
        self.shift = None

    def fit(self, X):
        # X: [N, D]
        if self.method == "minmax":
            x_min = X.min(dim=0)[0]
            x_max = X.max(dim=0)[0]
            self.scale = 1.0 / (x_max - x_min + 1e-8)
            self.shift = -x_min * self.scale
        elif self.method == "robust":
            # Use median and IQR for robustness to outliers
            median = X.median(dim=0)[0]
            q25 = torch.quantile(X, 0.25, dim=0)
            q75 = torch.quantile(X, 0.75, dim=0)
            iqr = q75 - q25 + 1e-8
            self.scale = 1.0 / iqr
            self.shift = -median * self.scale
        return self

    def to(self, device):
        if self.scale is not None:
            self.scale = self.scale.to(device)
            self.shift = self.shift.to(device)
        return self

    def __call__(self, x, idx=None):
        return self.scale * x + self.shift

    def state_dict(self):
        return {"scale": self.scale, "shift": self.shift}

    def load_state_dict(self, sd):
        self.scale = sd["scale"]
        self.shift = sd["shift"]


# ============================================================================
# Block Radius Linear Mixing (GNN-inspired transformation)
# ============================================================================

class BlockRadiusLinearMixing(StatelessTransform):
    """
    GNN-inspired transformation for multi-radius ECFPs.

    Applies orthogonal linear maps + nonlinearity within each radius block,
    then concatenates and L2 normalizes. This mimics GNN layer-wise processing
    where each "layer" (radius) has its own transformation.

    This transformation requires multi-radius ECFP inputs where different radii
    are stored in separate blocks of the feature vector.

    Args:
        radius_blocks: List of (start, end) indices for each radius block
                      e.g., [(0, 512), (512, 1024), (1024, 1536)] for radius=2
        nonlinearity: Activation function ("relu", "tanh", "gelu", "sigmoid")
        seed: Random seed for generating orthogonal matrices
    """
    def __init__(
        self,
        radius_blocks: Sequence[tuple],
        nonlinearity: str = "relu",
        seed: int = 0,
        normalize = True
    ):
        self.radius_blocks = list(radius_blocks)
        self.nonlinearity = nonlinearity
        self.seed = seed

        # Generate orthogonal matrix for each radius block
        g = torch.Generator().manual_seed(seed)
        self.Q_blocks = []

        for (start, end) in self.radius_blocks:
            dim = end - start
            # Generate random orthogonal matrix via QR decomposition
            A = torch.randn(dim, dim, generator=g)
            Q, R = torch.linalg.qr(A)
            # Fix sign ambiguity
            d = torch.sign(torch.diagonal(R))
            Q = Q * d
            self.Q_blocks.append(Q)

        # Set up nonlinearity
        if nonlinearity == "relu":
            self.activation = torch.relu
        elif nonlinearity == "tanh":
            self.activation = torch.tanh
        elif nonlinearity == "gelu":
            self.activation = F.gelu
        elif nonlinearity == "sigmoid":
            self.activation = torch.sigmoid
        elif nonlinearity == "identity":
            self.activation = lambda x : x
        else:
            raise ValueError(f"Unknown nonlinearity: {nonlinearity}")

    def to(self, device):
        self.Q_blocks = [Q.to(device) for Q in self.Q_blocks]
        return self

    def __call__(self, x, idx=None):
        """
        Apply block-wise linear mixing + nonlinearity + L2 normalization.

        Args:
            x: Input vector [D] where D = sum of all radius block dimensions
            idx: Optional sample index (unused)

        Returns:
            Transformed vector [D] after block mixing and L2 normalization
        """
        transformed_blocks = []

        for i, (start, end) in enumerate(self.radius_blocks):
            # Extract radius block
            block = x[start:end]

            # Apply orthogonal linear map
            block_transformed = self.Q_blocks[i] @ block

            # Apply nonlinearity
            block_transformed = self.activation(block_transformed)

            transformed_blocks.append(block_transformed)

        # Concatenate all transformed blocks
        result = torch.cat(transformed_blocks)

        # L2 normalize
        norm = torch.linalg.norm(result) + 1e-8
        result = result / norm

        return result
