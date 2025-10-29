import math, numpy as np, torch
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
