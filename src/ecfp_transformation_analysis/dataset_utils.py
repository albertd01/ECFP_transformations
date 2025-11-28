#!/usr/bin/env python3
"""
ECFPDataset: a reusable in-memory ECFP wrapper around PyG MoleculeNet.

- Works for any MoleculeNet dataset name available in torch_geometric.
- Computes binary Morgan fingerprints (ECFP) with RDKit.
- Supports scaffold or random splits.
- Multi-task aware (choose a single target or use all targets).
- Optional feature transform callable applied on-the-fly at __getitem__.
"""

from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch_geometric.datasets import MoleculeNet

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem,rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

    



# -----------------------------
# Utility functions
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def smiles_to_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, isomericSmiles=True)

def scaffold_split_indices(
    smiles_list: Sequence[str],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    seed: int = 0,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Deterministic scaffold split:
      - group by Murcko scaffold
      - sort groups by size (desc), tie-breaker scaffold string
      - greedy fill train -> valid -> test to target sizes
    """
    _ = seed  # deterministic without RNG
    scaffold_to_indices = {}
    for i, s in enumerate(smiles_list):
        scaf = smiles_to_scaffold(s)
        scaffold_to_indices.setdefault(scaf, []).append(i)

    bins = sorted(
        scaffold_to_indices.values(),
        key=lambda idxs: (-len(idxs), smiles_to_scaffold(smiles_list[idxs[0]]))
    )

    n_total = len(smiles_list)
    n_train = int(frac_train * n_total)
    n_valid = int(frac_valid * n_total)

    train_idx, valid_idx, test_idx = [], [], []
    for idxs in bins:
        if len(train_idx) + len(idxs) <= n_train:
            train_idx += idxs
        elif len(valid_idx) + len(idxs) <= n_valid:
            valid_idx += idxs
        else:
            test_idx += idxs

    return train_idx, valid_idx, test_idx

def random_split_indices(
    n: int,
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    seed: int = 0
) -> Tuple[List[int], List[int], List[int]]:
    rng = np.random.RandomState(seed)
    idxs = np.arange(n)
    rng.shuffle(idxs)
    n_train = int(frac_train * n)
    n_valid = int(frac_valid * n)
    train = idxs[:n_train].tolist()
    valid = idxs[n_train:n_train+n_valid].tolist()
    test  = idxs[n_train+n_valid:].tolist()
    return train, valid, test

def morgan_ecfp_bits(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
    use_count: bool = False
) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    if use_count:
        fp = mfpgen.GetCountFingerprint(mol)
    else:
        fp = mfpgen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def morgan_ecfp_radius_deltas(
    smiles: str,
    max_radius: int = 2,
    n_bits_per_radius: int = 512,
    use_chirality: bool = True,
    use_count: bool = False
) -> np.ndarray:
    """
    Generate radius-specific delta fingerprints (Approach B).

    For each radius r ∈ {0, 1, ..., max_radius}, compute:
    - fp[r=0]: Fingerprint with radius 0 (atom features)
    - fp[r=1]: Fingerprint with radius 1 MINUS radius 0 (new 1-hop patterns)
    - fp[r=2]: Fingerprint with radius 2 MINUS radius 1 (new 2-hop patterns)

    This ensures each radius block contains ONLY the substructures unique to that radius.

    Args:
        smiles: SMILES string
        max_radius: Maximum radius (e.g., 2)
        n_bits_per_radius: Bits allocated per radius (e.g., 512)
        use_chirality: Include chirality information
        use_count: Use count fingerprints instead of binary

    Returns:
        Concatenated array of shape [(max_radius+1) * n_bits_per_radius]
        with radius-specific delta fingerprints
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    # Generate fingerprints for each radius level
    fps = []
    for r in range(max_radius + 1):
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=r, fpSize=n_bits_per_radius)
        if use_count:
            fp = mfpgen.GetCountFingerprint(mol)
        else:
            fp = mfpgen.GetFingerprint(mol)
        arr = np.zeros((n_bits_per_radius,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)

    # Compute deltas: fp[r] - fp[r-1] (set negative values to 0)
    deltas = []
    deltas.append(fps[0])  # Radius 0: keep as-is

    for r in range(1, max_radius + 1):
        delta = fps[r] - fps[r-1]
        # For binary fingerprints, delta should be {0, 1}
        # For count fingerprints, delta shows new occurrences
        delta = np.maximum(delta, 0)  # Remove negative values
        deltas.append(delta)

    # Concatenate all radius blocks
    return np.concatenate(deltas)


# -----------------------------
# Dataset
# -----------------------------
@dataclass
class Split:
    train: List[int]
    valid: List[int]
    test:  List[int]

class ECFPSubset(Dataset):
    """A light wrapper to apply optional feature transforms at read-time."""
    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        indices: Sequence[int],
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ):
        self.X = X
        self.y = y
        self.idx = torch.tensor(indices, dtype=torch.long)
        self.transform = transform

    def __len__(self) -> int:
        return self.idx.numel()

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        j = self.idx[i].item()
        x = self.X[j]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[j]

class ECFPDataset:
    """
    In-memory ECFP holder for any PyG MoleculeNet dataset.

    Attributes
    ----------
    name : str
        MoleculeNet dataset name (e.g., "BACE", "BBBP", "ClinTox", "Tox21", ...).
    X : torch.FloatTensor [N, D]
        Binary ECFP feature matrix.
    y : torch.FloatTensor [N, T]
        Target matrix (float; use appropriate loss in trainer).
    smiles : List[str]
        SMILES strings aligned with rows of X/y.
    tasks : int
        Number of targets (T).
    split : Split
        Indices for train/valid/test.
    """

    def __init__(
        self,
        name: str,
        root: str = "data",
        radius: int = 2,
        n_bits: int = 2048,
        use_chirality: bool = True,
        use_count: bool = False,
        split_type: str = "scaffold",   # "scaffold" or "random"
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        seed: int = 0,
        target_index: Optional[int] = None,  # if None and multi-task, keep all tasks
        feature_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        device: Optional[torch.device] = None,
        multi_radius: bool = False,  # If True, use radius-delta fingerprints
        n_bits_per_radius: Optional[int] = None  # Bits per radius block (only for multi_radius=True)
    ):
        """
        Build the dataset fully in memory.

        Args:
            multi_radius: If True, generate radius-delta fingerprints where each radius block
                         contains only the substructures unique to that radius.
            n_bits_per_radius: Bits allocated per radius block (e.g., 512).
                              If None and multi_radius=True, uses n_bits // (radius + 1).
        """
        self.name = name
        self.root = root
        self.radius = radius
        self.n_bits = n_bits
        self.use_chirality = use_chirality
        self.use_count = use_count
        self.feature_transform = feature_transform
        self.device = device or torch.device("cpu")
        self.multi_radius = multi_radius

        # Determine bits per radius for multi-radius mode
        if multi_radius:
            if n_bits_per_radius is None:
                n_bits_per_radius = n_bits // (radius + 1)
            self.n_bits_per_radius = n_bits_per_radius
            # Create RadiusSchema for the block structure
            blocks = [(r * n_bits_per_radius, (r + 1) * n_bits_per_radius)
                      for r in range(radius + 1)]
            self.radius_schema = RadiusSchema(blocks=blocks)
        else:
            self.n_bits_per_radius = None
            self.radius_schema = None

        # 1) Load MoleculeNet via PyG
        pyg_ds = MoleculeNet(root=root, name=name)
        self.smiles: List[str] = list(pyg_ds.smiles)
        # Collect targets; many datasets are single-task, some are multi-task
        ys = []
        for data in pyg_ds:
            y_i = data.y
            y_i = y_i.view(-1).float()  # [T]
            ys.append(y_i.numpy())
        y_mat = np.stack(ys, axis=0)  # [N, T]
        self.tasks = y_mat.shape[1]

        # If a single task is requested, slice it to shape [N, 1]
        if target_index is not None:
            if not (0 <= target_index < self.tasks):
                raise ValueError(f"target_index {target_index} out of range [0, {self.tasks-1}] for {name}")
            y_mat = y_mat[:, [target_index]]

        # 2) Compute ECFPs in memory
        if multi_radius:
            # Generate radius-delta fingerprints
            X_np = np.stack(
                [morgan_ecfp_radius_deltas(
                    s,
                    max_radius=radius,
                    n_bits_per_radius=n_bits_per_radius,
                    use_chirality=use_chirality,
                    use_count=use_count
                ) for s in self.smiles],
                axis=0
            )  # [N, (radius+1) * n_bits_per_radius]
        else:
            # Standard single-radius fingerprints
            X_np = np.stack(
                [morgan_ecfp_bits(
                    s,
                    radius=radius,
                    n_bits=n_bits,
                    use_chirality=use_chirality,
                    use_count=use_count
                ) for s in self.smiles],
                axis=0
            )  # [N, D]

        # 3) Torch tensors
        self.X = torch.from_numpy(X_np).to(torch.float32).to(self.device)    # [N, D]
        self.y = torch.from_numpy(y_mat).to(torch.float32).to(self.device)   # [N, T’]

        # 4) Build split
        if split_type == "scaffold":
            tr, va, te = scaffold_split_indices(self.smiles, frac_train, frac_valid, seed)
        elif split_type == "random":
            tr, va, te = random_split_indices(len(self.smiles), frac_train, frac_valid, seed)
        else:
            raise ValueError("split_type must be 'scaffold' or 'random'")
        self.split = Split(train=tr, valid=va, test=te)

    def apply_transform(self, pipeline):
        """
        Apply a fitted or stateless pipeline to all features in-place.
        If the pipeline has a fit() method, it will be called first on the data.
        """
        # Fit the pipeline on the training data if it has a fit() method
        if hasattr(pipeline, 'fit'):
            pipeline.fit(self.X)

        # Apply transformation to all samples
        self.X = torch.stack([pipeline(x) for x in self.X], dim=0)
        return self

    @property
    def features(self):
        return self.X

    # ---------- Convenience ----------
    def get_subset(
        self, part: str
    ) -> ECFPSubset:
        if part == "train":
            idx = self.split.train
        elif part == "valid":
            idx = self.split.valid
        elif part == "test":
            idx = self.split.test
        else:
            raise ValueError("part must be 'train', 'valid', or 'test'")
        return ECFPSubset(self.X, self.y, idx, transform=self.feature_transform)

    def make_loader(
        self,
        part: str,
        batch_size: int = 256,
        shuffle: Optional[bool] = None,
        drop_last: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> DataLoader:
        ds = self.get_subset(part)
        if shuffle is None:
            shuffle = (part == "train")
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=num_workers, pin_memory=pin_memory
        )

    def class_balance(self, part: str = "train") -> Tuple[int, int]:
        """For binary single-task classification: returns (#neg, #pos) in split."""
        ds = self.get_subset(part)
        y = ds.y[ds.idx][:, 0] if ds.y.dim() == 2 else ds.y[ds.idx]
        y = y.cpu().numpy()
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        return n_neg, n_pos
    
class TransformedDataset(Dataset):
    def __init__(self, base_ds, transform):
        self.base = base_ds
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]  # assumes base returns (x, y)
        return self.transform(x, idx=i), y

@dataclass(frozen=True)
class RadiusSchema:
    # list of (start, end) indices for each radius block within the full vector
    blocks: list  # e.g., [(0,512), (512,1024), (1024,1536)]

    def slices(self):
        return [slice(a,b) for (a,b) in self.blocks]