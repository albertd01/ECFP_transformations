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
    use_chirality: bool = True
) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = mfpgen.GetFingerprint(mol)

    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


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
        split_type: str = "scaffold",   # "scaffold" or "random"
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        seed: int = 0,
        target_index: Optional[int] = None,  # if None and multi-task, keep all tasks
        feature_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Build the dataset fully in memory.
        """
        self.name = name
        self.root = root
        self.radius = radius
        self.n_bits = n_bits
        self.use_chirality = use_chirality
        self.feature_transform = feature_transform
        self.device = device or torch.device("cpu")

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
        X_np = np.stack(
            [morgan_ecfp_bits(s, radius=radius, n_bits=n_bits, use_chirality=use_chirality)
             for s in self.smiles],
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