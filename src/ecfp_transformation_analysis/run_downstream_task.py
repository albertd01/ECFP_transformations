#!/usr/bin/env python3
"""
BACE (MoleculeNet via PyTorch Geometric) + RDKit ECFP + MLP (PyTorch)

- Loads BACE from torch_geometric.datasets.MoleculeNet
- Computes binary ECFP (Morgan) with RDKit
- Scaffold split (Bemis–Murcko) on SMILES
- Trains a simple MLP classifier and reports ROC-AUC

Usage:
  python bace_ecfp_pyg.py --radius 2 --n_bits 2048 --hidden 512 256 --epochs 50
"""

import argparse
import os
import random
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from torch_geometric.datasets import MoleculeNet

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
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

def scaffold_split(
    smiles_list: List[str],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    seed: int = 0,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Deterministic scaffold split:
      - group indices by Murcko scaffold
      - sort scaffolds by descending frequency, then by SMILES string
      - greedily assign to train/valid/test to hit target fractions
    """
    rng = random.Random(seed)
    scaffold_to_indices: Dict[str, List[int]] = {}
    for i, s in enumerate(smiles_list):
        scaf = smiles_to_scaffold(s)
        scaffold_to_indices.setdefault(scaf, []).append(i)

    # sort scaffold bins by size desc, tie-breaker by scaffold string
    bins = sorted(scaffold_to_indices.values(), key=lambda idxs: (-len(idxs), smiles_to_scaffold(smiles_list[idxs[0]])))

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

    # If any remainder due to rounding, push to test
    return train_idx, valid_idx, test_idx

def morgan_ecfp_bits(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, useChirality=True)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    # Convert RDKit ExplicitBitVect to numpy
    Chem.DataStructs.ConvertToNumpyArray(bv, arr)
    return arr.astype(np.float32)

# -----------------------------
# Model
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], dropout: float = 0.1):
        super().__init__()
        dims = [in_dim] + hidden + [1]
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data", help="Data root for PyG MoleculeNet")
    ap.add_argument("--radius", type=int, default=2, help="Morgan radius (2=ECFP4)")
    ap.add_argument("--n_bits", type=int, default=2048, help="Fingerprint length")
    ap.add_argument("--hidden", type=int, nargs="+", default=[512, 256], help="MLP hidden sizes")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # 1) Load BACE from PyG (we'll use its SMILES + labels)
    dataset = MoleculeNet(root=args.root, name="BACE")
    smiles_list = dataset.smiles  # list[str], aligned with dataset indices
    labels_list = [int(d.y.item()) for d in dataset]  # 0/1

    # 2) Compute binary ECFP for each molecule
    print("Computing RDKit ECFP fingerprints...")
    fps = np.stack([morgan_ecfp_bits(s, radius=args.radius, n_bits=args.n_bits) for s in smiles_list], axis=0)
    y = np.array(labels_list, dtype=np.float32)

    # 3) Scaffold split
    tr_idx, va_idx, te_idx = scaffold_split(smiles_list, frac_train=0.8, frac_valid=0.1, seed=args.seed)
    def make_loader(idxs):
        X = torch.tensor(fps[idxs], dtype=torch.float32)
        Y = torch.tensor(y[idxs], dtype=torch.float32)
        ds = TensorDataset(X, Y)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    train_loader = make_loader(tr_idx)
    valid_loader = make_loader(va_idx)
    test_loader  = make_loader(te_idx)

    # 4) Model, optimizer, loss
    model = MLP(in_dim=args.n_bits, hidden=args.hidden, dropout=args.dropout).to(device)

    # class imbalance: pos_weight = N_neg / N_pos computed on TRAIN
    y_tr = y[tr_idx]
    n_pos = max(1, (y_tr == 1).sum())
    n_neg = max(1, (y_tr == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def evaluate(loader) -> float:
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                logits = model(xb)
                prob = torch.sigmoid(logits).cpu().numpy()
                ys.append(yb.numpy())
                ps.append(prob)
        y_true = np.concatenate(ys)
        y_prob = np.concatenate(ps)
        # Guard against edge cases (all same label in a split)
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return roc_auc_score(y_true, y_prob)

    # 5) Train
    best_val, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            running_loss += loss.item() * xb.size(0)

        val_auc = evaluate(valid_loader)
        if val_auc > best_val:
            best_val = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == args.epochs:
            tr_auc = evaluate(train_loader)
            te_auc = evaluate(test_loader)
            print(f"Epoch {epoch:03d} | loss {running_loss/len(tr_idx):.4f} | "
                  f"TR AUC {tr_auc:.4f} | VA AUC {val_auc:.4f} | TE AUC {te_auc:.4f}")

    # 6) Final test with best validation state
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    tr_auc = evaluate(train_loader)
    va_auc = evaluate(valid_loader)
    te_auc = evaluate(test_loader)
    print("\n=== Results (BACE, binary ECFP) ===")
    print(f"Radius: {args.radius}  Bits: {args.n_bits}  Hidden: {args.hidden}  Epochs: {args.epochs}")
    print(f"ROC-AUC  Train: {tr_auc:.4f}  Valid: {va_auc:.4f}  Test: {te_auc:.4f}")

if __name__ == "__main__":
    main()
