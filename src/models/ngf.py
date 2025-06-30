import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from sparsemax import Sparsemax
import numpy as np
from models.mlp import MLPClassifier, MLPRegressor


# === SUM functions ===
def sum_default(x, edge_index, edge_attr=None):
    row, col = edge_index
    neigh_sum = torch.zeros_like(x)
    neigh_sum = neigh_sum.index_add(0, row, x[col])
    return neigh_sum


def sum_bond_weighted(x, edge_index, edge_attr):
    row, col = edge_index
    weights = edge_attr[:, 0].unsqueeze(1) if edge_attr is not None else 1.0
    weighted = x[col] * weights
    neigh_sum = torch.zeros_like(x)
    neigh_sum = neigh_sum.index_add(0, row, weighted)
    return neigh_sum

SUM_FUNCTIONS = {
    "default": sum_default,
    "bond_weighted": sum_bond_weighted,
}

# === SMOOTH functions ===

SMOOTH_FUNCTIONS = {
    "tanh": torch.tanh,
    "relu": F.relu,
    "identity": lambda x: x,
    "sin" : lambda x: torch.sin(x),
}

# === SPARSIFY functions ===

SPARSIFY_FUNCTIONS = {
    "softmax": lambda x: F.softmax(x, dim=1),
    "gumbel": lambda x: F.gumbel_softmax(x, tau=0.5, hard=True, dim=-1),
    "sparsemax": lambda x: Sparsemax(dim=1)(x)
}


class NeuralGraphFingerprint(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim,
        fingerprint_dim,
        num_layers=3,
        weight_scale=5.0,
        max_degree: int = 5,          
        sum_fn=None,
        smooth_fn=None,
        sparsify_fn=None
    ):
        super().__init__()
        self.num_layers = num_layers
        self.weight_scale = weight_scale
        self.fingerprint_dim = fingerprint_dim
        self.max_degree = max_degree

        self.sum_fn    = SUM_FUNCTIONS[sum_fn]    or self.default_sum
        self.smooth_fn = SMOOTH_FUNCTIONS[smooth_fn] or torch.tanh
        self.sparsify_fn = SPARSIFY_FUNCTIONS[sparsify_fn] or (lambda x: F.softmax(x, dim=1))

        self.W_self  = nn.ModuleList()
        self.W_neigh = nn.ModuleList()
        self.W_fp    = nn.ModuleList()

        for layer in range(num_layers):
            in_dim = in_channels if layer == 0 else hidden_dim
            # self‐update
            self.W_self.append(nn.Linear(in_dim, hidden_dim, bias=True))
            # neighbor updates: one linear per degree 0..max_degree
            ddict = nn.ModuleDict({
                str(d): nn.Linear(in_dim, hidden_dim, bias=False)
                for d in range(self.max_degree+1)
            })
            self.W_neigh.append(ddict)
            # fingerprint projection
            self.W_fp.append(nn.Linear(hidden_dim, fingerprint_dim, bias=False))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in range(self.num_layers):
            nn.init.normal_(self.W_self[layer].weight, mean=0.0, std=self.weight_scale)
            nn.init.zeros_(self.W_self[layer].bias)
            for d, lin in self.W_neigh[layer].items():
                nn.init.normal_(lin.weight, mean=0.0, std=self.weight_scale)
            nn.init.normal_(self.W_fp[layer].weight, mean=0.0, std=self.weight_scale)

    def default_sum(self, x, edge_index, edge_attr=None):
        row, col = edge_index
        neigh_sum = torch.zeros_like(x)
        neigh_sum.index_add_(0, col, x[row])
        return neigh_sum
    
    def frequency_adjusted_sin(self,x):
        return torch.sin((1/self.weight_scale)*x)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        row, col = edge_index
        N = x.size(0)

        # compute degree of each source node
        deg_source = torch.zeros(N, dtype=torch.long, device=x.device)
        deg_source.index_add_(0, row, torch.ones_like(row, dtype=torch.long))

        fingerprint = x.new_zeros(batch.max().item()+1, self.fingerprint_dim)

        for layer in range(self.num_layers):
            # self‐contribution
            h_self = self.W_self[layer](x)

            neigh_contrib = torch.zeros_like(h_self)
            for d_str, lin in self.W_neigh[layer].items():
                d = int(d_str)
                mask = (deg_source[row] == d)
                if mask.any():
                    row_d = row[mask]
                    col_d = col[mask]
                    msgs = torch.zeros_like(x)
                    msgs.index_add_(0, col_d, x[row_d])
                    neigh_contrib += lin(msgs)

            # combine & smooth
            h = h_self + neigh_contrib
            h = self.smooth_fn(h)

            # fingerprint projection + pooling
            contrib = self.sparsify_fn(self.W_fp[layer](h))
            fingerprint += global_add_pool(contrib, batch)

            x = h

        return fingerprint

class NGFWithHead(nn.Module):
    def __init__(self, ngf_base, task_type="regression", hidden_dim=128, mode="from_scratch"):
        super().__init__()
        self.ngf = ngf_base
        self.task_type = task_type
        self.mode = mode  

        input_dim = getattr(ngf_base, "fingerprint_dim", None) or getattr(ngf_base, "out_channels", None)
        if input_dim is None:
            raise ValueError("Could not infer input_dim from NGF model")

        if task_type == "regression":
            self.head = MLPRegressor(input_dim=input_dim, hidden_dim=hidden_dim)
        elif task_type == "classification":
            self.head = MLPClassifier(input_dim=input_dim, hidden_dim=hidden_dim)
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

    def forward(self, data):
        if self.mode == "from_scratch":
            x = self.ngf(data)
        elif self.mode == "pytorch_geometric":
            x = self.ngf(data)  # PyG model expects a Data object
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return self.head(x)