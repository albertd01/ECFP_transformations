import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from sparsemax import Sparsemax
import numpy as np
from models.mlp import MLPClassifier, MLPRegressor
from entmax import entmax15


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
        sum_fn=None,
        smooth_fn=None,
        sparsify_fn=None
    ):
        super().__init__()
        self.num_layers = num_layers
        self.weight_scale = weight_scale
        self.fingerprint_dim = fingerprint_dim

        self.sum_fn = SUM_FUNCTIONS[sum_fn] or self.default_sum
        self.smooth_fn = SMOOTH_FUNCTIONS[smooth_fn] or torch.tanh
        self.sparsify_fn = SPARSIFY_FUNCTIONS[sparsify_fn] or (lambda x: F.softmax(x, dim=1))

        self.W_self = nn.ModuleList()
        self.W_neigh = nn.ModuleList()
        self.W_fp = nn.ModuleList()

        for layer in range(num_layers):
            in_dim = in_channels if layer == 0 else hidden_dim
            self.W_self.append(nn.Linear(in_dim, hidden_dim, bias=True))
            self.W_neigh.append(nn.Linear(in_dim, hidden_dim, bias=True))
            self.W_fp.append(nn.Linear(hidden_dim, fingerprint_dim, bias=False))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in range(self.num_layers):
            nn.init.normal_(self.W_self[layer].weight, mean=0.0, std=self.weight_scale)
            nn.init.normal_(self.W_neigh[layer].weight, mean=0.0, std=self.weight_scale)
            nn.init.zeros_(self.W_self[layer].bias)
            nn.init.zeros_(self.W_neigh[layer].bias)
            nn.init.normal_(self.W_fp[layer].weight, mean=0.0, std=self.weight_scale)

    def default_sum(self, x, edge_index, edge_attr=None):
        row, col = edge_index
        neigh_sum = torch.zeros_like(x)
        neigh_sum = neigh_sum.index_add(0, row, x[col])
        return neigh_sum

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        edge_attr = getattr(data, 'edge_attr', None)

        fingerprint = x.new_zeros(batch.max().item() + 1, self.W_fp[0].out_features)

        for layer in range(self.num_layers):
            neigh_sum = self.sum_fn(x, edge_index, edge_attr)
            x = self.W_self[layer](x) + self.W_neigh[layer](neigh_sum)
            x = self.smooth_fn(x)

            contrib = self.sparsify_fn(self.W_fp[layer](x))
            fingerprint_layer = global_add_pool(contrib, batch)
            fingerprint = fingerprint + fingerprint_layer

        return fingerprint