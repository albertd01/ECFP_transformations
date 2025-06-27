import torch
import torch.nn.functional as F
import torch.nn as nn

class NGFAdapter(nn.Module):
    def __init__(self, model, mode="custom"):
        super().__init__()
        self.model = model
        self.mode = mode  # "custom" or "pytorch_geometric"

    def forward(self, data):
        if self.mode == "custom":
            return self.model(data)  # your implementation handles batching
        elif self.mode == "pytorch_geometric":
            # PyG's NeuralFingerprint returns a list of node outputs
            # Aggregate into graph-level embeddings using global pooling
            x, edge_index, batch = data.x, data.edge_index, data.batch
            return self.model(x, edge_index, batch)
        else:
            raise ValueError(f"Unsupported NGF mode: {self.mode}")