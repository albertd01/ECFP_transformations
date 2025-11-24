from dataset_utils import ECFPDataset
from downstream import run_downstream_task
import torch
from transforms import Compose, Rotation, Permutation
import numpy as np
from plots import plot_norms
'''bace = ECFPDataset(name="BACE", split_type="scaffold", target_index=0, n_bits=2048, radius=2)
result_bace = run_downstream_task(bace, task_type="classification", hidden_dim=128, epochs=100, lr=1e-3, device="cpu")
print(result_bace) '''

def random_rotation_matrix(dim: int = 2048) -> torch.Tensor:
    """
    Returns a random orthogonal rotation matrix Q ∈ R^{dim×dim}
    such that QᵀQ = I.
    """
    g = torch.Generator()
    # Sample a random Gaussian matrix
    A = torch.randn(dim, dim, generator=g)
    Q, R = torch.linalg.qr(A)
    d = torch.sign(torch.diagonal(R))
    Q = Q * d
    return Q  # shape [dim, dim]


np.random.seed(1)
torch.manual_seed(1)

g = torch.Generator()
perm = torch.randperm(2048, generator=g)
permutation = Permutation(perm)

Q = random_rotation_matrix(2048)
rotation = Rotation(Q)


pipeline = Compose([rotation, permutation]) 

original_dataset = ECFPDataset(name="esol", split_type="random", target_index=0, n_bits=2048, radius=2, use_count=False)

transformed_dataset = ECFPDataset(name="esol", split_type="random", target_index=0, n_bits=2048, radius=2, use_count=False)
transformed_dataset.apply_transform(pipeline)

result_esol = run_downstream_task(transformed_dataset, task_type="regression", hidden_dim=128, epochs=100, lr=1e-3, device="cpu")
print(result_esol) 

