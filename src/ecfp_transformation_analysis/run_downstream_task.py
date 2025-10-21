from dataset_utils import ECFPDataset
from downstream import run_downstream_task
'''bace = ECFPDataset(name="BACE", split_type="scaffold", target_index=0, n_bits=2048, radius=2)
result_bace = run_downstream_task(bace, task_type="classification", hidden_dim=128, epochs=100, lr=1e-3, device="cpu")
print(result_bace) '''


esol = ECFPDataset(name="bace", split_type="random", target_index=0, n_bits=2048, radius=2)
result_esol = run_downstream_task(esol, task_type="classification", hidden_dim=128, epochs=100, lr=1e-3, device="cpu")
print(result_esol) 