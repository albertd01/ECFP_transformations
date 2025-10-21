import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
import torch
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from dataset_utils import ECFPDataset

from models.MLPClassifier import MLPClassifier
from models.MLPRegressor import MLPRegressor

from sklearn.model_selection import train_test_split

def run_downstream_task(
    ds,                    # <-- ECFPDataset instance
    task_type,             # "classification" or "regression"
    hidden_dim: int = 128,
    batch_size: int = 256,
    epochs: int = 200,
    lr: float = 1e-3,
    device: str = "cpu"
):
    """
    Uses ds.split (supports scaffold or random) and ds.make_loader to get train/val/test.
    Trains a simple MLP and reports VAL/TEST metric.
    """
    input_dim = ds.X.shape[1]
    if task_type == "regression":
        metric_name = "RMSE"
        model = MLPRegressor(input_dim=input_dim, hidden_dim=hidden_dim)
    else:
        metric_name = "ROC_AUC"
        model = MLPClassifier(input_dim=input_dim, hidden_dim=hidden_dim)

    train_loader = ds.make_loader("train", batch_size=batch_size, shuffle=True)
    valid_loader = ds.make_loader("valid", batch_size=1024, shuffle=False)
    test_loader  = ds.make_loader("test",  batch_size=1024, shuffle=False)

    val_score, test_score = train_model(
        model, train_loader, valid_loader, test_loader, task_type, epochs=epochs, lr=lr, device=device,
        # (optional) you can pass class weighting info from ds here if desired
        pos_weight=_compute_pos_weight(ds) if task_type == "classification" else None
    )

    return {"metric": metric_name, "val": val_score, "test": test_score}

def _compute_pos_weight(ds):
    """For binary single-task classification: pos_weight = N_neg / N_pos on the TRAIN split."""
    try:
        n_neg, n_pos = ds.class_balance("train")
        if n_pos == 0:
            return None
        return torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32)
    except Exception:
        return None

def train_model(
    model, train_loader, valid_loader, test_loader,
    task_type, epochs=200, lr=1e-3, device="cpu", pos_weight=None
):
    import torch
    import torch.nn as nn
    import numpy as np

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if task_type == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)

    best_val = -np.inf if task_type == "classification" else np.inf
    best_state = None

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            out = model(xb)  # could be [B,1] (binary) or [B,T] (multi-task)

            if task_type == "classification":
                # ---- make shapes match ----
                if out.dim() == 2 and out.size(-1) == 1:
                    out = out.squeeze(-1)            # [B]
                yb = yb.float()
                if yb.dim() > 1 and yb.size(-1) == 1:
                    yb = yb.squeeze(-1)              # [B]
                loss = criterion(out, yb)
            else:
                # regression
                if out.dim() == 2 and out.size(-1) == 1:
                    out = out.squeeze(-1)
                if yb.dim() == 2 and yb.size(-1) == 1:
                    yb = yb.squeeze(-1)
                loss = criterion(out, yb)

            loss.backward()
            optimizer.step()

        # validate
        val_score = _evaluate(model, valid_loader, task_type, device)
        if task_type == "classification":
            if val_score > best_val:
                best_val = val_score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            if val_score < best_val:
                best_val = val_score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val = _evaluate(model, valid_loader, task_type, device)
    final_test = _evaluate(model, test_loader, task_type, device)
    return final_val, final_test


def _evaluate(model, loader, task_type: str, device: str = "cpu"):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds = model(xb).detach().cpu()

            ys.append(yb.detach().cpu().numpy())
            if task_type == "classification":
                ps.append(torch.sigmoid(preds).numpy())
            else:
                ps.append(preds.numpy())

    y_true = np.concatenate(ys, axis=0)
    y_out  = np.concatenate(ps, axis=0)

    if y_true.ndim == 2 and y_true.shape[1] == 1:
        y_true = y_true[:, 0]
    if y_out.ndim == 2 and y_out.shape[1] == 1:
        y_out = y_out[:, 0]

    if task_type == "classification":
        if y_true.ndim == 2:
            aucs = []
            for t in range(y_true.shape[1]):
                ut = np.unique(y_true[:, t])
                if len(ut) < 2:
                    continue
                aucs.append(roc_auc_score(y_true[:, t], y_out[:, t]))
            return float(np.mean(aucs)) if aucs else float("nan")
        else:
            if len(np.unique(y_true)) < 2:
                return float("nan")
            return roc_auc_score(y_true, y_out)

    else:
        if y_true.ndim == 1 and y_out.ndim == 1:
            return root_mean_squared_error(y_true, y_out)
        if y_true.ndim == 1: y_true = y_true[:, None]
        if y_out.ndim == 1:  y_out  = y_out[:, None]
        rmses = [root_mean_squared_error(y_true[:, t], y_out[:, t])
                 for t in range(y_true.shape[1])]
        return float(np.mean(rmses))
