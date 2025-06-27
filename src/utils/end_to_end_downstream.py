import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error, roc_auc_score
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

def run_end_to_end_training(model, dataset, task_type, epochs=150, lr=1e-3, batch_size=64, test_size=0.2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    indices = np.arange(len(dataset))
    labels = np.array([data.y.item() for data in dataset])
    train_idx, val_idx = train_test_split(
        indices, test_size=test_size, random_state=42, stratify=labels if task_type == "classification" else None
    )

    train_subset = [dataset[i] for i in train_idx]
    val_subset = [dataset[i] for i in val_idx]

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            out = model(batch)
            y = batch.y.view(-1).to(device)
            loss = compute_loss(out, y, task_type)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # === Evaluation ===
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch)
            preds.append(out.view(-1).cpu())         
            targets.append(batch.y.view(-1).cpu()) 

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    if task_type == "regression":
        score = root_mean_squared_error(targets.numpy(), preds.numpy())
    else:
        preds = torch.sigmoid(preds).numpy()
        targets = targets.numpy()
        score = roc_auc_score(targets, preds)
    return score, model  

def compute_loss(preds, targets, task_type):
    if task_type == "regression":
        return F.mse_loss(preds, targets)
    else:
        return F.binary_cross_entropy_with_logits(preds.view(-1), targets.float())
