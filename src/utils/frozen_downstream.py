import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error, roc_auc_score
import torch

from models.mlp import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split

def run_frozen_downstream_task(ecfp_array, ngf_array, labels, task_type, hidden_dim=128, test_size=0.2):
    # Split data once for both ECFP and NGF
    X_train_ecfp, X_test_ecfp, y_train, y_test = train_test_split(
        ecfp_array, labels, test_size=test_size, random_state=42, stratify=labels if task_type == "classification" else None
    )
    X_train_ngf, X_test_ngf, _, _ = train_test_split(
        ngf_array, labels, test_size=test_size, random_state=42, stratify=labels if task_type == "classification" else None
    )

    if task_type == 'regression':
        ecfp_model = MLPRegressor(input_dim=ecfp_array.shape[1], hidden_dim=hidden_dim)
        ngf_model = MLPRegressor(input_dim=ngf_array.shape[1], hidden_dim=hidden_dim)
    else:
        ecfp_model = MLPClassifier(input_dim=ecfp_array.shape[1], hidden_dim=hidden_dim)
        ngf_model = MLPClassifier(input_dim=ngf_array.shape[1], hidden_dim=hidden_dim)

    score_ecfp = train_model(ecfp_model, X_train_ecfp, y_train, X_test_ecfp, y_test, task_type)
    score_ngf = train_model(ngf_model, X_train_ngf, y_train, X_test_ngf, y_test, task_type)

    return {
        "ecfp": (score_ecfp),
        "ngf": (score_ngf)
    }


def train_model(model, X_train, y_train, X_test, y_test, task_type, epochs=200, lr=1e-3):
    model = model.to('cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32 if task_type == 'regression' else torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32 if task_type == 'regression' else torch.long)

    for _ in range(epochs):
        model.eval()
        optimizer.zero_grad()
        out = model(X_train)
        if task_type == 'regression':
            loss = F.mse_loss(out, y_train)
        else:
            out = out.view(-1)
            loss = F.binary_cross_entropy_with_logits(out, y_train.float())
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        if task_type == 'classification':
            preds = torch.sigmoid(preds).view(-1).numpy()
            score = roc_auc_score(y_test.numpy(), preds)
        else:
            preds = preds.numpy()
            score = root_mean_squared_error(y_test.numpy(), preds) 
    return score


