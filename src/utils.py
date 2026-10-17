"""Training loop, metrics and seeding shared by every script."""

from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from src.attack import ThreatModel, pgd
from src.models import build_model

HPARAMS = dict(epochs=12, batch_size=512, lr=1e-3, weight_decay=1e-5, patience=4,
               adv_eps=0.25, adv_steps=7)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def metrics_from_scores(y_true: np.ndarray, p_attack: np.ndarray) -> dict:
    pred = (p_attack >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y_true, p_attack)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "roc_auc": float(auc),
    }


@torch.no_grad()
def predict_proba(model, X: torch.Tensor, batch: int = 8192) -> np.ndarray:
    model.eval()
    out = [torch.softmax(model(X[i:i + batch]), -1)[:, 1] for i in range(0, len(X), batch)]
    return torch.cat(out).numpy()


def evaluate(model, X: torch.Tensor, y: np.ndarray) -> dict:
    return metrics_from_scores(y, predict_proba(model, X))


def train_model(name: str, Xtr, ytr, Xva, yva, seed: int = 0, adversarial: bool = False,
                threat: ThreatModel | None = None, verbose: bool = True, **hp):
    """Train a neural detector. With `adversarial=True`, every batch's malicious
    rows are replaced by PGD examples (Madry-style adversarial training)."""
    h = {**HPARAMS, **hp}
    set_seed(seed)
    Xtr_t, ytr_t = torch.as_tensor(Xtr), torch.as_tensor(ytr)
    Xva_t = torch.as_tensor(Xva)
    model = build_model(name, Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=h["lr"], weight_decay=h["weight_decay"])
    threat = threat or ThreatModel.unconstrained(Xtr.shape[1])
    gen = torch.Generator().manual_seed(seed)

    best, best_state, bad = -1.0, None, 0
    for epoch in range(1, h["epochs"] + 1):
        perm = torch.randperm(len(Xtr_t), generator=gen)
        total = 0.0
        for i in range(0, len(perm), h["batch_size"]):
            b = perm[i:i + h["batch_size"]]
            xb, yb = Xtr_t[b], ytr_t[b]
            if adversarial:
                mal = yb == 1
                if mal.any():
                    xb = xb.clone()
                    xb[mal] = pgd(model, xb[mal], yb[mal], h["adv_eps"], threat,
                                  steps=h["adv_steps"], seed=seed + epoch + i)
            model.train()
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        score = evaluate(model, Xva_t, yva)["f1"]
        if verbose:
            print(f"    epoch {epoch:2d}  loss {total/len(perm):.4f}  val F1 {score:.4f}")
        if score > best:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= h["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model
