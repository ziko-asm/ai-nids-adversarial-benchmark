"""
Integrated Gradients (Sundararajan et al., 2017) for the detectors.

attribution_i = (x_i - x'_i) * mean_{k} dF(x' + k/m (x - x')) / dx_i

F is the "attack" logit, the baseline x' is the mean benign training record,
so attributions explain "why is this record more suspicious than typical normal
traffic". Completeness (sum of attributions = F(x) - F(x')) is checked and
reported, which is the standard sanity test for IG.
"""

from __future__ import annotations

import numpy as np
import torch


def integrated_gradients(model, X: torch.Tensor, baseline: torch.Tensor, steps: int = 32,
                         batch: int = 1024) -> tuple[np.ndarray, float]:
    """Returns (attributions [N, F], mean relative completeness error)."""
    model.eval()
    attrs, errs = [], []
    alphas = torch.linspace(0, 1, steps + 1)[1:]
    for i in range(0, len(X), batch):
        x = X[i:i + batch]
        diff = x - baseline
        grads = torch.zeros_like(x)
        for a in alphas:
            z = (baseline + a * diff).requires_grad_(True)
            out = model(z)[:, 1] - model(z)[:, 0]
            (g,) = torch.autograd.grad(out.sum(), z)
            grads += g
        attr = diff * grads / steps
        with torch.no_grad():
            fx = model(x)[:, 1] - model(x)[:, 0]
            fb = model(baseline.unsqueeze(0))[:, 1] - model(baseline.unsqueeze(0))[:, 0]
            gap = fx - fb
            err = (attr.sum(1) - gap).abs() / gap.abs().clamp(min=1e-6)
        attrs.append(attr.detach().numpy())
        errs.append(err.numpy())
    return np.concatenate(attrs), float(np.median(np.concatenate(errs)))


def top_features(attr: np.ndarray, names: list, k: int = 12) -> list:
    imp = np.abs(attr).mean(0)
    share = imp / max(imp.sum(), 1e-12)
    order = np.argsort(-imp)[:k]
    return [{"feature": names[i], "share": round(float(share[i]), 4),
             "mean_signed": round(float(attr[:, i].mean()), 4)} for i in order]
