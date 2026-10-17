"""
Evasion attacks on the detector's input space.

Only malicious records are perturbed: the attacker controls its own attack
traffic and wants it classified as normal. Benign records are left untouched.

Two threat models:
  unconstrained - every input feature may move by up to eps (in units of the
                  feature's standard deviation). An upper bound on damage, the
                  setting most papers report.
  constrained   - protocol, service, TCP flag and the victim-side binary flags
                  cannot change, and every feature stays inside the range seen
                  in training data. Closer to what a real attacker can do.

FGSM: single step (Goodfellow et al., 2015).
PGD : iterated, random start, projection onto the eps-ball (Madry et al., 2018).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class ThreatModel:
    def __init__(self, n_features: int, mask=None, lower=None, upper=None):
        self.mask = torch.ones(n_features) if mask is None else torch.as_tensor(mask, dtype=torch.float32)
        self.lower = None if lower is None else torch.as_tensor(lower, dtype=torch.float32)
        self.upper = None if upper is None else torch.as_tensor(upper, dtype=torch.float32)

    @classmethod
    def unconstrained(cls, n_features: int) -> "ThreatModel":
        return cls(n_features)

    @classmethod
    def constrained(cls, prep) -> "ThreatModel":
        return cls(len(prep.feature_names), prep.mutable_mask(), prep.lower, prep.upper)

    def project(self, x_adv: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
        delta = (x_adv - x).clamp(-eps, eps) * self.mask
        out = x + delta
        if self.lower is not None:
            # stay inside the range seen in training, but never move a feature
            # the attacker does not control and never push x itself further out
            lo = torch.min(self.lower, x)
            hi = torch.max(self.upper, x)
            out = torch.max(torch.min(out, hi), lo)
        return out


def _grad_wrt_input(model, x, target):
    x = x.clone().requires_grad_(True)
    loss = F.cross_entropy(model(x), target)
    (g,) = torch.autograd.grad(loss, x)
    return g


def fgsm(model, x, y, eps: float, threat: ThreatModel) -> torch.Tensor:
    if eps <= 0:
        return x
    model.eval()
    g = _grad_wrt_input(model, x, y)
    return threat.project(x + eps * g.sign(), x, eps).detach()


def pgd(model, x, y, eps: float, threat: ThreatModel, steps: int = 20,
        alpha: float | None = None, random_start: bool = True, seed: int = 0) -> torch.Tensor:
    if eps <= 0:
        return x
    model.eval()
    alpha = alpha if alpha is not None else 2.5 * eps / steps
    x_adv = x.clone()
    if random_start:
        gen = torch.Generator().manual_seed(seed)
        x_adv = threat.project(x + (torch.rand(x.shape, generator=gen) * 2 - 1) * eps, x, eps)
    for _ in range(steps):
        g = _grad_wrt_input(model, x_adv, y)
        x_adv = threat.project(x_adv + alpha * g.sign(), x, eps).detach()
    return x_adv


ATTACKS = {"fgsm": fgsm, "pgd": pgd}


def attack_malicious(model, X: torch.Tensor, y: torch.Tensor, attack: str, eps: float,
                     threat: ThreatModel, batch: int = 4096, **kw) -> torch.Tensor:
    """Perturb only rows with y == 1; return the full (mixed) test matrix."""
    X_out = X.clone()
    idx = torch.where(y == 1)[0]
    fn = ATTACKS[attack]
    for i in range(0, len(idx), batch):
        b = idx[i:i + batch]
        X_out[b] = fn(model, X[b], y[b], eps, threat, **kw)
    return X_out
