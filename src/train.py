"""
Train one detector and save a self-contained checkpoint (weights + preprocessing).

    python -m src.train --model mlp
    python -m src.train --model cnn --adversarial      # PGD adversarial training

The checkpoint is what src/api.py serves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.attack import ThreatModel
from src.dataset import DEFAULT_PATH, load_splits
from src.models import NEURAL_MODELS
from src.utils import HPARAMS, evaluate, train_model


def save_checkpoint(model, name: str, prep, path: Path, adversarial: bool, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": name, "adversarial": adversarial, "state_dict": model.state_dict(),
                "preprocessor": prep.to_dict(), "test_metrics": metrics}, path)


def main():
    p = argparse.ArgumentParser(description="Train an intrusion detector on KDD Cup 99")
    p.add_argument("--model", default="mlp", choices=NEURAL_MODELS)
    p.add_argument("--adversarial", action="store_true", help="PGD adversarial training")
    p.add_argument("--data", default=str(DEFAULT_PATH))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=HPARAMS["epochs"])
    p.add_argument("--out", default="checkpoints")
    a = p.parse_args()

    s = load_splits(a.data, seed=a.seed)
    print(f"[*] {s.stats['rows_raw']:,} raw rows -> {s.stats['rows_unique']:,} unique, "
          f"{s.stats['attack_share']:.1%} attacks, {s.stats['n_features']} model features")
    print(f"[*] Training {a.model.upper()}{' with PGD adversarial training' if a.adversarial else ''}")
    model = train_model(a.model, s.X_train, s.y_train, s.X_val, s.y_val, seed=a.seed,
                        adversarial=a.adversarial, threat=ThreatModel.unconstrained(s.X_train.shape[1]),
                        epochs=a.epochs)
    m = evaluate(model, torch.as_tensor(s.X_test), s.y_test)
    print(f"[*] Test: acc {m['accuracy']:.4f} | F1 {m['f1']:.4f} | recall {m['recall']:.4f} | AUC {m['roc_auc']:.4f}")
    path = Path(a.out) / f"{a.model}{'_adv' if a.adversarial else ''}.pt"
    save_checkpoint(model, a.model, s.prep, path, a.adversarial, m)
    print(f"[*] Saved -> {path}")


if __name__ == "__main__":
    main()
