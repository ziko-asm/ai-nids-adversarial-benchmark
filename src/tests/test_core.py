"""Unit tests on a small synthetic KDD-shaped dataset (the real file is not in git)."""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from src.attack import ThreatModel, attack_malicious, fgsm, pgd
from src.dataset import FEATURES, Preprocessor, prepare, synthetic_kdd
from src.explain import integrated_gradients
from src.models import NEURAL_MODELS, build_model
from src.utils import evaluate, train_model


@pytest.fixture(scope="module")
def splits():
    return prepare(synthetic_kdd(600, seed=1), seed=0)


@pytest.fixture(scope="module")
def mlp(splits):
    return train_model("mlp", splits.X_train, splits.y_train, splits.X_val, splits.y_val,
                       seed=0, verbose=False, epochs=15, batch_size=64)


def test_duplicates_removed():
    df = synthetic_kdd(200)
    s = prepare(pd.concat([df, df], ignore_index=True), seed=0)
    assert s.stats["rows_raw"] == 400 and s.stats["rows_unique"] == 200


def test_splits_disjoint_and_stratified(splits):
    n = len(splits.y_train) + len(splits.y_val) + len(splits.y_test)
    assert n == splits.stats["rows_used"]
    for y in (splits.y_train, splits.y_val, splits.y_test):
        assert 0 < y.mean() < 1


def test_preprocessor_roundtrip(splits):
    p2 = Preprocessor.from_dict(splits.prep.to_dict())
    df = synthetic_kdd(5, seed=3)
    assert np.allclose(splits.prep.transform(df), p2.transform(df))
    assert splits.X_train.shape[1] == len(splits.prep.feature_names)


def test_mutable_mask_freezes_categorical(splits):
    mask = splits.prep.mutable_mask()
    names = splits.prep.feature_names
    assert all(mask[i] == 0 for i, n in enumerate(names) if n.startswith(("protocol_type=", "service=", "flag=")))
    assert mask[names.index("src_bytes")] == 1


@pytest.mark.parametrize("name", NEURAL_MODELS)
def test_forward(name, splits):
    m = build_model(name, splits.X_train.shape[1]).eval()
    out = m(torch.as_tensor(splits.X_test[:7]))
    assert out.shape == (7, 2)


def test_model_learns(mlp, splits):
    assert evaluate(mlp, torch.as_tensor(splits.X_test), splits.y_test)["f1"] > 0.8


def test_fgsm_increases_loss(mlp, splits):
    x, y = torch.as_tensor(splits.X_test), torch.as_tensor(splits.y_test)
    th = ThreatModel.unconstrained(x.shape[1])
    x_adv = fgsm(mlp, x, y, 0.5, th)
    with torch.no_grad():
        assert F.cross_entropy(mlp(x_adv), y) > F.cross_entropy(mlp(x), y)


def test_pgd_respects_threat_model(mlp, splits):
    x, y = torch.as_tensor(splits.X_test), torch.as_tensor(splits.y_test)
    th = ThreatModel.constrained(splits.prep)
    eps = 0.3
    x_adv = pgd(mlp, x, y, eps, th, steps=10)
    d = (x_adv - x).abs()
    assert d.max() <= eps + 1e-5
    frozen = torch.as_tensor(splits.prep.mutable_mask()) == 0
    assert d[:, frozen].max() == 0
    lo, hi = torch.min(th.lower, x), torch.max(th.upper, x)
    assert (x_adv >= lo - 1e-5).all() and (x_adv <= hi + 1e-5).all()


def test_only_malicious_rows_change(mlp, splits):
    x, y = torch.as_tensor(splits.X_test), torch.as_tensor(splits.y_test)
    x_adv = attack_malicious(mlp, x, y, "pgd", 0.3, ThreatModel.unconstrained(x.shape[1]), steps=5)
    changed = (x_adv != x).any(1)
    assert not changed[y == 0].any()


def test_integrated_gradients_completeness(mlp, splits):
    x = torch.as_tensor(splits.X_test[:50])
    base = torch.as_tensor(splits.X_train[splits.y_train == 0].mean(0))
    attr, err = integrated_gradients(mlp, x, base, steps=64)
    assert attr.shape == x.shape
    assert err < 0.1


def test_feature_list_is_kdd():
    assert len(FEATURES) == 41
