"""
KDD Cup 1999 (10% subset) loading and preprocessing.

Choices that matter for honest numbers:
  * Exact duplicate rows are removed BEFORE splitting. The raw 10% file has
    494,021 rows but only 145,586 unique ones; without de-duplication the same
    record lands in train and test and accuracy is inflated.
  * Split is stratified train / val / test (60 / 20 / 20) and seeded.
  * Heavy-tailed count and byte features go through log1p; everything is then
    standardised with statistics from the training split only.
  * The three categorical columns are one-hot encoded with categories learned
    on the training split (unseen values map to all-zeros).

The preprocessing state (`Preprocessor`) is saved with each checkpoint so the
API can turn a raw 41-field connection record into the exact model input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PATH = Path("data/kddcup.data_10_percent.csv")

FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
COLUMNS = FEATURES + ["label"]
CATEGORICAL = ["protocol_type", "service", "flag"]
BINARY = ["land", "logged_in", "root_shell", "su_attempted", "is_host_login", "is_guest_login"]
# features a remote attacker cannot change for its own connection without
# changing what the connection IS (protocol, target service, TCP state machine
# outcome) or that are flags set by the victim host
IMMUTABLE = CATEGORICAL + BINARY
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]
LOG_FEATURES = [
    "duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "num_compromised", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "count", "srv_count",
    "dst_host_count", "dst_host_srv_count",
]


def read_raw(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download kddcup.data_10_percent.gz from "
            "http://kdd.ics.uci.edu/databases/kddcup99/ and extract it to data/."
        )
    return pd.read_csv(path, header=None, names=COLUMNS)


def binary_labels(df: pd.DataFrame) -> np.ndarray:
    return (df["label"].astype(str).str.strip() != "normal.").astype(np.int64).to_numpy()


@dataclass
class Preprocessor:
    categories: dict = field(default_factory=dict)   # column -> list of values
    mean: list = field(default_factory=list)
    std: list = field(default_factory=list)
    feature_names: list = field(default_factory=list)  # model input order
    lower: list = field(default_factory=list)          # per-feature min on train (model space)
    upper: list = field(default_factory=list)

    # ------------------------------------------------------------------
    def _raw_matrix(self, df: pd.DataFrame) -> np.ndarray:
        num = df[NUMERIC].astype(np.float64).copy()
        for c in LOG_FEATURES:
            num[c] = np.log1p(num[c].clip(lower=0))
        blocks = [num.to_numpy()]
        for c in CATEGORICAL:
            vals = df[c].astype(str).to_numpy()
            cats = self.categories[c]
            onehot = np.zeros((len(df), len(cats)))
            index = {v: i for i, v in enumerate(cats)}
            for r, v in enumerate(vals):
                j = index.get(v)
                if j is not None:
                    onehot[r, j] = 1.0
            blocks.append(onehot)
        return np.concatenate(blocks, axis=1)

    def fit(self, df: pd.DataFrame) -> Preprocessor:
        self.categories = {c: sorted(df[c].astype(str).unique().tolist()) for c in CATEGORICAL}
        self.feature_names = NUMERIC + [f"{c}={v}" for c in CATEGORICAL for v in self.categories[c]]
        raw = self._raw_matrix(df)
        mu, sd = raw.mean(0), raw.std(0)
        sd[sd < 1e-8] = 1.0
        self.mean, self.std = mu.tolist(), sd.tolist()
        z = (raw - mu) / sd
        self.lower, self.upper = z.min(0).tolist(), z.max(0).tolist()
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        raw = self._raw_matrix(df)
        return ((raw - np.array(self.mean)) / np.array(self.std)).astype(np.float32)

    def mutable_mask(self) -> np.ndarray:
        """1 for features the attacker may change in the constrained threat model."""
        base = {f.split("=")[0] for f in IMMUTABLE}
        return np.array([0.0 if n.split("=")[0] in base else 1.0 for n in self.feature_names],
                        dtype=np.float32)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Preprocessor:
        return cls(**d)


@dataclass
class Splits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    prep: Preprocessor
    attack_types_test: np.ndarray
    stats: dict


def _stratified_indices(y: np.ndarray, rng: np.random.Generator, train=0.6, val=0.2):
    tr, va, te = [], [], []
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        n_tr, n_va = round(len(idx) * train), round(len(idx) * val)
        tr.append(idx[:n_tr])
        va.append(idx[n_tr:n_tr + n_va])
        te.append(idx[n_tr + n_va:])
    return (rng.permutation(np.concatenate(tr)), rng.permutation(np.concatenate(va)),
            rng.permutation(np.concatenate(te)))


def prepare(df: pd.DataFrame, seed: int = 0, max_rows: int | None = None) -> Splits:
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_unique = len(df)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed).reset_index(drop=True)
    y = binary_labels(df)
    rng = np.random.default_rng(seed)
    tr, va, te = _stratified_indices(y, rng)
    prep = Preprocessor().fit(df.iloc[tr])
    X = prep.transform(df)
    stats = {"rows_raw": n_raw, "rows_unique": n_unique, "rows_used": len(df),
             "attack_share": float(y.mean()), "n_features": X.shape[1],
             "n_mutable": int(prep.mutable_mask().sum())}
    return Splits(X[tr], y[tr], X[va], y[va], X[te], y[te], prep,
                  df["label"].astype(str).str.strip().str.rstrip(".").to_numpy()[te], stats)


def load_splits(path: str | Path = DEFAULT_PATH, seed: int = 0,
                max_rows: int | None = None) -> Splits:
    return prepare(read_raw(path), seed=seed, max_rows=max_rows)


def synthetic_kdd(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Tiny KDD-shaped frame for tests and CI (the real file is not in the repo)."""
    rng = np.random.default_rng(seed)
    attack = rng.random(n) < 0.4
    df = pd.DataFrame({c: rng.random(n) for c in NUMERIC})
    for c in LOG_FEATURES:
        df[c] = rng.integers(0, 1000, n)
    for c in BINARY:
        df[c] = rng.integers(0, 2, n)
    df["src_bytes"] = np.where(attack, rng.integers(0, 50, n), rng.integers(100, 5000, n))
    df["serror_rate"] = np.where(attack, rng.uniform(0.6, 1.0, n), rng.uniform(0, 0.2, n))
    df["protocol_type"] = rng.choice(["tcp", "udp", "icmp"], n)
    df["service"] = rng.choice(["http", "smtp", "private", "ecr_i"], n)
    df["flag"] = rng.choice(["SF", "S0", "REJ"], n)
    df["label"] = np.where(attack, rng.choice(["neptune.", "smurf."], n), "normal.")
    return df[COLUMNS]
