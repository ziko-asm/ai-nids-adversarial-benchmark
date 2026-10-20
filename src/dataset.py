"""
Dataset loading and preprocessing for KDD Cup 1999 (10% subset) and UNSW-NB15.

Choices that matter for honest numbers:
  * Exact duplicate rows are removed BEFORE splitting. The raw KDD 10% file has
    494,021 rows but only 145,586 unique ones; without de-duplication the same
    record lands in train and test and accuracy is inflated.
  * UNSW-NB15 also contains rows whose features are identical but whose labels
    differ. They cannot be learned and would sit in train and test with opposite
    labels, so every row of such a feature vector is dropped (counted in stats).
  * Split is stratified train / val / test (60 / 20 / 20) and seeded. For
    UNSW-NB15 the official training and testing files are pooled first, so both
    datasets go through the same protocol.
  * Heavy-tailed count and byte features go through log1p; everything is then
    standardised with statistics from the training split only.
  * Categorical columns are one-hot encoded with categories learned on the
    training split (unseen values map to all-zeros).

Each dataset is described by a `DatasetSpec`, including which features the
attacker cannot change in the constrained threat model. The preprocessing state
(`Preprocessor`) records its dataset and is saved with each checkpoint so the
API can turn a raw record into the exact model input.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- KDD Cup 1999
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

# ---------------------------------------------------------------- UNSW-NB15
UNSW_FILES = ("data/UNSW_NB15_training-set.csv", "data/UNSW_NB15_testing-set.csv")

UNSW_FEATURES = [
    "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit",
    "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
    "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
    "is_sm_ips_ports",
]
UNSW_CATEGORICAL = ["proto", "service", "state"]
UNSW_BINARY = ["is_ftp_login", "is_sm_ips_ports"]
# What the victim sends back, or what the monitor derives from it. An attacker
# shapes its own packets (sizes, timing, TTL, how many connections it opens),
# not the destination's replies.
UNSW_VICTIM_SIDE = [
    "dpkts", "dbytes", "dttl", "dload", "dloss", "dinpkt", "djit", "dwin", "dtcpb",
    "dmean", "synack", "tcprtt", "response_body_len", "ct_state_ttl",
]
UNSW_LOG_FEATURES = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sload", "dload", "sloss",
    "dloss", "sinpkt", "dinpkt", "sjit", "djit", "stcpb", "dtcpb", "tcprtt", "synack",
    "ackdat", "smean", "dmean", "trans_depth", "response_body_len", "ct_srv_src",
    "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    title: str
    features: list
    categorical: list
    immutable: list          # frozen in the constrained threat model (categoricals included)
    log_features: list
    labels: Callable[[pd.DataFrame], np.ndarray]
    attack_types: Callable[[pd.DataFrame], np.ndarray]
    read: Callable[..., pd.DataFrame]
    drop_label_conflicts: bool = False

    @property
    def numeric(self) -> list:
        return [f for f in self.features if f not in self.categorical]


def _read_kdd(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path or DEFAULT_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download kddcup.data_10_percent.gz from "
            "http://kdd.ics.uci.edu/databases/kddcup99/ and extract it to data/."
        )
    return pd.read_csv(path, header=None, names=COLUMNS)


def _read_unsw(path: str | Path | None = None) -> pd.DataFrame:
    folder = Path(path) if path else Path(UNSW_FILES[0]).parent
    files = [folder / Path(f).name for f in UNSW_FILES]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"{', '.join(missing)} not found. Download the 'Training and Testing Sets' CSVs from "
            "https://research.unsw.edu.au/projects/unsw-nb15-dataset into data/."
        )
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # `id` is a row number per file and would let a model memorise position
    return df[UNSW_FEATURES + ["attack_cat", "label"]]


def binary_labels(df: pd.DataFrame) -> np.ndarray:
    return (df["label"].astype(str).str.strip() != "normal.").astype(np.int64).to_numpy()


SPECS: dict[str, DatasetSpec] = {
    "kdd": DatasetSpec(
        name="kdd", title="KDD Cup 1999 (10%)", features=FEATURES, categorical=CATEGORICAL,
        immutable=IMMUTABLE, log_features=LOG_FEATURES, labels=binary_labels,
        attack_types=lambda df: df["label"].astype(str).str.strip().str.rstrip(".").to_numpy(),
        read=_read_kdd,
    ),
    "unsw": DatasetSpec(
        name="unsw", title="UNSW-NB15", features=UNSW_FEATURES, categorical=UNSW_CATEGORICAL,
        immutable=UNSW_CATEGORICAL + UNSW_BINARY + UNSW_VICTIM_SIDE, log_features=UNSW_LOG_FEATURES,
        labels=lambda df: df["label"].astype(np.int64).to_numpy(),
        attack_types=lambda df: df["attack_cat"].astype(str).str.strip().to_numpy(),
        read=_read_unsw, drop_label_conflicts=True,
    ),
}


def read_raw(path: str | Path | None = None, dataset: str = "kdd") -> pd.DataFrame:
    return SPECS[dataset].read(path)


@dataclass
class Preprocessor:
    categories: dict = field(default_factory=dict)   # column -> list of values
    mean: list = field(default_factory=list)
    std: list = field(default_factory=list)
    feature_names: list = field(default_factory=list)  # model input order
    lower: list = field(default_factory=list)          # per-feature min on train (model space)
    upper: list = field(default_factory=list)
    dataset: str = "kdd"                               # older checkpoints predate this field

    @property
    def spec(self) -> DatasetSpec:
        return SPECS[self.dataset]

    # ------------------------------------------------------------------
    def _raw_matrix(self, df: pd.DataFrame) -> np.ndarray:
        spec = self.spec
        num = df[spec.numeric].astype(np.float64).copy()
        for c in spec.log_features:
            num[c] = np.log1p(num[c].clip(lower=0))
        blocks = [num.to_numpy()]
        for c in spec.categorical:
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
        spec = self.spec
        self.categories = {c: sorted(df[c].astype(str).unique().tolist()) for c in spec.categorical}
        self.feature_names = spec.numeric + [f"{c}={v}" for c in spec.categorical for v in self.categories[c]]
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
        base = set(self.spec.immutable)
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


def prepare(df: pd.DataFrame, seed: int = 0, max_rows: int | None = None,
            dataset: str = "kdd") -> Splits:
    spec = SPECS[dataset]
    n_raw = len(df)
    # features + binary label; for KDD this is the whole row, for UNSW it ignores attack_cat
    df = df.drop_duplicates(subset=spec.features + ["label"]).reset_index(drop=True)
    n_unique = len(df)
    n_conflicting = 0
    if spec.drop_label_conflicts:
        y_all = spec.labels(df)
        conflicted = (pd.Series(y_all).groupby([df[c] for c in spec.features], dropna=False)
                      .transform("nunique") > 1).to_numpy()
        n_conflicting = int(conflicted.sum())
        df = df[~conflicted].reset_index(drop=True)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed).reset_index(drop=True)
    y = spec.labels(df)
    rng = np.random.default_rng(seed)
    tr, va, te = _stratified_indices(y, rng)
    prep = Preprocessor(dataset=dataset).fit(df.iloc[tr])
    X = prep.transform(df)
    stats = {"rows_raw": n_raw, "rows_unique": n_unique, "rows_label_conflict_dropped": n_conflicting,
             "rows_used": len(df), "attack_share": float(y.mean()), "n_features": X.shape[1],
             "n_mutable": int(prep.mutable_mask().sum())}
    return Splits(X[tr], y[tr], X[va], y[va], X[te], y[te], prep,
                  spec.attack_types(df)[te], stats)


def load_splits(path: str | Path | None = None, seed: int = 0,
                max_rows: int | None = None, dataset: str = "kdd") -> Splits:
    return prepare(read_raw(path, dataset), seed=seed, max_rows=max_rows, dataset=dataset)


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


def synthetic_unsw(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Tiny UNSW-NB15-shaped frame for tests and CI."""
    rng = np.random.default_rng(seed)
    attack = rng.random(n) < 0.45
    numeric = [f for f in UNSW_FEATURES if f not in UNSW_CATEGORICAL]
    df = pd.DataFrame({c: rng.random(n) for c in numeric})
    for c in UNSW_LOG_FEATURES:
        df[c] = rng.integers(0, 1000, n)
    for c in UNSW_BINARY:
        df[c] = rng.integers(0, 2, n)
    df["sbytes"] = np.where(attack, rng.integers(0, 80, n), rng.integers(200, 9000, n))
    df["sttl"] = np.where(attack, 254, rng.choice([31, 62], n))
    df["proto"] = rng.choice(["tcp", "udp", "unas"], n)
    df["service"] = rng.choice(["-", "http", "dns"], n)
    df["state"] = rng.choice(["FIN", "INT", "CON"], n)
    df["attack_cat"] = np.where(attack, rng.choice(["Exploits", "Fuzzers"], n), "Normal")
    df["label"] = attack.astype(int)
    return df[UNSW_FEATURES + ["attack_cat", "label"]]
