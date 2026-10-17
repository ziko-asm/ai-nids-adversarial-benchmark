"""
REST API that runs the trained detector on raw KDD Cup 99 connection records.

    python -m src.train --model mlp          # creates checkpoints/mlp.pt
    uvicorn src.api:app --reload             # docs at http://localhost:8000/docs

Set NIDS_CHECKPOINT to serve a different checkpoint (e.g. checkpoints/mlp_adv.pt).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.dataset import FEATURES, Preprocessor
from src.models import build_model

app = FastAPI(
    title="NIDS Adversarial Robustness API",
    description="Serves a trained intrusion detector and the benchmark results.",
    version="2.0.0",
)


def checkpoint_path() -> Path:
    return Path(os.environ.get("NIDS_CHECKPOINT", "checkpoints/mlp.pt"))


@lru_cache(maxsize=4)
def _load(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    prep = Preprocessor.from_dict(ck["preprocessor"])
    model = build_model(ck["model"], len(prep.feature_names))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, prep, ck


def get_model():
    path = checkpoint_path()
    if not path.exists():
        raise HTTPException(503, f"No checkpoint at {path}. Run: python -m src.train --model mlp")
    return _load(str(path))


class Record(BaseModel):
    features: list | dict = Field(
        ..., description="41 raw KDD Cup 99 fields, either as a list in the standard order "
                         "or as a {name: value} object")


@app.get("/")
def root():
    return {"service": app.title, "version": app.version, "docs": "/docs"}


@app.get("/health")
def health():
    path = checkpoint_path()
    return {"status": "healthy", "checkpoint": str(path), "model_loaded": path.exists()}


@app.post("/predict")
def predict(rec: Record):
    model, prep, ck = get_model()
    if isinstance(rec.features, list):
        if len(rec.features) != len(FEATURES):
            raise HTTPException(422, f"Expected {len(FEATURES)} values in KDD order, got {len(rec.features)}")
        row = dict(zip(FEATURES, rec.features))
    else:
        missing = [f for f in FEATURES if f not in rec.features]
        if missing:
            raise HTTPException(422, f"Missing fields: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        row = rec.features
    try:
        x = torch.as_tensor(prep.transform(pd.DataFrame([row], columns=FEATURES)))
    except (ValueError, TypeError) as e:
        raise HTTPException(422, f"Could not parse record: {e}") from e
    with torch.no_grad():
        p = float(torch.softmax(model(x), -1)[0, 1])
    return {"prediction": "attack" if p >= 0.5 else "normal", "p_attack": round(p, 4),
            "model": ck["model"], "adversarially_trained": ck["adversarial"]}


@app.get("/results")
def results():
    path = Path(os.environ.get("NIDS_RESULTS", "results/results.json"))
    if not path.exists():
        raise HTTPException(404, "No benchmark yet. Run: python -m src.evaluate")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {"meta": data["meta"], "summary": data["summary"], "xai": data.get("xai", {})}
