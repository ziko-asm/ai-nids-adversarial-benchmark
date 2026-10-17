import torch
from fastapi.testclient import TestClient

from src import api
from src.dataset import FEATURES, prepare, synthetic_kdd
from src.train import save_checkpoint
from src.utils import evaluate, train_model

client = TestClient(api.app)


def test_root_and_health(monkeypatch, tmp_path):
    monkeypatch.setenv("NIDS_CHECKPOINT", str(tmp_path / "missing.pt"))
    assert client.get("/").status_code == 200
    r = client.get("/health").json()
    assert r["status"] == "healthy" and r["model_loaded"] is False


def test_predict_without_checkpoint_is_503(monkeypatch, tmp_path):
    monkeypatch.setenv("NIDS_CHECKPOINT", str(tmp_path / "missing.pt"))
    r = client.post("/predict", json={"features": [0] * 41})
    assert r.status_code == 503


def test_predict_with_real_checkpoint(monkeypatch, tmp_path):
    s = prepare(synthetic_kdd(400, seed=2), seed=0)
    model = train_model("mlp", s.X_train, s.y_train, s.X_val, s.y_val, verbose=False, epochs=5, batch_size=64)
    path = tmp_path / "mlp.pt"
    save_checkpoint(model, "mlp", s.prep, path, False, evaluate(model, torch.as_tensor(s.X_test), s.y_test))
    monkeypatch.setenv("NIDS_CHECKPOINT", str(path))

    row = synthetic_kdd(1, seed=9).iloc[0]
    as_list = [row[f].item() if hasattr(row[f], "item") else row[f] for f in FEATURES]
    r = client.post("/predict", json={"features": as_list})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in {"attack", "normal"} and 0 <= body["p_attack"] <= 1

    r2 = client.post("/predict", json={"features": dict(zip(FEATURES, as_list))})
    assert r2.json()["p_attack"] == body["p_attack"]

    assert client.post("/predict", json={"features": [0] * 10}).status_code == 422
