"""Tests de la API con TestClient (M5, §6)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.app.main import app

client = TestClient(app)


def _score_body(events, hour=21, day=4):
    return {"events": events, "hour_of_day": hour, "day_of_week": day}


def test_health() -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True
    assert body["version"]


def test_score_valid() -> None:
    r = client.post("/api/score", json=_score_body([
        {"type": "view", "item_category": "electronics", "seconds_since_prev": 0},
        {"type": "view", "item_category": "electronics", "seconds_since_prev": 45},
        {"type": "addtocart", "item_category": "electronics", "seconds_since_prev": 30},
    ]))
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["conversion_probability"] <= 1.0
    assert body["label"] in {"caliente", "tibio", "frío"}
    assert set(body["model_breakdown"]) == {"baseline", "gru", "transformer", "ensemble"}
    assert "threshold" in body


def test_score_rejects_transaction_type() -> None:
    # 'transaction' no es un tipo de evento válido de entrada (anti-leakage).
    r = client.post("/api/score", json=_score_body([
        {"type": "transaction", "item_category": "x", "seconds_since_prev": 0},
    ]))
    assert r.status_code == 422


def test_score_rejects_invalid_hour() -> None:
    r = client.post("/api/score", json=_score_body(
        [{"type": "view", "item_category": "x", "seconds_since_prev": 0}], hour=99
    ))
    assert r.status_code == 422


def test_unknown_category_maps_to_oov() -> None:
    # Una categoría desconocida no debe romper la inferencia (→ OOV).
    r = client.post("/api/score", json=_score_body([
        {"type": "view", "item_category": "categoria-inexistente", "seconds_since_prev": 0},
        {"type": "view", "item_category": "otra-rara", "seconds_since_prev": 5},
    ]))
    assert r.status_code == 200


def test_score_batch_csv() -> None:
    csv = (
        "lead_id,n_views,n_addtocart,n_unique_items,duration_sec,hour_of_day,day_of_week\n"
        "A,10,3,5,300,21,4\n"
        "B,2,0,2,20,3,1\n"
    )
    r = client.post("/api/score/batch", files={"file": ("leads.csv", csv, "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    assert body[0]["probability"] >= body[1]["probability"]  # orden descendente
    assert set(body[0]) == {"lead_id", "probability", "label", "segment", "recommended_action"}


def test_score_batch_missing_columns() -> None:
    csv = "lead_id,n_views\nA,1\n"
    r = client.post("/api/score/batch", files={"file": ("x.csv", csv, "text/csv")})
    assert r.status_code == 422
