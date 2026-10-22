import pytest
from fastapi.testclient import TestClient

from fraudguard.api import create_app

TXN = dict(
    amount=189.99, amount_to_avg_ratio=4.2, hour=2, card_present=False,
    distance_from_home_km=640.0, new_device=True, txn_count_24h=3,
    account_age_days=412, merchant_category="electronics",
)


@pytest.fixture(params=["fast", "pandas"])
def client(trained, request):
    _, path = trained
    with TestClient(create_app(model_dir=path, scorer=request.param)) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_version"] == "test"


def test_score_single(client):
    r = client.post("/v1/score", json=TXN)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["fraud_probability"] <= 1
    assert body["decision"] in {"approve", "review"}
    assert "X-Process-Time-ms" in r.headers


def test_score_batch_matches_single(client):
    single = client.post("/v1/score", json=TXN).json()["fraud_probability"]
    batch = client.post("/v1/score/batch", json={"transactions": [TXN, TXN]}).json()
    assert [x["fraud_probability"] for x in batch["results"]] == [single, single]


@pytest.mark.parametrize("bad", [{"hour": 24}, {"amount": -5}, {"merchant_category": "casino"}])
def test_validation_rejects_bad_input(client, bad):
    assert client.post("/v1/score", json={**TXN, **bad}).status_code == 422
