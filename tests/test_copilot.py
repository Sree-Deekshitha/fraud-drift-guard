"""Copilot logic and /v1/explain, tested with a fake LLM client (no API key, no network)."""

import json
import shutil
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from fraudguard.api import create_app
from fraudguard.copilot import SCHEMA, Copilot, case_data, grounding_check

from .test_api import TXN

NOTE = {
    "suspected_pattern": "classic_cnp", "confidence": "high",
    "risk_factors": [{"field": "distance_from_home_km", "observation": "Purchase 640.0 km from home."},
                     {"field": "hour", "observation": "Made at hour 2, overnight."}],
    "mitigating_factors": [],
    "recommended_action": "block_and_contact_customer",
    "summary": "Card-not-present electronics purchase of $189.99 at 2am, 640.0 km away.",
}


class FakeClient:
    def __init__(self, note=NOTE):
        self.note, self.calls = note, []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(self.note))],
            usage=SimpleNamespace(input_tokens=1600, output_tokens=400), stop_reason="end_turn")


@pytest.fixture
def model_dir(trained, tmp_path):
    _, path = trained
    d = tmp_path / "m"
    shutil.copytree(path, d)
    (d / "typical.json").write_text(json.dumps([45, 1.2, 14, 1, 9, 0, 2, 520, 0]))
    return d


def test_copilot_sends_schema_and_grounds_note(trained, model_dir):
    model, _ = trained
    client = FakeClient()
    note = Copilot(model, json.loads((model_dir / "typical.json").read_text()), client=client).explain(
        {**TXN, "card_present": 0, "new_device": 1})
    sent = client.calls[0]
    assert sent["output_config"]["format"]["schema"] == SCHEMA
    assert "CASE DATA" in sent["messages"][0]["content"]
    assert note.ungrounded == []
    assert note.cost_usd == pytest.approx((1600 * 2 + 400 * 10) / 1e6)


def test_grounding_flags_invented_numbers(trained, model_dir):
    model, _ = trained
    case = case_data(model, {**TXN, "card_present": 0, "new_device": 1},
                     json.loads((model_dir / "typical.json").read_text()))
    bad = {**NOTE, "summary": "Customer had 7 chargebacks and spent $5,000 last month."}
    assert set(grounding_check(bad, case)) == {"7", "$5,000"}
    assert grounding_check(NOTE, case) == []


def test_explain_endpoint(model_dir):
    with TestClient(create_app(model_dir=model_dir, llm_client=FakeClient())) as c:
        assert c.get("/health").json()["copilot"] is True
        r = c.post("/v1/explain", json=TXN)
    assert r.status_code == 200
    body = r.json()
    assert body["case_note"]["suspected_pattern"] == "classic_cnp"
    assert body["ungrounded_numbers"] == [] and body["llm_cost_usd"] > 0


def test_explain_disabled_without_key(model_dir, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with TestClient(create_app(model_dir=model_dir)) as c:
        assert c.post("/v1/explain", json=TXN).status_code == 503
