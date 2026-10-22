"""FastAPI scoring service.

Run:  uvicorn fraudguard.api:app --port 8000
Env:  FRAUDGUARD_MODEL_DIR  (default: models/production)
      ANTHROPIC_API_KEY     enables POST /v1/explain (LLM case notes); never committed
      FRAUDGUARD_SCORER     "fast" (default) or "pandas" - kept to reproduce the
                            latency benchmark in reports/load_test.json
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .config import PRODUCTION_DIR
from .model import FraudModel, to_array

Category = Literal[
    "grocery", "fuel", "restaurants", "online_retail",
    "electronics", "travel", "gift_cards", "digital_goods",
]


class Transaction(BaseModel):
    amount: float = Field(gt=0, le=100_000, examples=[189.99])
    amount_to_avg_ratio: float = Field(gt=0, examples=[4.2])
    hour: int = Field(ge=0, le=23, examples=[2])
    card_present: bool = Field(examples=[False])
    distance_from_home_km: float = Field(ge=0, examples=[640.0])
    new_device: bool = Field(examples=[True])
    txn_count_24h: int = Field(ge=1, examples=[3])
    account_age_days: float = Field(ge=0, examples=[412])
    merchant_category: Category = Field(examples=["electronics"])


class Score(BaseModel):
    fraud_probability: float
    decision: Literal["approve", "review"]


class ScoreResponse(Score):
    threshold: float
    model_version: str


class FactorOut(BaseModel):
    field: str
    observation: str


class CaseNoteOut(BaseModel):
    suspected_pattern: str
    confidence: str
    risk_factors: list[FactorOut]
    mitigating_factors: list[FactorOut]
    recommended_action: str
    summary: str


class ExplainResponse(ScoreResponse):
    case_note: CaseNoteOut
    ungrounded_numbers: list[str]
    llm_model: str
    llm_latency_s: float
    llm_cost_usd: float


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    results: list[Score]
    threshold: float
    model_version: str


def create_app(model_dir: Path | None = None, scorer: str | None = None, llm_client=None) -> FastAPI:
    model_dir = Path(model_dir or os.getenv("FRAUDGUARD_MODEL_DIR", PRODUCTION_DIR))
    scorer = scorer or os.getenv("FRAUDGUARD_SCORER", "fast")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = FraudModel.load(model_dir)
        app.state.copilot = None
        if llm_client is not None or os.getenv("ANTHROPIC_API_KEY"):
            from .copilot import Copilot, load_typical

            app.state.copilot = Copilot(app.state.model, load_typical(model_dir), client=llm_client)
        yield

    app = FastAPI(
        title="fraud-drift-guard",
        version="1.0.0",
        description="Real-time card-fraud scoring with a cost-optimised decision threshold.",
        lifespan=lifespan,
    )

    def _score(model: FraudModel, txns: list[Transaction]) -> np.ndarray:
        records = [t.model_dump() for t in txns]
        if scorer == "pandas":
            import pandas as pd

            return model.predict_proba(pd.DataFrame(records))
        return model.predict_proba_array(to_array(records))

    @app.middleware("http")
    async def timing(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - start) * 1000:.3f}"
        return response

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": app.state.model.version, "scorer": scorer,
                "copilot": app.state.copilot is not None}

    @app.get("/v1/model")
    def model_info():
        return app.state.model.meta()

    @app.post("/v1/score", response_model=ScoreResponse)
    def score(txn: Transaction):
        m = app.state.model
        p = float(_score(m, [txn])[0])
        return ScoreResponse(
            fraud_probability=round(p, 6),
            decision="review" if p >= m.threshold else "approve",
            threshold=m.threshold, model_version=m.version,
        )

    @app.post("/v1/explain", response_model=ExplainResponse)
    def explain(txn: Transaction):
        """Score the transaction and ask the LLM copilot for an analyst case note."""
        if app.state.copilot is None:
            raise HTTPException(503, "LLM copilot disabled: set ANTHROPIC_API_KEY on the server")
        m = app.state.model
        rec = txn.model_dump()
        rec["card_present"], rec["new_device"] = int(rec["card_present"]), int(rec["new_device"])
        p = float(_score(m, [txn])[0])
        cn = app.state.copilot.explain(rec)
        return ExplainResponse(
            fraud_probability=round(p, 6), decision="review" if p >= m.threshold else "approve",
            threshold=m.threshold, model_version=m.version, case_note=cn.note,
            ungrounded_numbers=cn.ungrounded, llm_model=cn.model,
            llm_latency_s=round(cn.latency_s, 3), llm_cost_usd=round(cn.cost_usd, 6),
        )

    @app.post("/v1/score/batch", response_model=BatchResponse)
    def score_batch(req: BatchRequest):
        m = app.state.model
        probs = _score(m, req.transactions)
        return BatchResponse(
            results=[
                Score(fraud_probability=round(float(p), 6),
                      decision="review" if p >= m.threshold else "approve")
                for p in probs
            ],
            threshold=m.threshold, model_version=m.version,
        )

    return app


app = create_app()
