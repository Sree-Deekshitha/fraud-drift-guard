"""Business cost of a decision policy, and threshold selection by cost."""

from __future__ import annotations

import numpy as np

from .config import CHARGEBACK_FEE, REVIEW_COST


def cost_breakdown(y, flagged, amount) -> dict:
    """Cost of flagging ``flagged`` transactions. Flagged fraud is assumed stopped."""
    y = np.asarray(y).astype(bool)
    flagged = np.asarray(flagged).astype(bool)
    amount = np.asarray(amount, dtype=float)
    missed = y & ~flagged
    fraud_loss = float((amount[missed] + CHARGEBACK_FEE).sum())
    review = float(flagged.sum() * REVIEW_COST)
    n = len(y)
    tp = int((y & flagged).sum())
    return dict(
        fraud_loss=fraud_loss,
        review_cost=review,
        total_cost=fraud_loss + review,
        cost_per_1k=1000 * (fraud_loss + review) / max(n, 1),
        recall=tp / max(int(y.sum()), 1),
        precision=tp / max(int(flagged.sum()), 1),
        flag_rate=float(flagged.mean()) if n else 0.0,
        n=n,
    )


def best_threshold(y, scores, amount, grid=None) -> tuple[float, np.ndarray, np.ndarray]:
    """Return the cost-minimising threshold plus the (grid, cost) curve."""
    if grid is None:
        grid = np.unique(np.round(np.concatenate([np.linspace(0.002, 0.05, 25), np.linspace(0.05, 0.95, 181)]), 4))
    scores = np.asarray(scores)
    costs = np.array([cost_breakdown(y, scores >= t, amount)["total_cost"] for t in grid])
    return float(grid[int(np.argmin(costs))]), grid, costs
