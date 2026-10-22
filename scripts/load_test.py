"""Latency benchmark of the scoring API: pandas scorer vs. numpy fast path.

Starts uvicorn (1 worker) per scorer, warms up, then fires N single-transaction
requests with fixed concurrency and records client-side latency.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import subprocess
import sys
import time

import httpx
import numpy as np

from fraudguard.config import FEATURES, IMAGES_DIR, REPORTS_DIR, ROOT
from fraudguard.plotting import INK_2, SERIES, apply_style, plt
from fraudguard.simulate import generate_week

N_REQUESTS = 3000
CONCURRENCY = 8
PORT = 8765


async def _fire(payloads: list[dict]) -> np.ndarray:
    lat = np.empty(len(payloads))
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT}", timeout=10) as client:
        async def one(i, body):
            async with sem:
                t = time.perf_counter()
                r = await client.post("/v1/score", json=body)
                lat[i] = (time.perf_counter() - t) * 1000
                r.raise_for_status()
        await asyncio.gather(*(one(i, p) for i, p in enumerate(payloads)))
    return lat


def bench(scorer: str, payloads: list[dict]) -> dict:
    env = {**os.environ, "FRAUDGUARD_SCORER": scorer}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "fraudguard.api:app", "--port", str(PORT),
         "--log-level", "warning", "--workers", "1"], cwd=ROOT, env=env,
    )
    try:
        for _ in range(100):
            try:
                if httpx.get(f"http://127.0.0.1:{PORT}/health").status_code == 200:
                    break
            except httpx.TransportError:
                time.sleep(0.1)
        asyncio.run(_fire(payloads[:200]))  # warm-up
        t = time.perf_counter()
        lat = asyncio.run(_fire(payloads))
        wall = time.perf_counter() - t
    finally:
        proc.terminate()
        proc.wait()
    return dict(scorer=scorer, p50=float(np.percentile(lat, 50)), p95=float(np.percentile(lat, 95)),
                p99=float(np.percentile(lat, 99)), rps=len(payloads) / wall, latencies=lat)


def main() -> None:
    apply_style()
    df = generate_week(30).head(N_REQUESTS)
    payloads = df[FEATURES].to_dict("records")
    for p in payloads:
        p["card_present"], p["new_device"] = bool(p["card_present"]), bool(p["new_device"])
        p["hour"], p["txn_count_24h"] = int(p["hour"]), int(p["txn_count_24h"])

    print(f"Load test: {N_REQUESTS} requests, concurrency {CONCURRENCY}, 1 uvicorn worker, "
          f"{os.cpu_count()} vCPU ({platform.machine()}, Python {platform.python_version()})\n")
    results = [bench(s, payloads) for s in ("pandas", "fast")]
    print(f"{'Scorer':<10}{'p50 ms':>9}{'p95 ms':>9}{'p99 ms':>9}{'req/s':>9}")
    for r in results:
        print(f"{r['scorer']:<10}{r['p50']:>9.2f}{r['p95']:>9.2f}{r['p99']:>9.2f}{r['rps']:>9.0f}")
    a, b = results
    print(f"\nFast path: p50 {1 - b['p50'] / a['p50']:.0%} lower, p99 {1 - b['p99'] / a['p99']:.0%} lower, "
          f"throughput x{b['rps'] / a['rps']:.1f}")

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    hi = max(np.percentile(r["latencies"], 99.5) for r in results)
    bins = np.linspace(0, hi, 70)
    for i, r in enumerate(results):
        label = "pandas DataFrame scorer" if r["scorer"] == "pandas" else "numpy fast path"
        ax.hist(r["latencies"], bins=bins, color=SERIES[i], alpha=0.75,
                label=f"{label}  (p50 {r['p50']:.1f} / p99 {r['p99']:.1f} ms)")
        ax.axvline(r["p99"], color=SERIES[i], lw=1, ls="--")
    ax.set(xlabel="Client-side latency per request (ms)", ylabel="Requests",
           title="Scoring latency, single transaction")
    ax.text(0.62, 0.55, "dashed lines = p99", transform=ax.transAxes, ha="left", color=INK_2, fontsize=8)
    ax.legend(loc="upper right")
    fig.savefig(IMAGES_DIR / "latency.png")
    plt.close(fig)

    out = [{k: v for k, v in r.items() if k != "latencies"} for r in results]
    (REPORTS_DIR / "load_test.json").write_text(json.dumps(
        {"requests": N_REQUESTS, "concurrency": CONCURRENCY, "cpus": os.cpu_count(),
         "results": out}, indent=2))


if __name__ == "__main__":
    main()
