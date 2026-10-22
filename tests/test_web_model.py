"""The browser demo must give the same scores as the Python model."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "web_parity.json"

JS = """
const fs = require('fs');
const { score } = require(process.argv[1]);
const model = JSON.parse(fs.readFileSync(process.argv[2]));
const fx = JSON.parse(fs.readFileSync(process.argv[3]));
console.log(JSON.stringify(fx.rows.map(r => score(model, r).probability)));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_browser_scorer_matches_python():
    out = subprocess.run(
        ["node", "-e", JS, str(ROOT / "docs" / "scorer.js"), str(ROOT / "docs" / "model.json"),
         str(FIXTURES)],
        capture_output=True, text=True, check=True,
    )
    got = np.array(json.loads(out.stdout))
    expected = np.array(json.loads(FIXTURES.read_text())["expected"])
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-9)
