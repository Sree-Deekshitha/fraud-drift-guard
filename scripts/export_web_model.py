"""Export the production LightGBM model to compact JSON for the in-browser demo
(docs/index.html), plus parity fixtures that tests/test_web_model.py checks.

Node format:  leaf  -> number (raw score contribution)
              split -> [feature_idx, is_categorical, threshold | [cats], default_left,
                        missing_type, left, right]
"""

from __future__ import annotations

import json

from fraudguard.config import CATEGORIES, FEATURES, PRODUCTION_DIR, ROOT
from fraudguard.model import FraudModel
from fraudguard.simulate import generate_week, generate_weeks

OUT = ROOT / "docs" / "model.json"
FIXTURES = ROOT / "tests" / "fixtures" / "web_parity.json"


def convert(node: dict):
    if "leaf_value" in node:
        return round(node["leaf_value"], 10)
    cat = node["decision_type"] == "=="
    thr = [int(c) for c in str(node["threshold"]).split("||")] if cat else node["threshold"]
    return [node["split_feature"], int(cat), thr, int(node["default_left"]),
            node["missing_type"], convert(node["left_child"]), convert(node["right_child"])]


def main() -> None:
    model = FraudModel.load(PRODUCTION_DIR)
    dump = model.booster.dump_model(num_iteration=model.booster.best_iteration or None)
    assert dump["feature_names"] == FEATURES
    train = generate_weeks(model.trained_weeks)
    legit = train[train["is_fraud"] == 0]
    # "Typical" legit value per feature, used for the demo's what-if attributions.
    typical = [float(legit[f].median()) for f in FEATURES[:-1]]
    typical.append(CATEGORIES.index(legit["merchant_category"].mode()[0]))
    payload = dict(
        version=model.version, threshold=model.threshold, features=FEATURES, typical=typical,
        categories=CATEGORIES, trees=[convert(t["tree_structure"]) for t in dump["tree_info"]],
    )
    OUT.write_text(json.dumps(payload, separators=(",", ":")))

    sample = generate_week(31).sample(300, random_state=1)
    FIXTURES.parent.mkdir(parents=True, exist_ok=True)
    FIXTURES.write_text(json.dumps(dict(
        rows=sample[FEATURES].to_dict("records"),
        expected=model.predict_proba(sample).tolist(),
    )))
    print(f"{len(payload['trees'])} trees -> {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
