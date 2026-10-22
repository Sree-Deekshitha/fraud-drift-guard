// Evaluates the exported LightGBM model (docs/model.json) in the browser.
// Kept dependency-free so the same file is tested from Node (tests/test_web_model.py).

const ZERO_THRESHOLD = 1e-35;

function walk(node, x) {
  while (Array.isArray(node)) {
    const [f, isCat, thr, defaultLeft, missingType, left, right] = node;
    const v = x[f];
    let goLeft;
    if (isCat) {
      goLeft = Number.isInteger(v) && v >= 0 ? thr.includes(v) : false;
    } else if (Number.isNaN(v)) {
      goLeft = missingType === "NaN" ? Boolean(defaultLeft) : 0 <= thr;
    } else if (missingType === "Zero" && Math.abs(v) <= ZERO_THRESHOLD) {
      goLeft = Boolean(defaultLeft);
    } else {
      goLeft = v <= thr;
    }
    node = goLeft ? left : right;
  }
  return node;
}

function toVector(model, txn) {
  return model.features.map((name) =>
    name === "merchant_category" ? model.categories.indexOf(txn[name]) : Number(txn[name])
  );
}

function contributions(model, txn) {
  // Per-feature attribution by leave-one-out on the raw score: replace one feature with
  // its "typical" value and measure the change. Simple, honest and fast enough for 9 features.
  const x = toVector(model, txn);
  const base = rawScore(model, x);
  return model.features.map((name, i) => {
    const y = x.slice();
    y[i] = model.typical[i];
    return { feature: name, delta: base - rawScore(model, y) };
  });
}

function rawScore(model, x) {
  let s = 0;
  for (const t of model.trees) s += walk(t, x);
  return s;
}

function score(model, txn) {
  const p = 1 / (1 + Math.exp(-rawScore(model, toVector(model, txn))));
  return { probability: p, decision: p >= model.threshold ? "review" : "approve" };
}

if (typeof module !== "undefined") module.exports = { score, rawScore, toVector, contributions };
