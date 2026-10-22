"""Project-wide constants: paths, feature schema and the business cost model."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
PRODUCTION_DIR = MODELS_DIR / "production"
REPORTS_DIR = ROOT / "reports"
IMAGES_DIR = ROOT / "docs" / "images"

CATEGORIES = [
    "grocery",
    "fuel",
    "restaurants",
    "online_retail",
    "electronics",
    "travel",
    "gift_cards",
    "digital_goods",
]

NUMERIC_FEATURES = [
    "amount",
    "amount_to_avg_ratio",
    "hour",
    "card_present",
    "distance_from_home_km",
    "new_device",
    "txn_count_24h",
    "account_age_days",
]
CATEGORICAL_FEATURES = ["merchant_category"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "is_fraud"

# --- Business cost model -------------------------------------------------------
# Every flagged transaction goes to manual review / step-up auth: analyst time
# plus friction (abandoned purchases, support contacts) for legitimate customers.
REVIEW_COST = 12.0
# A missed fraud costs the transaction amount plus the chargeback fee.
CHARGEBACK_FEE = 25.0

# Chargebacks (our labels) arrive roughly two weeks after the transaction.
LABEL_DELAY_WEEKS = 2

SEED = 7
