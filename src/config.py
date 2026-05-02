"""Central configuration aligned with ``notebooks/german_credit_risk.ipynb``."""

from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


DATA_DIR = get_project_root() / "data"
RAW_DATA_FILE = DATA_DIR / "german_credit_data.csv"
ARTIFACTS_DIR = get_project_root() / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "credit_risk_model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_COL = "risk"
DROP_FOR_X = ("risk", "credit_amount")

NUM_FEATURES = ["age", "credit_amount_log", "duration"]
ORDINAL_FEATURES = ["job", "saving_accounts", "checking_account"]
NOMINAL_FEATURES = ["sex", "housing", "purpose"]

# Ordinal category orders (must match notebook / business encoding)
JOB_CATEGORIES = [0, 1, 2, 3]
SAVING_CATEGORIES = ["unknown", "little", "moderate", "quite rich", "rich"]
CHECKING_CATEGORIES = ["unknown", "little", "moderate", "rich"]

RISK_MAP = {"bad": 1, "good": 0}

# Primary model (matches notebook baseline XGBoost)
XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
}

DEFAULT_THRESHOLD = 0.45
