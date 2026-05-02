"""
Score new applicants using the saved training artifact.

Run from repository root::

    PYTHONPATH=src python -m pipeline.predict_pipeline --input data/new_applicants.csv

``--input`` CSV must contain the same raw feature columns as training (``Risk`` optional).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import numpy as np
import pandas as pd

from components.data_transformation import DataTransformation
from config import DEFAULT_THRESHOLD, MODEL_PATH
from logger import logger
from utils import load_object


def predict_proba(
    df: pd.DataFrame,
    artifact_path: Path | None = None,
) -> np.ndarray:
    path = artifact_path or MODEL_PATH
    bundle = load_object(path)
    pipeline = bundle["pipeline"]
    X = DataTransformation.prepare_inference_features(df)
    return pipeline.predict_proba(X)[:, 1]


def predict_labels(
    df: pd.DataFrame,
    threshold: float | None = None,
    artifact_path: Path | None = None,
) -> np.ndarray:
    bundle = load_object(artifact_path or MODEL_PATH)
    thr = threshold if threshold is not None else float(
        bundle.get("default_threshold", DEFAULT_THRESHOLD)
    )
    proba = predict_proba(df, artifact_path=artifact_path)
    return (proba >= thr).astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description="Credit risk scoring")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV with same schema as training raw data",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help=f"Joblib bundle path (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"Bad-risk cutoff on predicted probability (default: from artifact or {DEFAULT_THRESHOLD})",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, index_col=0)
    proba = predict_proba(df, artifact_path=args.artifact)
    labels = predict_labels(df, threshold=args.threshold, artifact_path=args.artifact)
    out = df.copy()
    out["bad_risk_proba"] = proba
    out["bad_risk_pred"] = labels
    print(out[["bad_risk_proba", "bad_risk_pred"]].to_string())
    logger.info("Scored %s rows", len(out))


if __name__ == "__main__":
    main()
