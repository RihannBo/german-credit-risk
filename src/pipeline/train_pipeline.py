"""
Train credit-risk models from the cleaned German Credit dataset.

Run from repository root::

    PYTHONPATH=src python -m pipeline.train_pipeline

Or from ``src``::

    python -m pipeline.train_pipeline
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import pandas as pd

from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer
from logger import logger


def main(data_path: Path | None = None) -> pd.DataFrame:
    raw = DataIngestion(data_path).read_data()
    cleaned = DataTransformation.clean(raw)
    X, y = DataTransformation.split_features_target(cleaned)
    trainer = ModelTrainer()
    metrics = trainer.train_all(X, y, save_primary=True)
    return pd.DataFrame(metrics).T


if __name__ == "__main__":
    logger.info("Starting training pipeline")
    comparison = main()
    print(comparison.to_string())
    logger.info("Training pipeline finished")
