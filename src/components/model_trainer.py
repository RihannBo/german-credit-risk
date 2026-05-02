from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from config import (
    ARTIFACTS_DIR,
    DEFAULT_THRESHOLD,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    XGB_PARAMS,
)
from components.data_transformation import DataTransformation
from exception import CustomException
from logger import logger
from utils import save_object


def evaluate_model(
    y_true, y_pred, y_proba
) -> tuple[dict[str, float], Any, str]:
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_proba)),
    }
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, zero_division=0)
    return metrics, cm, cr


@dataclass
class ModelTrainer:
    """Train baseline models (LR, RF, XGB) and persist primary XGBoost pipeline."""

    def build_pipelines(self, y_train: pd.Series) -> dict[str, Pipeline]:
        # Each estimator needs its own preprocessor instance (independent fit state).
        pre_lr = DataTransformation.build_preprocessor()
        pre_rf = DataTransformation.build_preprocessor()
        pre_xgb = DataTransformation.build_preprocessor()

        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        scale_pos_weight = (neg / pos) if pos else 1.0

        xgb_params = {**XGB_PARAMS, "scale_pos_weight": scale_pos_weight}

        return {
            "logistic_regression": Pipeline(
                [
                    ("preprocessor", pre_lr),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=1000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "random_forest": Pipeline(
                [
                    ("preprocessor", pre_rf),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            min_samples_split=5,
                            min_samples_leaf=2,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "xgboost": Pipeline(
                [
                    ("preprocessor", pre_xgb),
                    ("model", XGBClassifier(**xgb_params)),
                ]
            ),
        }

    def train_all(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        save_primary: bool = True,
    ) -> dict[str, dict[str, float]]:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y,
            )
            pipelines = self.build_pipelines(y_train)
            metrics_out: dict[str, dict[str, float]] = {}

            for name, pipe in pipelines.items():
                logger.info("Fitting %s", name)
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                y_proba = pipe.predict_proba(X_test)[:, 1]
                metrics, _, _ = evaluate_model(y_test, y_pred, y_proba)
                metrics_out[name] = metrics
                logger.info("%s metrics: %s", name, metrics)

            primary = pipelines["xgboost"]
            artifact = {
                "pipeline": primary,
                "default_threshold": DEFAULT_THRESHOLD,
                "feature_columns": list(X.columns),
            }
            if save_primary:
                ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
                save_object(artifact, MODEL_PATH)
                logger.info("Saved primary artifact to %s", MODEL_PATH)

            return metrics_out
        except Exception as e:
            logger.error("Training failed: %s", e)
            raise CustomException(e) from e


def apply_threshold(proba: pd.Series | Any, threshold: float) -> Any:
    import numpy as np

    return (np.asarray(proba) >= threshold).astype(int)
