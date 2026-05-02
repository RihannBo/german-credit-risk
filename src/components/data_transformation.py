from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from config import (
    CHECKING_CATEGORIES,
    DROP_FOR_X,
    JOB_CATEGORIES,
    NOMINAL_FEATURES,
    NUM_FEATURES,
    ORDINAL_FEATURES,
    RISK_MAP,
    SAVING_CATEGORIES,
    TARGET_COL,
)
from exception import CustomException
from logger import logger


class DataTransformation:
    """Cleaning + feature engineering from the EDA / modeling notebook."""

    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = (
            out.columns.str.lower().str.strip().str.replace(" ", "_", regex=False)
        )
        out["saving_accounts"] = out["saving_accounts"].fillna("unknown")
        out["checking_account"] = out["checking_account"].fillna("unknown")

        cat_cols = out.select_dtypes(include=["object"]).columns.tolist()
        if "job" not in cat_cols:
            cat_cols.append("job")
        for col in cat_cols:
            out[col] = out[col].astype("category")
        return out

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add ``credit_amount_log`` and encode ``risk`` as 0/1 for training."""
        out = df.copy()
        if TARGET_COL not in out.columns:
            raise ValueError(f"Missing target column '{TARGET_COL}' for training.")
        out[TARGET_COL] = out[TARGET_COL].map(RISK_MAP)
        if out[TARGET_COL].isna().any():
            raise ValueError("Unknown labels in risk column; expected 'good'/'bad'.")
        out["credit_amount_log"] = np.log1p(out["credit_amount"].astype(float))
        return out

    @staticmethod
    def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Return X, y matching notebook (drops raw credit_amount and target)."""
        engineered = DataTransformation.engineer_features(df)
        y = engineered[TARGET_COL]
        X = engineered.drop(columns=list(DROP_FOR_X), errors="ignore")
        missing = set(NUM_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f"Missing expected columns: {sorted(missing)}")
        return X, y

    @staticmethod
    def prepare_inference_features(df: pd.DataFrame) -> pd.DataFrame:
        """Build feature matrix for scoring (no ``risk`` required)."""
        out = DataTransformation.clean(df)
        if TARGET_COL in out.columns:
            out = out.drop(columns=[TARGET_COL])
        out["credit_amount_log"] = np.log1p(out["credit_amount"].astype(float))
        out = out.drop(columns=["credit_amount"], errors="ignore")
        missing = set(NUM_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES) - set(out.columns)
        if missing:
            raise ValueError(f"Missing expected columns: {sorted(missing)}")
        return out[NUM_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES]

    @staticmethod
    def build_preprocessor() -> ColumnTransformer:
        """Same ``ColumnTransformer`` as the notebook."""
        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUM_FEATURES),
                (
                    "nom",
                    OneHotEncoder(handle_unknown="ignore"),
                    NOMINAL_FEATURES,
                ),
                (
                    "ord",
                    OrdinalEncoder(
                        categories=[JOB_CATEGORIES, SAVING_CATEGORIES, CHECKING_CATEGORIES]
                    ),
                    ORDINAL_FEATURES,
                ),
            ]
        )


def run_cleaning_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    try:
        return DataTransformation.clean(raw_df)
    except Exception as e:
        logger.error("Cleaning failed: %s", e)
        raise CustomException(e) from e
