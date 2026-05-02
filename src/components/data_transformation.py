import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from exception import CustomException
from logger import logging
from utils import save_object


# ================= CONFIG =================
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


# ================= MAIN CLASS =================
class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    # ================= FEATURE ENGINEERING =================
    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies feature engineering steps
        """
        try:
            # Log transformation
            df["credit_amount_log"] = np.log1p(df["credit_amount"])

            # Convert job to string (for consistent categorical handling)
            df["job"] = df["job"].astype(str)

            return df

        except Exception as e:
            raise CustomException(e, sys)

    # ================= PREPROCESSOR =================
    def get_data_transformer_object(self):
        """
        Creates and returns preprocessing pipeline
        """
        try:
            # Feature groups
            num_features = ["age", "credit_amount_log", "duration"]

            ordinal_features = ["job", "saving_accounts", "checking_account"]

            nominal_features = ["sex", "housing", "purpose"]

            # Category ordering
            job_categories = ["unknown", "0", "1", "2", "3"]

            saving_categories = ["unknown", "little", "moderate", "rich", "quite rich"]

            checking_categories = ["unknown", "little", "moderate", "rich"]

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Nominal pipeline
            nominal_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            # Ordinal pipeline
            ordinal_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                    ("ordinal", OrdinalEncoder(categories=[
                        job_categories,
                        saving_categories,
                        checking_categories
                    ]))
                ]
            )

            logging.info(f"Numerical features: {num_features}")
            logging.info(f"Ordinal features: {ordinal_features}")
            logging.info(f"Nominal features: {nominal_features}")

            # Combine pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, num_features),
                    ("ord", ordinal_pipeline, ordinal_features),
                    ("nom", nominal_pipeline, nominal_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    # ================= MAIN TRANSFORMATION =================
    def initiate_data_transformation(self, train_path, test_path):
        """
        Executes full transformation pipeline
        """
        try:
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded")

            # ================= FEATURE ENGINEERING =================
            train_df = self.apply_feature_engineering(train_df)
            test_df = self.apply_feature_engineering(test_df)

            # ================= TARGET MAPPING =================
            train_df["risk"] = train_df["risk"].map({"bad": 1, "good": 0})
            test_df["risk"] = test_df["risk"].map({"bad": 1, "good": 0})

            if train_df["risk"].isnull().any():
                raise ValueError("Target mapping produced NaN values")

            logging.info("Target mapping completed")

            # ================= SPLIT FEATURES =================
            target_column = "risk"

            X_train = train_df.drop(columns=[target_column, "credit_amount"])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column, "credit_amount"])
            y_test = test_df[target_column]

            # ================= PREPROCESSING =================
            preprocessor = self.get_data_transformer_object()

            logging.info("Applying preprocessing pipeline")

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            # Combine X and y
            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            # ================= SAVE PREPROCESSOR =================
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            logging.info(f"Preprocessor saved at {self.config.preprocessor_obj_file_path}")

            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)