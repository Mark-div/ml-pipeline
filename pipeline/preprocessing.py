import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    target_column: str = "target"
    drop_columns: list = None
    categorical_threshold: int = 20  # Max unique values to treat as categorical


class DataPreprocessor:
    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        self.scaler = StandardScaler()
        self.label_encoders: dict = {}
        self.feature_columns: list = []
        self.categorical_columns: list = []
        self.numerical_columns: list = []
        self._fitted = False

    def _detect_column_types(self, df: pd.DataFrame) -> Tuple[list, list]:
        categorical, numerical = [], []
        for col in df.columns:
            if col == self.config.target_column:
                continue
            if df[col].dtype == "object" or df[col].nunique() <= self.config.categorical_threshold:
                categorical.append(col)
            else:
                numerical.append(col)
        return categorical, numerical

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.numerical_columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        for col in self.categorical_columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        df = df.copy()
        for col in self.categorical_columns:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    # Handle unseen labels gracefully
                    df[col] = df[col].astype(str).apply(
                        lambda x: x if x in le.classes_ else le.classes_[0]
                    )
                    df[col] = le.transform(df[col])
        return df

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                        np.ndarray, np.ndarray, np.ndarray]:
        logger.info(f"Preprocessing dataset: {df.shape}")
        drop_cols = self.config.drop_columns or []
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        self.categorical_columns, self.numerical_columns = self._detect_column_types(df)
        logger.info(f"Categorical: {self.categorical_columns}")
        logger.info(f"Numerical: {self.numerical_columns}")

        df = self._handle_missing(df)
        df = self._encode_categoricals(df, fit=True)

        target = df[self.config.target_column].values
        self.feature_columns = [c for c in df.columns if c != self.config.target_column]
        features = df[self.feature_columns].values

        # Scale numerical features
        num_indices = [self.feature_columns.index(c) for c in self.numerical_columns if c in self.feature_columns]
        if num_indices:
            features[:, num_indices] = self.scaler.fit_transform(features[:, num_indices])

        self._fitted = True

        # Split
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, target,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=target if len(np.unique(target)) < 100 else None,
        )
        val_ratio = self.config.val_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=self.config.random_state,
        )

        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert self._fitted, "Call fit_transform first"
        df = self._handle_missing(df)
        df = self._encode_categoricals(df, fit=False)
        features = df[self.feature_columns].values
        num_indices = [self.feature_columns.index(c) for c in self.numerical_columns if c in self.feature_columns]
        if num_indices:
            features[:, num_indices] = self.scaler.transform(features[:, num_indices])
        return features
