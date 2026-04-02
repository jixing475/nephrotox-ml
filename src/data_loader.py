from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

from . import config


def load_data_full(file_path) -> pd.DataFrame:
    """Load the full dataframe (keeps ID/label for flexible splitting)."""
    return pd.read_csv(file_path)


def handle_missing_values(X: pd.DataFrame, missing_threshold: float = 0.5):
    """Replace inf with NaN, drop columns with too many missing, median-impute the rest."""
    X = X.replace([np.inf, -np.inf], np.nan)
    missing_ratio = X.isnull().sum() / len(X)
    cols_to_drop = missing_ratio[missing_ratio > missing_threshold].index.tolist()
    X = X.drop(columns=cols_to_drop)

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index,
    )
    return X_imputed, imputer, cols_to_drop


def remove_low_variance(X: pd.DataFrame, threshold: float = 0.0):
    selector = VarianceThreshold(threshold=threshold)
    X_var = selector.fit_transform(X)
    selected_features = selector.get_feature_names_out()
    X_var_df = pd.DataFrame(X_var, columns=selected_features, index=X.index)
    return X_var_df, selector


def remove_correlated_features(X: pd.DataFrame, threshold: float = 0.95):
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    cols_to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
    X_reduced = X.drop(columns=cols_to_drop)
    return X_reduced, cols_to_drop


class FeatureEngineer:
    """Reusable feature engineering pipeline.

    Steps:
      1) Handle missing/inf (drop > threshold missing, median impute)
      2) Remove low variance
      3) Remove highly correlated
      4) Standardize features (for SVM compatibility)
    """

    def __init__(
        self,
        variance_threshold: float = config.VARIANCE_THRESHOLD,
        corr_threshold: float = config.CORRELATION_THRESHOLD,
        missing_threshold: float = config.MISSING_VALUE_THRESHOLD,
        clip_range: tuple = (-10.0, 10.0),
    ):
        self.variance_threshold = variance_threshold
        self.corr_threshold = corr_threshold
        self.missing_threshold = missing_threshold
        self.clip_range = clip_range

        self.imputer: SimpleImputer | None = None
        self.variance_selector: VarianceThreshold | None = None
        self.corr_cols_to_drop: List[str] | None = None
        self.remaining_features: List[str] | None = None
        self.scaler: StandardScaler | None = None
        self._imputed_columns: List[str] | None = None

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        n_initial = X.shape[1]
        print(f"Feature engineering: starting with {n_initial} features")
        
        # Step 1: Missing/inf handling
        X, self.imputer, dropped_missing = handle_missing_values(X, self.missing_threshold)
        self._imputed_columns = X.columns.tolist()
        print(f"  After missing value handling: {X.shape[1]} features ({len(dropped_missing)} dropped)")

        # Step 2: Low variance
        X_var_df, self.variance_selector = remove_low_variance(X, self.variance_threshold)
        n_low_var_dropped = X.shape[1] - X_var_df.shape[1]
        print(f"  After low variance removal: {X_var_df.shape[1]} features ({n_low_var_dropped} dropped)")

        # Step 3: High correlation
        X_corr_reduced, self.corr_cols_to_drop = remove_correlated_features(
            X_var_df, self.corr_threshold
        )
        self.remaining_features = X_corr_reduced.columns.tolist()
        print(f"  After correlation removal: {X_corr_reduced.shape[1]} features ({len(self.corr_cols_to_drop)} dropped)")

        # Step 4: Scaling
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_corr_reduced),
            columns=self.remaining_features,
            index=X.index,
        )

        # Step 5: Post-scaling clipping to prevent extreme values
        if self.clip_range:
            X_scaled = X_scaled.clip(lower=self.clip_range[0], upper=self.clip_range[1])

        print(f"  Final feature count: {X_scaled.shape[1]} (total removed: {n_initial - X_scaled.shape[1]})")
        return X_scaled

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if any(v is None for v in [self.imputer, self.variance_selector, self.corr_cols_to_drop, self.scaler]):
            raise RuntimeError("FeatureEngineer must be fitted before calling transform().")

        # Replace inf, drop any columns removed during fit, and align ordering
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.reindex(columns=self._imputed_columns)
        X_imputed = pd.DataFrame(
            self.imputer.transform(X),
            columns=self._imputed_columns,
            index=X.index,
        )

        # Low variance transform
        X_var = self.variance_selector.transform(X_imputed)
        var_features = self.variance_selector.get_feature_names_out()
        X_var_df = pd.DataFrame(X_var, columns=var_features, index=X.index)

        # Drop correlated columns learned during fit
        X_reduced = X_var_df.drop(columns=[c for c in self.corr_cols_to_drop if c in X_var_df.columns])

        # Ensure column order matches training
        X_reduced = X_reduced[self.remaining_features]

        # Scale
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_reduced),
            columns=self.remaining_features,
            index=X.index,
        )

        # Post-scaling clipping to prevent extreme values
        if self.clip_range:
            X_scaled = X_scaled.clip(lower=self.clip_range[0], upper=self.clip_range[1])

        return X_scaled


def split_features_labels(df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    ids = df["ID"]
    X = df.drop(columns=["ID", "SMILES", "label"], errors="ignore")
    y = df["label"]
    return ids, X, y
