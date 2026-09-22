"""Custom, sklearn-compatible preprocessing for the vehicle insurance fraud dataset.
The saved model pipeline imports this module, so keep it importable (see README)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TARGET = "fraud reported"
POSITIVE = "Y"
DROP_COLS = ["claim_number", "claim_day_of_week", TARGET]  # ID, redundant with claim_date, target
PLACEHOLDERS = ["*", "?", "", "NA", "N/A", "nan"]
NUM_COLS = ["age_of_driver", "safety_rating", "annual_income", "past_num_of_claims", "liab_prct",
            "age_of_vehicle", "vehicle_price", "total_claim", "injury_claim", "policy deductible",
            "annual premium", "days open", "form defects"]
BIN_COLS = ["high_education", "address_change", "police_report", "marital_status", "witness_present"]
CAT_COLS = ["gender", "property_status", "accident_site", "channel", "vehicle_category", "vehicle_color"]


def load_raw(path):
    return pd.read_csv(path)


def clean_target(df):
    """Drop rows with missing target; return X (raw columns kept) and binary y (1 = fraud)."""
    df = df.copy()
    df[TARGET] = df[TARGET].replace(PLACEHOLDERS, np.nan)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    y = (df[TARGET] == POSITIVE).astype(int)
    return df.drop(columns=[TARGET]), y


class FraudCleaner(BaseEstimator, TransformerMixin):
    """Stateless cleaning + feature engineering (no statistics learned, so no leakage).
    - '*' placeholders -> NaN, numeric coercion
    - invalid values -> NaN (driver age outside 16-100, income <= 0)
    - claim_date -> month / weekday / year (parse failures -> NaN)
    - engineered ratios (see README)"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X = X.drop(columns=[c for c in DROP_COLS if c in X.columns])
        X = X.replace(PLACEHOLDERS, np.nan)
        for c in NUM_COLS + BIN_COLS:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X.loc[(X["age_of_driver"] < 16) | (X["age_of_driver"] > 100), "age_of_driver"] = np.nan
        X.loc[X["annual_income"] <= 0, "annual_income"] = np.nan
        d = pd.to_datetime(X["claim_date"], errors="coerce", format="%m/%d/%Y")
        X["claim_month"] = d.dt.month
        X["claim_weekday"] = d.dt.dayofweek
        X["claim_year"] = d.dt.year
        X["zip_region"] = (X["zip_code"] // 10000)  # coarse region from the leading digit
        X = X.drop(columns=["claim_date", "zip_code"])
        eps = 1e-6
        X["claim_to_price"] = X["total_claim"] / (X["vehicle_price"] + eps)
        X["injury_share"] = X["injury_claim"] / (X["total_claim"] + eps)
        X["claim_to_income"] = X["total_claim"] / (X["annual_income"] + eps)
        X["premium_to_income"] = X["annual premium"] / (X["annual_income"] + eps)
        X["claim_to_deductible"] = X["total_claim"] / (X["policy deductible"] + eps)
        X = X.replace([np.inf, -np.inf], np.nan)
        return X


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorise numeric columns at train-fitted quantiles (rows are never removed)."""

    def __init__(self, lower=0.01, upper=0.99, columns=None):
        self.lower, self.upper, self.columns = lower, upper, columns

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(self.columns) if self.columns is not None else list(X.columns)
        self.lo_ = X[self.columns_].quantile(self.lower)
        self.hi_ = X[self.columns_].quantile(self.upper)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X[self.columns_] = X[self.columns_].clip(self.lo_, self.hi_, axis=1)
        return X
