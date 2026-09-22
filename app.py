"""Streamlit inference application for the saved vehicle-fraud pipeline.

The serialized pipeline owns every trained preprocessing operation.  This
module only validates raw claim data, puts it in the pipeline's recorded
column order, and calls predict_proba; it never fits or rebuilds transforms.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "vehicle_insurance_fraud_best_model.pkl"
THRESHOLD_PATH = PROJECT_ROOT / "model" / "threshold.json"
SRC_PATH = PROJECT_ROOT / "src"

LOGGER = logging.getLogger(__name__)

# These are the raw fields used when the saved pipeline was trained.  The two
# fields marked internal are present in the raw schema but are explicitly
# dropped by FraudCleaner before model preprocessing.
INTERNAL_DEFAULTS: dict[str, Any] = {"claim_number": 0, "claim_day_of_week": ""}
# FraudCleaner is the first pipeline step and deliberately has no
# feature_names_in_. This raw contract is therefore taken from the project's
# training module/data schema and is verified against the fitted downstream
# ColumnTransformer when the artifact is loaded.
RAW_INPUT_COLUMNS = [
    "claim_number", "age_of_driver", "gender", "marital_status", "safety_rating",
    "annual_income", "high_education", "address_change", "property_status", "zip_code",
    "claim_date", "claim_day_of_week", "accident_site", "past_num_of_claims",
    "witness_present", "liab_prct", "channel", "police_report", "age_of_vehicle",
    "vehicle_category", "vehicle_price", "vehicle_color", "total_claim", "injury_claim",
    "policy deductible", "annual premium", "days open", "form defects",
]
CLEANED_FEATURE_COLUMNS = [
    "age_of_driver", "gender", "marital_status", "safety_rating", "annual_income",
    "high_education", "address_change", "property_status", "accident_site",
    "past_num_of_claims", "witness_present", "liab_prct", "channel", "police_report",
    "age_of_vehicle", "vehicle_category", "vehicle_price", "vehicle_color", "total_claim",
    "injury_claim", "policy deductible", "annual premium", "days open", "form defects",
    "claim_month", "claim_weekday", "claim_year", "zip_region", "claim_to_price",
    "injury_share", "claim_to_income", "premium_to_income", "claim_to_deductible",
]
CATEGORY_OPTIONS = {
    "gender": ["M", "F"],
    "marital_status": ["0", "1"],
    "property_status": ["Own", "Rent"],
    "accident_site": ["Highway", "Local", "Parking Lot"],
    "channel": ["Phone", "Online", "Broker"],
    "vehicle_category": ["Large", "Medium", "Compact"],
    "vehicle_color": ["silver", "black", "gray", "red", "white", "blue", "other"],
}
BINARY_FIELDS = {
    "high_education": "Higher education",
    "address_change": "Address changed recently",
    "witness_present": "Witness present",
    "police_report": "Police report filed",
}
NUMERIC_FIELDS = {
    "age_of_driver": ("Driver age", 16.0, 100.0, 1.0, 40.0),
    "safety_rating": ("Safety rating", 0.0, 100.0, 1.0, 70.0),
    "annual_income": ("Annual income", 1.0, 10_000_000.0, 100.0, 60_000.0),
    "zip_code": ("ZIP code", 0.0, 99999.0, 1.0, 50000.0),
    "past_num_of_claims": ("Previous claims", 0.0, 100.0, 1.0, 0.0),
    "liab_prct": ("Liability percentage", 0.0, 100.0, 1.0, 50.0),
    "age_of_vehicle": ("Vehicle age (years)", 0.0, 100.0, 1.0, 5.0),
    "vehicle_price": ("Vehicle price", 1.0, 10_000_000.0, 100.0, 25_000.0),
    "total_claim": ("Total claim amount", 0.0, 10_000_000.0, 100.0, 20_000.0),
    "injury_claim": ("Injury claim amount", 0.0, 10_000_000.0, 100.0, 4_000.0),
    "policy deductible": ("Policy deductible", 0.0, 1_000_000.0, 100.0, 1_000.0),
    "annual premium": ("Annual premium", 0.0, 1_000_000.0, 10.0, 1_200.0),
    "days open": ("Days claim open", 0.0, 10_000.0, 0.1, 9.0),
    "form defects": ("Form defects", 0.0, 100.0, 1.0, 0.0),
}


@st.cache_resource(show_spinner="Loading the fraud detection pipeline…")
def load_artifacts() -> tuple[Any, float, list[str]]:
    """Load the one approved model pipeline and its saved decision threshold."""
    if not MODEL_PATH.is_file() or not THRESHOLD_PATH.is_file():
        raise FileNotFoundError("Required model artifacts are unavailable.")
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))
    # Required before unpickling so FraudCleaner and QuantileClipper resolve.
    import fraud_preprocessing  # noqa: F401

    pipeline = joblib.load(MODEL_PATH)
    try:
        fitted_processed_columns = list(pipeline.named_steps["prep"].feature_names_in_)
    except (AttributeError, KeyError) as error:
        raise ValueError("The saved pipeline does not expose its fitted preprocessing schema.") from error
    if fitted_processed_columns != CLEANED_FEATURE_COLUMNS:
        raise ValueError("The saved pipeline schema differs from this approved form contract.")
    with THRESHOLD_PATH.open(encoding="utf-8") as handle:
        threshold = float(json.load(handle)["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("The saved model threshold is invalid.")
    return pipeline, threshold, RAW_INPUT_COLUMNS


def prepare_input(values: dict[str, Any], expected_columns: list[str]) -> pd.DataFrame:
    """Build the exact raw schema/order recorded by the fitted pipeline."""
    row = {**INTERNAL_DEFAULTS, **values}
    missing = [column for column in expected_columns if column not in row]
    extra = set(row).difference(expected_columns)
    if missing or extra:
        raise ValueError("The form schema does not match the saved model schema.")
    return pd.DataFrame([[row[column] for column in expected_columns]], columns=expected_columns)


def predict_claim(pipeline: Any, features: pd.DataFrame, threshold: float) -> tuple[bool, float, float]:
    """Score with the saved pipeline; class 1 is fraud per clean_target()."""
    if not hasattr(pipeline, "predict_proba"):
        raise ValueError("This deployment requires probability predictions for its saved threshold.")
    classes = list(pipeline.classes_)
    if 1 not in classes:
        raise ValueError("The pipeline has no recorded fraud class (1).")
    probabilities = pipeline.predict_proba(features)[0]
    fraud_probability = float(probabilities[classes.index(1)])
    return fraud_probability >= threshold, fraud_probability, 1.0 - fraud_probability


def calculate_local_impacts(
    pipeline: Any,
    values: dict[str, Any],
    expected_columns: list[str],
    baseline_fraud_probability: float,
) -> pd.DataFrame:
    """Calculate transparent one-at-a-time what-if changes for this claim.

    This is local sensitivity analysis, not global feature importance. Each
    scenario is scored by the unchanged fitted pipeline; no transformation is
    fitted and the original submitted claim remains the comparison baseline.
    """
    scenarios: list[dict[str, Any]] = []
    descriptions: list[str] = []
    for field in expected_columns:
        if field in INTERNAL_DEFAULTS:
            continue  # FraudCleaner drops these fields before prediction.
        scenario = values.copy()
        if field in NUMERIC_FIELDS:
            _, minimum, maximum, step, current = NUMERIC_FIELDS[field]
            current_value = float(scenario[field])
            changed_value = min(maximum, current_value * 1.10 if current_value else minimum + step)
            if changed_value == current_value:
                changed_value = max(minimum, current_value - step)
            scenario[field] = changed_value
            descriptions.append(f"Change from {current_value:g} to {changed_value:g}")
        elif field in BINARY_FIELDS:
            scenario[field] = 1 - int(scenario[field])
            descriptions.append(f"Switch to {'Yes' if scenario[field] else 'No'}")
        elif field in CATEGORY_OPTIONS:
            choices = CATEGORY_OPTIONS[field]
            current_index = choices.index(scenario[field])
            scenario[field] = choices[(current_index + 1) % len(choices)]
            descriptions.append(f"Switch to {scenario[field]}")
        elif field == "claim_date":
            current_date = date.fromisoformat(str(scenario[field])) if isinstance(scenario[field], str) and "-" in str(scenario[field]) else None
            if current_date is None:
                # The form stores a training-compatible M/D/YYYY date string.
                month, day, year = (int(part) for part in str(scenario[field]).split("/"))
                current_date = date(year, month, day)
            changed_date = current_date + timedelta(days=30)
            scenario[field] = f"{changed_date.month}/{changed_date.day}/{changed_date.year}"
            descriptions.append("Move claim date 30 days later")
        else:
            continue
        scenarios.append(scenario)

    scenario_frame = pd.concat(
        [prepare_input(scenario, expected_columns) for scenario in scenarios], ignore_index=True
    )
    classes = list(pipeline.classes_)
    fraud_index = classes.index(1)
    scenario_probabilities = pipeline.predict_proba(scenario_frame)[:, fraud_index]
    fields = [field for field in expected_columns if field not in INTERNAL_DEFAULTS]
    deltas = (scenario_probabilities - baseline_fraud_probability) * 100
    return pd.DataFrame(
        {
            "Field": [field.replace("_", " ").title() for field in fields],
            "What-if change": descriptions,
            "Fraud probability": [f"{value:.2%}" for value in scenario_probabilities],
            "Change": [f"{delta:+.2f} percentage points" for delta in deltas],
        }
    )


def number_widget(key: str) -> float:
    label, minimum, maximum, step, value = NUMERIC_FIELDS[key]
    return st.number_input(label, min_value=minimum, max_value=maximum, value=value, step=step, key=key)


def render_form(expected_columns: list[str]) -> dict[str, Any] | None:
    required = set(expected_columns) - set(INTERNAL_DEFAULTS)
    supported = set(CATEGORY_OPTIONS) | set(BINARY_FIELDS) | set(NUMERIC_FIELDS) | {"claim_date"}
    unsupported = required.difference(supported)
    if unsupported:
        raise ValueError("The saved model includes fields without an approved form specification.")

    with st.form("claim_form"):
        st.subheader("Policy and customer information")
        left, right = st.columns(2)
        with left:
            age = number_widget("age_of_driver")
            income = number_widget("annual_income")
            gender = st.selectbox("Gender", CATEGORY_OPTIONS["gender"])
            marital_status = st.selectbox("Marital status", CATEGORY_OPTIONS["marital_status"], format_func=lambda x: "Married" if x == "1" else "Not married")
            high_education = st.radio(BINARY_FIELDS["high_education"], [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
            address_change = st.radio(BINARY_FIELDS["address_change"], [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
        with right:
            safety_rating = number_widget("safety_rating")
            property_status = st.selectbox("Property status", CATEGORY_OPTIONS["property_status"])
            zip_code = number_widget("zip_code")
            past_num_of_claims = number_widget("past_num_of_claims")
            liab_prct = number_widget("liab_prct")
            annual_premium = number_widget("annual premium")
            policy_deductible = number_widget("policy deductible")

        st.subheader("Vehicle information")
        left, right = st.columns(2)
        with left:
            age_of_vehicle = number_widget("age_of_vehicle")
            vehicle_category = st.selectbox("Vehicle category", CATEGORY_OPTIONS["vehicle_category"])
            vehicle_price = number_widget("vehicle_price")
        with right:
            vehicle_color = st.selectbox("Vehicle color", CATEGORY_OPTIONS["vehicle_color"])

        st.subheader("Claim information")
        left, right = st.columns(2)
        with left:
            claim_date = st.date_input("Claim date", value=date.today())
            accident_site = st.selectbox("Accident site", CATEGORY_OPTIONS["accident_site"])
            channel = st.selectbox("Reporting channel", CATEGORY_OPTIONS["channel"])
            total_claim = number_widget("total_claim")
            injury_claim = number_widget("injury_claim")
        with right:
            witness_present = st.radio(BINARY_FIELDS["witness_present"], [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
            police_report = st.radio(BINARY_FIELDS["police_report"], [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
            days_open = number_widget("days open")
            form_defects = number_widget("form defects")

        submitted = st.form_submit_button("Predict Claim", type="primary", use_container_width=True)
    if not submitted:
        return None
    return {
        "age_of_driver": age, "gender": gender, "marital_status": marital_status,
        "safety_rating": safety_rating, "annual_income": income, "high_education": high_education,
        "address_change": address_change, "property_status": property_status, "zip_code": zip_code,
        "claim_date": f"{claim_date.month}/{claim_date.day}/{claim_date.year}", "accident_site": accident_site,
        "past_num_of_claims": past_num_of_claims, "witness_present": witness_present,
        "liab_prct": liab_prct, "channel": channel, "police_report": police_report,
        "age_of_vehicle": age_of_vehicle, "vehicle_category": vehicle_category,
        "vehicle_price": vehicle_price, "vehicle_color": vehicle_color, "total_claim": total_claim,
        "injury_claim": injury_claim, "policy deductible": policy_deductible,
        "annual premium": annual_premium, "days open": days_open, "form defects": form_defects,
    }


def main() -> None:
    st.set_page_config(page_title="Claim Guard", page_icon="🛡️", layout="centered")
    st.title("🛡️ Claim Guard")
    st.write(
        "This model was trained on historical vehicle-insurance claims using a held-out evaluation split. "
        "Its fitted pipeline cleans raw data, engineers claim and policy features, imputes missing values, "
        "encodes categories, clips outliers, and scores the claim with HistGradientBoosting."
    )
    st.write(
        "For every prediction, Claim Guard sends your entered raw claim data through that same saved pipeline. "
        "Nothing is retrained or re-fitted in the application, ensuring training-time and inference-time preprocessing are identical."
    )
    st.sidebar.header("Model information")
    st.sidebar.write("HistGradientBoostingClassifier with its fitted preprocessing pipeline.")
    st.sidebar.write("Claims are flagged at the saved fraud threshold of 18%.")
    try:
        pipeline, threshold, expected_columns = load_artifacts()
        st.sidebar.caption(f"Validated raw input schema: {len(expected_columns)} columns")
        values = render_form(expected_columns)
        if values is not None:
            features = prepare_input(values, expected_columns)
            flagged, fraud_probability, valid_probability = predict_claim(pipeline, features, threshold)
            if flagged:
                st.error("⚠️ Fraudulent Claim Detected")
            else:
                st.success("✅ Claim Appears Valid")
            first, second, third = st.columns(3)
            first.metric("Fraud probability", f"{fraud_probability:.2%}")
            second.metric("Valid probability", f"{valid_probability:.2%}")
            third.metric("Decision threshold", f"{threshold:.0%}")
            with st.expander("How each entered field affects this prediction", expanded=False):
                st.caption(
                    "Each row changes only the stated field while keeping every other submitted value fixed. "
                    "The change is the resulting fraud-probability difference in percentage points; it is specific "
                    "to this claim and is not a global causal effect."
                )
                impacts = calculate_local_impacts(pipeline, values, expected_columns, fraud_probability)
                st.dataframe(impacts, use_container_width=True, hide_index=True)
                st.caption("Claim number and claim day of week are omitted because the saved preprocessing pipeline drops them before scoring.")
    except (FileNotFoundError, ValueError, KeyError, TypeError) as error:
        LOGGER.exception("Application configuration or inference error")
        st.error("The prediction service is currently unavailable. Please contact the application owner.")
    except Exception:
        LOGGER.exception("Unexpected prediction error")
        st.error("Unable to score this claim. Please review the entered values and try again.")
    st.divider()
    st.caption("Decision-support only. A fraud flag requires human review; this model has not been validated for live claims use or bias/fairness.")


if __name__ == "__main__":
    main()
