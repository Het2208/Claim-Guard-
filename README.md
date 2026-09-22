# Vehicle Insurance Fraud Detection

## Problem Definition

Predict whether a vehicle insurance claim is fraudulent from claim,
policyholder, and vehicle attributes recorded at claim time. This is a
binary classification problem on an imbalanced target, where the cost of
missing fraud (false negative) is generally treated as more serious than
the cost of investigating a legitimate claim (false positive) — but that
trade-off is a business decision, not a modeling one, so this project
reports the full precision/recall curve rather than assuming one answer.

## Dataset

`data/original_dataset.csv` — 12,002 rows, 29 columns, one row per claim.
Columns include driver demographics (age, gender, marital status),
policy attributes (annual premium, deductible, policy duration signals),
claim attributes (total claim, injury claim, claim date, accident site,
witness presence, police report), and vehicle attributes (age, price,
category, color).

**Target column:** `fraud reported` (`Y` / `N`). 8 rows had no valid label
and were dropped before modeling, leaving **11,994 labelled rows**.

**Class imbalance:** 75.4% not fraud (`N`) vs. **24.6% fraud (`Y`)**.

## Preprocessing

See `results/preprocessing_summary.txt` for the full write-up. Summary:
- `'*'` placeholders and physically-invalid values (driver age outside
  16–100, non-positive income) converted to `NaN`.
- Dropped `claim_number` (row ID) and `claim_day_of_week` (redundant with
  `claim_date`).
- Numeric imputation: median. Categorical/binary imputation: most-frequent.
  **Both are fit on training data only**, inside the saved pipeline.
- Skewed numeric columns (income, claim amounts, engineered ratios):
  Yeo-Johnson transform + standardization. Roughly-symmetric numeric
  columns: standardization only.
- Categorical columns: one-hot encoding (unknown categories at inference
  time are ignored, not errored on).
- 5 engineered ratio/date features (see below).

## Outlier Strategy

Numeric columns are **winsorized** (clipped) at the 1st/99th percentile,
fit on training data only. **No rows were deleted** for being statistical
outliers — unusually large claims can themselves be a fraud signal, so
capping was used instead of removal to avoid discarding real (if rare)
fraud cases.

## Feature Engineering

- `claim_month`, `claim_weekday`, `claim_year` — parsed from `claim_date`.
- `zip_region` — coarse region bucket (`zip_code // 10000`); the raw
  5-digit zip code was dropped as too high-cardinality to encode safely.
- `claim_to_price` = total_claim / vehicle_price
- `injury_share` = injury_claim / total_claim
- `claim_to_income` = total_claim / annual_income
- `premium_to_income` = annual premium / annual_income
- `claim_to_deductible` = total_claim / policy deductible

No target leakage: all engineered features are derived only from
information available at claim time, not from the outcome.

## Models Evaluated

38 model / sampling-strategy / PCA combinations were compared by 5-fold
stratified cross-validation on the training split (`results/model_comparison.csv`):

- **Standard classifiers:** Logistic Regression (plain and class-weighted),
  KNN, Gaussian Naive Bayes, Decision Tree, SVC (RBF), Linear SVC, SGD
  (log loss), Perceptron, MLP.
- **Ensembles:** Random Forest, Extra Trees, AdaBoost, Gradient Boosting,
  HistGradientBoosting, Bagging, Voting (soft), Stacking.
- **Imbalanced-learn ensembles:** BalancedRandomForest, BalancedBagging,
  EasyEnsemble. (RUSBoost failed to fit in both attempts — its underlying
  AdaBoost base learner was rejected as worse than random on this data —
  and is recorded as a failure, not silently dropped.)
- Promising candidates (Logistic Regression, HistGradientBoosting,
  LightGBM, Gradient Boosting, Random Forest) were then tuned with
  randomized search (`results/model_comparison.csv`, rows marked
  `2-tuned`).

## Imbalance Techniques Evaluated

`class_weight='balanced'`, `RandomUnderSampler`, `RandomOverSampler`,
`SMOTENC` (categorical-aware SMOTE), and `SMOTENC + RandomUnderSampler`
combinations were tested on the two strongest model families (rows marked
`3-imbalance/PCA`). **None improved fraud PR-AUC or fraud F1** over the
unmodified class distribution — the final model uses no resampling and
instead handles the imbalance purely through decision-threshold selection.

PCA (95% explained variance) was also tested and **rejected**: it reduced
PR-AUC on both a linear and a tree-based model.

## Final Model

**HistGradientBoostingClassifier** (scikit-learn), with a single pipeline
handling all preprocessing. Default-style hyperparameters
(`max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0`)
performed as well as or better than every tuned, resampled, or ensembled
alternative tested, so the simpler configuration was kept.

## Final Evaluation Metrics (held-out 20% test set, n=2,399, evaluated once)

| Metric | Value |
|---|---|
| Fraud Precision | 0.282 |
| Fraud Recall | 0.946 |
| Fraud F1 | 0.435 |
| ROC-AUC | 0.645 |
| PR-AUC (Average Precision) | 0.412 |
| Selected threshold | 0.18 |

Full threshold sweep (0.05–0.60) is in `results/threshold_search.csv`, and
the reasoning behind model/threshold selection is in
`results/best_model_summary.txt`.

**Honest caveat:** this dataset's fraud signal is weak and concentrated
almost entirely in `annual_income` (fraud rate is ~90–100% below ~$40k,
~25% in the $40k–$70k range, under 5% above $80k); nearly every other
column is close to uninformative on its own. The model is a modest
improvement over a naive baseline, not a strong fraud detector, and it has
not been validated for production use or checked for demographic bias
given its reliance on income. Treat it as a starting point, not a
deployable system.

## Project Structure

```
vehicle_insurance_fraud_project/
├── data/
│   ├── original_dataset.csv
│   └── preprocessed_vehicle_insurance_fraud.csv
├── notebook/
│   └── vehicle_insurance_fraud_best_model.ipynb
├── model/
│   ├── vehicle_insurance_fraud_best_model.pkl
│   └── threshold.json
├── results/
│   ├── model_comparison.csv
│   ├── best_model_summary.txt
│   ├── preprocessing_summary.txt
│   ├── threshold_search.csv
│   └── test_metrics.json
├── src/
│   └── fraud_preprocessing.py    # required to load the .pkl (see below)
├── app.py                         # Streamlit inference application
├── requirements.txt
└── README.md
```

## Streamlit Claim-Scoring Application

`app.py` provides a local or Streamlit Community Cloud interface for scoring
one vehicle claim at a time. It loads only
`model/vehicle_insurance_fraud_best_model.pkl`, the selected final artifact,
and uses the saved `model/threshold.json` decision threshold (0.18).

The model is a fitted scikit-learn pipeline: `FraudCleaner`, the fitted
column preprocessing transformer, and `HistGradientBoostingClassifier`.
Consequently, the app submits the raw fields in the artifact's recorded order
and calls the pipeline directly. It does **not** recreate, fit, or infer any
encoding, imputation, scaling, clipping, feature engineering, PCA, or
resampling at prediction time. `claim_number` and `claim_day_of_week` are
raw-schema fields deliberately discarded by `FraudCleaner`, so they are not
shown to users.

The form contains the remaining raw claim attributes used by the pipeline:
customer/policy details, vehicle characteristics, date and location, claim
amounts, and reporting details. Categorical choices match the training data;
the saved encoder still handles unknown categories as configured in the
pipeline.

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The artifact was created with `pandas==3.0.2` and `scikit-learn==1.8.0`;
retain the pinned versions in `requirements.txt` when deploying, as model
pickles are not reliably portable across library versions. The pinned
Streamlit release supports pandas 3.

### Deploy to Streamlit Community Cloud

1. Push this complete project directory, including `model/` and `src/`, to a
   Git repository.
2. Create a Community Cloud app pointing to `app.py` on the desired branch.
3. Use the repository `requirements.txt`; no secrets or machine-specific
   paths are required.

Keep the model artifact and `src/fraud_preprocessing.py` together: the custom
classes must be importable before the pipeline is unpickled. The application
uses paths relative to its own root, so it is compatible with local runs and
Community Cloud.

## How to Install Dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run the Notebook

```bash
jupyter notebook notebook/vehicle_insurance_fraud_best_model.ipynb
```

The notebook loads `data/original_dataset.csv`, rebuilds and trains the
final pipeline, evaluates it once on a held-out test split, and re-saves
`model/vehicle_insurance_fraud_best_model.pkl`. It does not retrain or
compare the other 37 models — that comparison already lives in
`results/model_comparison.csv`.

## How to Load the `.pkl` Model

The saved pipeline depends on the custom classes in
`src/fraud_preprocessing.py` (`FraudCleaner`, `QuantileClipper`), so that
file must be importable — keep `src/` alongside your script, or add it to
`sys.path` — before unpickling.

```python
import sys, json
sys.path.insert(0, "src")
import joblib
import pandas as pd
from fraud_preprocessing import load_raw, clean_target  # registers the custom classes

model = joblib.load("model/vehicle_insurance_fraud_best_model.pkl")
threshold = json.load(open("model/threshold.json"))["threshold"]

# Example: score new raw claims (same raw column schema as original_dataset.csv,
# minus the target column)
new_claims = pd.read_csv("data/original_dataset.csv").drop(columns=["fraud reported"]).head(5)
proba_fraud = model.predict_proba(new_claims)[:, 1]
is_fraud = (proba_fraud >= threshold).astype(int)

print(pd.DataFrame({"fraud_probability": proba_fraud, "flagged_as_fraud": is_fraud}))
```

## Not Production-Ready

This model runs end-to-end and its metrics are real, measured results —
but running successfully is not the same as being production-ready. It
has not been tested for fairness/bias (particularly around its reliance
on income), has not been validated against a live claims workflow, and
was trained on a single static extract. Any real deployment would need
further validation, monitoring, and a deliberate, business-driven choice
of operating threshold.
