# Random Forest — Cardiovascular Event Prediction
### BSAN6070: Introduction to Machine Learning | Spring 2026 | Loyola Marymount University

**Research Question:** Can cardiovascular event occurrence be accurately predicted using sleep metrics, demographic variables, and baseline health indicators?

---

## Overview

This notebook builds a tuned Random Forest classifier on the Sleep Heart Health Study (SHHS) dataset to predict whether a patient experienced a cardiovascular event. It covers data loading, exploratory analysis, preprocessing, hyperparameter tuning, model evaluation, and feature importance analysis using three methods: MDI, permutation importance, and SHAP.

---

## Repository Structure

```
├── random_forest_final_project.ipynb   # Main notebook
├── model_ready_cvd_prediction.csv      # Input dataset (place in same directory)
├── rf_cvd_model.sav                    # Trained Random Forest model
├── scaler.sav                          # StandardScaler fit on training set
├── feature_names.json                  # Ordered feature list for inference
├── final_metrics.json                  # Test set performance metrics
└── README.md
```

---

## Dataset

| Property | Value |
|---|---|
| Source | Sleep Heart Health Study (SHHS) v0.13.0 |
| File | `model_ready_cvd_prediction.csv` |
| Patients | 5,042 |
| Features | 77 (continuous health/sleep metrics + binary demographic indicators) |
| Missing values | 0 |
| Target | `cvd_event` — 1 = CVD event occurred, 0 = no event |
| Class distribution | 76.3% no event / 23.7% event (3.2:1 imbalance) |

---

## Methodology

**Preprocessing**
- 80/20 stratified train/test split (preserves 23.7% event rate in both sets)
- `StandardScaler` fit on training data only, applied to test
- Class imbalance handled via `class_weight='balanced'` in the Random Forest — no synthetic oversampling

**Hyperparameter Tuning**
- `RandomizedSearchCV` — 60 iterations, 5-fold stratified cross-validation, optimizing AUC-ROC

**Best hyperparameters found:**

| Parameter | Value |
|---|---|
| `n_estimators` | 500 |
| `max_depth` | 20 |
| `max_features` | 0.3 |
| `min_samples_split` | 15 |
| `min_samples_leaf` | 6 |
| `criterion` | entropy |

---

## Results

| Metric | Baseline RF | Tuned RF |
|---|---|---|
| **Accuracy** | 0.7830 | 0.7750 |
| **AUC** | 0.7862 | **0.7893** |
| Precision | 0.7381 | 0.5250 |
| Recall | 0.1297 | **0.5272** |
| F1-Score | 0.2206 | **0.5261** |
| Specificity | — | 0.8519 |

**Confusion Matrix (Tuned RF, Test Set):**
- True Positives: 126 | False Positives: 114
- True Negatives: 656 | False Negatives: 113

Tuning significantly improved recall (+0.40) and F1 (+0.31) by shifting the decision boundary to catch more true CVD events, at a trade-off in precision.

---

## Top Predictive Features

Ranked by average rank across all three importance methods (MDI, Permutation Importance, SHAP):

| Rank | Feature | Description |
|---|---|---|
| 1 | `age_s1` | Patient age at baseline |
| 2 | `SystBP` | Systolic blood pressure |
| 3 | `DiasBP` | Diastolic blood pressure |
| 4 | `WASO` | Wake after sleep onset (minutes) |
| 5 | `SRHype_1.0` | Self-reported hypertension |
| 6 | `gender_2` | Female gender indicator |
| 7 | `Trig` | Triglycerides |
| 8 | `CgPkYr` | Cigarette pack-years |
| 9 | `bmi_s1` | BMI at baseline |
| 10 | `ParRptDiab_1.0` | Participant-reported diabetes |

---

## Feature Importance Methods

Three complementary methods were used to ensure robust importance estimates:

- **MDI (Mean Decrease in Impurity)** — built into the Random Forest; fast but can favor high-cardinality continuous features
- **Permutation Importance** — measures AUC drop when each feature is shuffled on the test set (15 repeats); model-agnostic
- **SHAP (TreeExplainer)** — game-theoretic Shapley values; provides both global rankings and individual patient-level explanations

---

## Outputs Generated

The notebook saves the following plots:

| File | Description |
|---|---|
| `class_balance.png` | Class distribution bar chart and pie chart |
| `feature_distributions.png` | Key feature histograms by CVD outcome |
| `correlation_heatmap.png` | Correlation matrix — continuous features + target |
| `roc_confusion.png` | ROC curve (baseline vs. tuned) + confusion matrix |
| `cv_auc.png` | 5-fold cross-validation AUC bar chart |
| `final_metrics.png` | Final performance dashboard |
| `mdi_importance.png` | Top 20 features — MDI importance |
| `permutation_importance.png` | Top 20 features — permutation importance |
| `shap_beeswarm.png` | SHAP summary bee swarm plot |
| `shap_bar.png` | SHAP mean absolute value bar chart |
| `shap_dependence.png` | SHAP dependence plots for top 2 features |

---

## Loading the Saved Model

```python
import joblib
import json

model  = joblib.load('rf_cvd_model.sav')
scaler = joblib.load('scaler.sav')

with open('feature_names.json') as f:
    features = json.load(f)

# X_new must be a DataFrame with columns matching features
X_new_sc = scaler.transform(X_new[features])
prob     = model.predict_proba(X_new_sc)[:, 1]   # CVD event probability
pred     = model.predict(X_new_sc)                # Binary prediction
```

---

## Requirements

```
scikit-learn
shap
pandas
numpy
matplotlib
seaborn
joblib
```

Install with:
```bash
pip install seaborn shap joblib scikit-learn
```

---

## Limitations

- SHHS is observational — causal claims require cautious interpretation
- CVD event timing is not modeled; a survival approach would be more precise
- 77 features include 44 sparse alcohol frequency dummies which contribute minimal predictive value
- `class_weight='balanced'` improves recall but reduces precision due to the 3.2:1 class imbalance

---

*BSAN6070 — Introduction to Machine Learning | Loyola Marymount University | Spring 2026*
