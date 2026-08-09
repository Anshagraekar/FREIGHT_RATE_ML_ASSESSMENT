# Freight Rate Prediction — Machine Learning Assessment

A machine learning solution for predicting freight load rates using shipment, geographic, equipment, weight, market, and temporal information.

The solution was developed as part of a Machine Learning Engineer assessment and focuses on realistic temporal validation, data-quality handling, model experimentation, error analysis, and reproducible prediction generation.

---

## Overview

The objective is to predict the `posted_rate` of freight loads from historical shipment information.

The development dataset contains:

- 48,000 labeled loads
- 14 columns
- January–October 2025 observations
- `posted_rate` as the prediction target

The validation dataset contains:

- 12,000 unlabeled loads
- 13 predictor columns
- November 2025 observations

The final model is trained on the complete development dataset and generates predictions for all 12,000 validation loads.

---

## Final Model

The selected model is a **CatBoost regression model trained using a log-transformed target**.

### Why CatBoost?

CatBoost was selected because the problem contains:

- Categorical variables such as pickup, delivery, and equipment
- Continuous numerical variables
- Nonlinear relationships
- Potential interactions between shipment characteristics
- A moderate-sized tabular dataset

CatBoost provides strong performance on this type of mixed tabular data while requiring relatively little preprocessing.

### Why a log-transformed target?

The target distribution is strongly right-skewed:

| Statistic | `posted_rate` |
|---|---:|
| Mean | $2,373.98 |
| Median | $2,030.76 |
| Std. Dev. | $1,486.49 |
| 95th percentile | $4,953.77 |
| 99th percentile | $5,972.83 |
| Maximum | $25,533.00 |

The long upper tail contains unusually high-rate loads.

Training with:

```python
log1p(posted_rate)
```
and converting predictions back to the original dollar scale improved the typical absolute prediction error.

Model Performance

A temporal expanding-window validation strategy was used instead of a random train/test split.

The final log-target CatBoost model achieved:

Metric	Average
MAE	$107.15
RMSE	$626.96
R²	0.8271
Temporal validation
Fold	Training Rows	Validation Rows	MAE	RMSE	R²
July	28,806	4,912	$111.25	$627.26	0.8262
August	33,718	4,759	$99.21	$615.09	0.8261
September	38,477	4,670	$110.62	$619.04	0.8349
October	43,147	4,853	$107.51	$646.44	0.8211
Average	—	—	$107.15	$626.96	0.8271

The temporal validation approach was chosen to better represent the real-world scenario of predicting future freight rates from historical observations.

EDA & Key Findings
Target distribution

The target is strongly right-skewed, with a relatively small number of very high-rate loads.

This motivated the log-target experiment.

Distance

Distance was the strongest individual numerical predictor.

Correlation with posted_rate:

distance          0.9085

This indicates that shipment distance is a major driver of freight rate.

Data quality

The following issues were identified:

Missing weight
Missing market_index
Negative weight values
No duplicate rows
No duplicate load_id
No zero or negative distances
No zero or negative target rates

Negative weights were treated as invalid and converted to missing values.

Geographic analysis

A Haversine distance feature was investigated.

The supplied distance and calculated Haversine distance were highly correlated:

Correlation = 0.9995

Additional geographic features were tested during experimentation but did not consistently improve temporal validation performance.

Feature Engineering

The following features were investigated:

Haversine distance
Latitude difference
Longitude difference
Route
Weight per mile
Weight missing indicator
Market index missing indicator
Year
Month
Day of month
Day of week
Day of year
Week of year
Days since dataset start
Cyclic month features
Cyclic weekday features

Feature experiments were evaluated using the same temporal validation framework.

The experiments showed that adding a large number of engineered features did not automatically improve performance.

This led to a preference for the simpler feature representation used by the final model.

Model Experiments

Several CatBoost configurations were evaluated.

The best normal-target CatBoost experiment achieved approximately:

MAE  : $126.91
RMSE : $628.49
R²   : 0.8262

The final log-target approach improved MAE to approximately:

MAE  : $107.15
RMSE : $626.96
R²   : 0.8271

This improvement in MAE was the main reason for selecting the log-target model.

Error Analysis

Model errors were analyzed by equipment type and shipment distance.

Error by distance
Distance	MAE
<500 miles	$45.97
500–1,000 miles	$71.11
1,000–1,500 miles	$120.85
1,500–2,000 miles	$162.18
2,000–3,000 miles	$212.95
>3,000 miles	$148.30

Absolute error generally increases as shipment distance increases.

Error by equipment
Equipment	MAE
Dry Van	$100.43
Flatbed	$106.04
Reefer	$124.99

The largest individual errors were generally associated with unusually high-rate loads where the observed rate was substantially above the model prediction.

This indicates that the sparse high-rate tail remains the primary limitation of the current feature set.

December 2025 Scenario

The assessment includes a fixed December scenario where the shipment characteristics remain constant and only the date changes.

Fixed shipment
Pickup:       Lexington
Delivery:     Fort Wayne
Distance:     360 miles
Equipment:    Dry Van
Weight:       32,000 lb
Period:       December 1–31, 2025

The final model generated predictions for all 31 days.

December prediction statistics
Statistic	Predicted Rate
Mean	$832.85
Median	$831.27
Minimum	$828.96
Maximum	$841.23
Standard deviation	$3.25

The official assessment scorer successfully validated all 31 December predictions and generated the required chart.

The chart is available at:

scorer_results/candidate_december.png
Final Prediction File

The final submission file is:

validation_predictions.csv

It contains exactly two columns:

load_id,predicted_rate

The file contains:

12,000 predictions
0 duplicate IDs
0 missing IDs
0 missing predictions
0 non-positive predictions

The official scorer successfully validated all 12,000 predictions.

Repository Structure
freight_rate_ml_assessment/
│
├── src/
│   ├── eda.py
│   ├── features.py
│   ├── model_experiments.py
│   ├── error_analysis.py
│   ├── train.py
│   ├── predict.py
│   ├── predict_december.py
│   └── inspect_december.py
│
├── outputs/
│   └── models/
│       └── catboost_final_log_target.cbm
│
├── scorer_results/
│   └── candidate_december.png
│
├── validation_predictions.csv
├── requirements.txt
├── README.md
└── Freight_Rate_ML_Assessment_Report.pdf
Installation

Python 3.11 was used during development.

Create and activate a virtual environment:

python -m venv .venv
Windows
.venv\Scripts\activate
macOS/Linux
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt
Running the Pipeline

The assessment-provided datasets should be placed in the expected data/ directory locally.

1. Exploratory Data Analysis
python src/eda.py

This performs dataset inspection, missing-value analysis, duplicate checks, target analysis, correlations, route analysis, and other exploratory checks.

2. Feature Analysis
python src/features.py

This performs the feature engineering and geographic feature analysis used during experimentation.

3. Model Experiments
python src/model_experiments.py

This evaluates the CatBoost configurations and temporal validation folds used for model selection.

4. Error Analysis
python src/error_analysis.py

This analyzes prediction errors by distance and equipment and identifies high-error observations.

5. Train the Final Model
python src/train.py

The final model is saved to:

outputs/models/catboost_final_log_target.cbm
6. Generate Validation Predictions
python src/predict.py

The final validation predictions are generated for all 12,000 validation loads.

The submission file must contain:

load_id,predicted_rate
7. Generate December Predictions
python src/predict_december.py

This generates predictions for the fixed December scenario.

8. Inspect December Predictions
python src/inspect_december.py

This checks:

Number of rows
Required columns
Missing values
Duplicate rows
Date range
Prediction completeness
Validation

The solution uses an expanding-window temporal validation strategy.

For each validation month:

Train on historical months
        ↓
Predict the next month
        ↓
Calculate MAE / RMSE / R²
        ↓
Expand training window
        ↓
Repeat

This avoids using future observations to predict earlier periods and better reflects the intended forecasting setting.

Reproducibility

The repository separates the major stages of the workflow:

EDA
 ↓
Data Quality Checks
 ↓
Feature Engineering
 ↓
Temporal Validation
 ↓
Model Experiments
 ↓
Error Analysis
 ↓
Final Model Training
 ↓
Validation Prediction
 ↓
December Scenario Prediction

The final model artifact and prediction file are included as submission outputs.

Limitations & Future Improvements

The main limitation is the sparse high-rate tail of the target distribution.

The model performs substantially better on typical freight rates than on rare observations with unusually high posted rates.

Potential future improvements include:

Richer route-level historical features
Historical route pricing statistics
External freight-market indicators
More granular temporal market features
Specialized modeling of rare high-rate regimes
Quantile or uncertainty-aware prediction
Additional production monitoring for model drift

These improvements would require additional historical or external market data beyond the assessment dataset.

Assessment Outcome

The complete prediction pipeline was successfully validated using the provided scoring utility:

Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results\candidate_december.png
Final validation metrics are calculated by Spotter after submission.

The final deliverables therefore include the solution code, dependencies, run instructions, validated validation_predictions.csv, final model artifact, December prediction chart, and assessment report.


### A couple of important notes before you paste it

**1. Don't include the `data/` folder in GitHub**, as we decided. The README already explains that the assessment-provided data should be placed locally.

**2. Keep the final model only:**

```text
outputs/models/catboost_final_log_target.cbm
