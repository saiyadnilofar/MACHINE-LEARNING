# Heart Disease Prediction from BRFSS2015 Health Indicators

A machine learning group project that builds a complete, leakage-free classification
pipeline to predict whether an individual has had heart disease or a heart attack, using
self-reported health indicators from the CDC's 2015 Behavioral Risk Factor Surveillance
System (BRFSS) survey.

## Overview

Heart disease remains one of the leading causes of death worldwide, and much of the risk is
tied to observable lifestyle and health indicators — blood pressure, cholesterol, BMI,
smoking status, physical activity, and more. This project explores how well a handful of
standard classification algorithms can predict heart disease risk from that kind of survey
data, using a properly separated train / validation / test workflow so that model selection
never touches the final evaluation data.

The aim here is a correct, explainable pipeline and an honest read of the results — not a
production-ready diagnostic tool.

## Dataset

- **Source:** [BRFSS 2015 Health Indicators Dataset](https://www.cdc.gov/brfss/) (heart disease subset)
- **File:** `data/raw/heart_disease_health_indicators_BRFSS2015.csv`
- **Size:** 253,680 survey responses · 22 columns (21 features + target)
- **Target variable:** `HeartDiseaseorAttack` (binary: 0 = No, 1 = Yes)
- **Key features:** `HighBP`, `HighChol`, `BMI`, `Smoker`, `Stroke`, `Diabetes`,
  `PhysActivity`, `GenHlth`, `MentHlth`, `Age`, `Income`, and more

## Project Structure

```
MACHINE-LEARNING/
├── data/
│   └── raw/
│       └── heart_disease_health_indicators_BRFSS2015.csv
├── data collection.ipynb                     # Data sourcing & initial load
├── Heart_disease_prediction_(1)_UPDATED.ipynb # Full pipeline: cleaning → EDA →
│                                               # feature engineering → modeling → evaluation
└── README.md
```

## Methodology

1. **Data Cleaning** — duplicate rows removed prior to splitting to prevent leakage
2. **Train / Validation / Test Split** — 50% / 25% / 25%, stratified on the target class
3. **Exploratory Data Analysis** — target class balance, BMI distribution, feature
   correlation heatmap
4. **Feature Engineering** — 13 domain-informed features derived from the raw indicators
   (e.g. `CardiometabolicRiskScore`, `ObesityFlag`, `MetabolicSyndromeProxy`, `AgeRiskTier`),
   applied identically across all three splits via a single shared function
5. **Preprocessing** — numerical features standardized, categorical features one-hot
   encoded; both fit only on the training set
6. **Modeling** — three classifiers trained and compared: K-Nearest Neighbors, Decision
   Tree, Random Forest
7. **Model Selection** — all three compared on the **validation set**; the best performer is
   confirmed **once** on the held-out **test set**

## Team & Roles

| Student | Role | Name |
|---|---|---|
| Student 1 | Data Collection | *[Name]* |
| Student 2 | Data Cleaning & EDA | NILOFARBANU SAIYAD|
| Student 3 | Feature Engineering | *[Name]* |
| Student 4 | Modeling | Aiswarya Suresh |
| Student 5 | Evaluation & Presentation | Saketh Rishi Nallagondla |

## Results

**Validation Set Comparison:**

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| KNN | 0.887 | 0.373 | 0.145 | 0.208 |
| **Decision Tree** ✅ | 0.835 | 0.247 | 0.293 | **0.268** |
| Random Forest | 0.893 | 0.429 | 0.123 | 0.192 |

**Decision Tree** was selected as the best model by validation F1-Score — despite having the
lowest accuracy, it struck the best balance of actually catching positive cases rather than
defaulting to "predict No."

**Final Test Set Result (Decision Tree):**

| Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|
| 0.833 | 0.235 | 0.273 | 0.252 |

Test performance closely tracks validation performance, indicating the model generalizes
consistently rather than overfitting to the validation split.

## Key Findings & Limitations

- **Accuracy is a misleading metric here.** Only ~9% of records are positive cases, so a
  model predicting "No" for everyone would still score ~90% accuracy while being clinically
  useless. F1-Score was used for model selection instead.
- **Several engineered features carry limited signal on this dataset.** `RespiratoryRiskScore`,
  `HypertensionSeverity`, and parts of `MentalHealthBurden` / `CholTreatmentGap` depend on
  columns (`Asthma`, `COPD`, `Depressed`, `BP_Medication`) not present in this BRFSS2015
  extract — a dataset limitation, not a pipeline bug.
- **No hyperparameter tuning or class-rebalancing was applied**, by design, to keep the
  pipeline simple and fully explainable end-to-end.

## Future Work

- Address class imbalance directly (class weighting or SMOTE)
- Hyperparameter tuning using the validation set
- Re-run feature engineering on a dataset revision that includes the missing health-condition
  columns
- Explore additional models (e.g. gradient boosting) for comparison

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/saiyadnilofar/MACHINE-LEARNING.git
cd MACHINE-LEARNING
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**3. Open and run the notebook**
Open `Heart_disease_prediction_(1)_UPDATED.ipynb` in Jupyter or VS Code and run all cells
top to bottom. The dataset is already included at `data/raw/`, so no additional download is
required.

---

*This is an academic group project completed as part of a Machine Learning course
assignment.*
