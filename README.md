# Heart Disease Prediction from BRFSS2015 Health Indicators

A machine learning group project that builds a complete, leakage-free classification
pipeline to predict whether an individual has had heart disease or a heart attack, using
self-reported health indicators from the CDC's 2015 Behavioral Risk Factor Surveillance
System (BRFSS) survey.

## Overview

Heart disease remains one of the leading causes of death worldwide, and much of the risk is
tied to observable lifestyle and health indicators such as blood pressure, cholesterol, BMI,
smoking status, and physical activity. This project explores how well a handful of standard
classification algorithms can predict heart disease risk from that kind of survey data,
using a properly separated train, validation, and test workflow so that model selection
never touches the final evaluation data.

The aim is a correct, explainable pipeline and an honest read of the results, not a
production-ready diagnostic tool.

## Dataset

- **Source:** [BRFSS 2015 Health Indicators Dataset](https://www.cdc.gov/brfss/) (heart disease subset)
- **File:** `data/raw/heart_disease_health_indicators_BRFSS2015.csv`
- **Size:** 253,680 survey responses, 22 columns (21 features plus target)
- **Target variable:** `HeartDiseaseorAttack` (binary: 0 = No, 1 = Yes)
- **Key features:** `HighBP`, `HighChol`, `BMI`, `Smoker`, `Stroke`, `Diabetes`,
  `PhysActivity`, `GenHlth`, `MentHlth`, `Age`, `Income`, and more

## Project Structure

```
MACHINE-LEARNING/
|-- data/
|   `-- raw/
|       `-- heart_disease_health_indicators_BRFSS2015.csv
|-- data collection.ipynb                        # Data sourcing and initial load
|-- Heart_disease_prediction_(1)_UPDATED_1.ipynb # Full pipeline: cleaning, EDA,
|                                                 # feature engineering, modeling, evaluation
`-- README.md
```

## Methodology

1. **Data Cleaning:** duplicate rows removed prior to splitting to prevent leakage
2. **Train / Validation / Test Split:** 50% / 25% / 25%, stratified on the target class
3. **Exploratory Data Analysis:** target class balance, BMI distribution, feature
   correlation heatmap
4. **Feature Engineering:** five features derived from the raw indicators
   (`ObesityFlag`, `CardiometabolicRiskScore`, `AgeRiskTier`, `HealthcareBarrierScore`,
   `SocialIsolationProxy`), applied identically across all three splits via a single shared
   function, with an assertion confirming none is constant
5. **Preprocessing:** continuous and ordinal columns standardized (binary flags left as-is
   so they don't distort distance-based models), fit on the training set only
6. **Modeling:** three classifiers trained and compared: K-Nearest Neighbors, Decision Tree,
   and Random Forest, with class weighting on the tree models to address imbalance
7. **Model Selection:** all three compared on the validation set using F1-Score, ROC-AUC,
   and PR-AUC; the best performer is confirmed once on the held-out test set

## Team & Roles

| Student | Role | Name |
|---|---|---|
| Student 1 | Data Collection | Marcus Nathanael Priyan |
| Student 2 | Data Cleaning & EDA | Nilofarbanu Saiyad |
| Student 3 | Feature Engineering | Jaweria Aijaz |
| Student 4 | Modeling | Aiswarya Suresh |
| Student 5 | Evaluation & Presentation | Saketh Rishi Nallagondla |

## Results

**Validation Set Comparison:**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| KNN | 0.885 | 0.361 | 0.144 | 0.206 | 0.707 | 0.216 |
| **Decision Tree** | 0.723 | 0.237 | 0.757 | **0.361** | 0.793 | 0.319 |
| Random Forest | 0.893 | 0.424 | 0.101 | 0.164 | 0.798 | 0.296 |

Decision Tree was selected as the best model by validation F1-Score. With class weighting, it
catches far more real cases than the other two (recall 0.757), at the cost of lower precision
and overall accuracy. Accuracy alone would have favored Random Forest, which mostly predicts
"No" and misses most positive cases.

**Final Test Set Result (Decision Tree):**

| Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| 0.725 | 0.239 | 0.761 | 0.363 | 0.799 | 0.320 |

Test performance closely tracks validation performance, indicating the model generalizes
consistently rather than overfitting to the validation split.

## Key Findings & Limitations

- **Accuracy is a misleading metric here.** Only about 9% of records are positive cases, so a
  model predicting "No" for everyone would still score around 90% accuracy while being
  clinically useless. F1-Score, ROC-AUC, and PR-AUC were used instead.
- **Class weighting shifts the precision-recall trade-off.** It raises Decision Tree's recall
  substantially but lowers precision, meaning more false alarms. This is a genuine trade-off
  rather than a model that is simply weak, but none of the three models are tuned enough for
  clinical use as they stand.
- **Some training rows share identical features but different outcomes.** This is not a data
  error; because BRFSS has no person identifier, different respondents can give identical
  answers. It places a real ceiling on how well any model can separate the classes.
- **The deduplication step is worth revisiting.** Nearly 24,000 rows were dropped as
  duplicates, but some were likely distinct respondents who answered identically, so this
  removes real data and should be a team decision.

## Future Work

- Tune the decision threshold deliberately based on which error type matters more, missed
  cases or false alarms, rather than leaving it at the default 0.5
- Hyperparameter tuning using the validation set
- Explore additional models such as gradient boosting for comparison

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/saiyadnilofar/MACHINE-LEARNING.git
cd MACHINE-LEARNING
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tabulate
```

**3. Open and run the notebook**
Open `Heart_disease_prediction_(1)_UPDATED_1.ipynb` in Jupyter or VS Code and run all cells
top to bottom. The dataset is already included at `data/raw/`, so no additional download is
required.

---

*This is an academic group project completed as part of a Machine Learning course
assignment.*
