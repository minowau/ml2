🧠 ML Project – Founder Retention & Personality Classification

Machine Learning Course Project — November 27, 2025

This repository contains the complete implementation, experiments, and artifacts for two supervised learning challenges completed as part of the Machine Learning course.
Both tasks follow a rigorous ML workflow including EDA, feature engineering, preprocessing pipelines, model training, hyperparameter tuning, and final inference pipeline construction.

📌 Project Tasks
1. Founder Retention Prediction (Binary Classification)

Predict whether a startup founder Stayed or Left based on demographic, operational, and organisational features.

2. Personality Cluster Classification (Multi-Class Classification)

Predict a participant’s Personality Cluster (A–E) from behavioural and lifestyle attributes using extensive engineered features.

Both tasks use separate training and test datasets and produce separate submission files.

👨‍💻 Team Members

Jaswant Jupalli Prabhas – IMT2023105

Likith Kasam – IMT2023573

Surya Sumeeth Singh – IMT2023590

Repository Link: https://github.com/minowau/ml2

📁 Dataset Overview
Founder Retention Dataset

Key columns:

Founder demographics and role

Startup operations: revenue, satisfaction, WLB rating

Staffing and funding attributes

Social/support attributes

Target: Stayed / Left

Contains missing values in several numeric fields.

Personality Cluster Dataset

Key columns:

Demographics

Behavioural indices (focus, consistency, creativity, altruism, etc.)

Target: Cluster A–E

Data is mostly complete but suffers from class imbalance.

🔍 Exploratory Data Analysis (EDA)
✔ Missing Value Analysis

Retention dataset contains non-uniform missingness.

Personality dataset mostly clean with noticeable class imbalance (Cluster E most common).

✔ Distribution & Outliers

Right-skewed numeric variables → log transforms used.

Rare categorical classes grouped into “Other”.

✔ Correlation Analysis

Guided feature engineering and PCA selection.

Identified key behavioural correlations in personality dataset.

🛠 Feature Engineering
Retention Task

Binning continuous features

Interaction terms (e.g., WLB × Satisfaction)

Boolean feature composites

Aggregated categorical groups

Personality Task (extensive engineered features)

Ratios & composite indices (stability, lifestyle balance, expressive altruism)

Interaction terms (focus × consistency, activity × creativity)

Polynomial features (squared terms)

Geometric-mean–based multipliers

High-dimensional engineered space → PCA applied (95% variance retained)

🔄 Preprocessing Pipeline (sklearn)
Common Blocks

Imputation: median (numeric), “Missing” token (categorical)

Encoding: One-Hot, Target Encoding

Scaling: RobustScaler, log1p for skewed variables

Dimensionality Reduction: PCA for personality dataset

Sampling: SMOTE + Class Weights for imbalanced multi-class dataset

Reproducibility

All pipelines, encoders, and models are saved as .joblib artifacts.

🤖 Models Trained
Binary Retention Models

SVM (RBF & Linear)

MLPClassifier + threshold tuning

Logistic Regression baseline

All models wrapped in unified pipelines

Multi-Class Personality Models

SVM with PCA

MLP with RandomizedSearch

RandomForest

XGBoost

Ensembles (Averaged & Stacked) – best macro F1 ≈ 0.640

🎯 Hyperparameter Tuning

Stratified K-Fold CV (3 folds)

GridSearchCV for SVM & small models

RandomizedSearchCV for MLP & tree models

Metrics:

Retention → accuracy + tuned probability thresholds

Personality → macro-averaged F1

📊 Performance Summary
Personality Prediction
Model	Macro F1
SVM	~0.576
MLP (tuned)	~0.616
RandomForest	~0.618
Ensemble (SVM + MLP)	~0.640 (Best)
Retention Prediction

Threshold-tuned MLP and SVM performed best on validation.

Logistic regression used as strong, stable baseline.

📦 Final Deliverables

Serialized .joblib model artifacts

Serialized encoders and PCA objects

Final submission CSVs for both tasks

Personality ensemble predictions

Retention predictions from tuned pipelines

📝 Key Observations

Heavy feature engineering significantly improves personality predictions.

PCA reduces redundancy in engineered space without losing signal.

Ensembling increases model robustness.

Threshold tuning is essential for operational accuracy in binary classification.

Reproducibility ensured via full pipeline serialization.

✅ Conclusion

This repository delivers two complete, end-to-end machine learning pipelines:

Founder Retention Prediction:
A robust binary classification system with tuned thresholds and reproducible artifacts.

Personality Cluster Classification:
A feature-deep, PCA-enhanced, ensemble-powered multi-class ML workflow achieving strong macro F1.

Future Work

SHAP-based feature selection

More advanced ensembling (meta-learning / stacking)

Survival modelling for retention (predict when, not just whether)

Probability calibration for improved decision-making in personality prediction
