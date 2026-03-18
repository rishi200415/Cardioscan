# Healthcare ML — Patient & Heart Disease Prediction

A collection of three machine learning notebooks that predict healthcare outcomes using real clinical data. Built end-to-end: from pulling data out of a database, cleaning it, training models, and visualizing results.

---

## Notebooks

### 1. Heart Disease EDA (`heart-disease-predictions.ipynb`)

A deep exploratory analysis of the UCI Heart Disease dataset (296 patients). Before building any model, this notebook answers the question: *which features actually matter for predicting heart disease?*

- Cleans the data by removing known faulty values
- Renames cryptic column names (e.g. `cp` → `chest_pain_type`) for readability
- Visualizes distributions, pair-plots, and density curves for all features
- Applies three different correlation methods depending on feature type:
  - **Pearson** for numerical vs. numerical
  - **Point-Biserial** for numerical vs. binary
  - **Cramér's V** for categorical vs. categorical

This is the analytical foundation the other notebooks build on.

---

### 2. Hospital Department Classifier (`healthcare_ml.ipynb`)

Connects to a live PostgreSQL database and trains a model to predict which hospital department a patient visit belongs to — across 13 departments like Cardiology, Neurology, and Orthopedics.

- Joins 4 database tables (visits, patients, doctors, procedures) into one flat dataset
- Engineers new features: age groups, procedure cost tiers, visit month and day of week
- Trains and compares three models: Random Forest, Gradient Boosting, Logistic Regression
- Selects the best model using 5-fold cross-validation
- Outputs charts: EDA overview, model comparison, confusion matrix, feature importances
- Ends with a live prediction on a hypothetical new patient

---

### 3. Heart Disease Classifier with Smart Imputation (`combined_ml.ipynb`)

The most advanced notebook. Merges the UCI heart disease data with a custom patients table from PostgreSQL — without a shared ID, using age and gender as the join keys. Then builds a full ML pipeline that handles missing data the way real clinical projects do.

- ~70% of `city` values are null from the join mismatch — all imputed
- Simulates realistic missing data (8% null rate) in clinical columns like cholesterol and blood pressure
- **Smart imputation strategy:**
  - Numerical columns filled with the group mean by age decade
  - Categorical columns filled with the group mode by gender
- Adds `_was_imputed` flag columns so the model can learn which values were observed vs. filled
- Measures whether the imputation flags actually improved accuracy
- Trains and compares all three models using both accuracy and AUC-ROC
- Plots ROC curves, confusion matrices, and feature importances broken down by feature type

---

## Tech Stack

| Tool | Used for |
|------|----------|
| Python | Everything |
| pandas | Data loading, cleaning, feature engineering |
| scikit-learn | Model training, cross-validation, pipelines, metrics |
| PostgreSQL + SQLAlchemy | Live database connection and querying |
| matplotlib + seaborn | All charts and visualizations |
| scipy | Statistical correlation (Cramér's V, Point-Biserial) |
| Jupyter Notebooks | Development environment |

---

## Skills Demonstrated

**Data Engineering**
- Writing multi-table SQL JOIN queries
- Connecting to and querying a live PostgreSQL database
- Merging datasets without a shared key using domain logic

**Data Science**
- Exploratory data analysis with appropriate statistical methods per feature type
- Feature engineering (binning, encoding, date extraction)
- Group-aware imputation strategies (mean by age group, mode by gender)
- Using imputation flags as model features

**Machine Learning**
- Training and comparing multiple classifiers (Random Forest, Gradient Boosting, Logistic Regression)
- Using cross-validation and AUC-ROC for model selection on small datasets
- Building scikit-learn Pipelines for clean, reproducible workflows
- Evaluating models with confusion matrices, classification reports, and ROC curves

**Communication**
- Explaining every modeling decision in plain English within the notebooks
- Visualizing results clearly for a non-technical audience
