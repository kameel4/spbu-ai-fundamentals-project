# Adult Census Income Prediction

## Goal
Predict whether a person earns **>50K per year** using classical ML methods.

## Dataset
Adult Census Income dataset (UCI Adult): mix of numerical + categorical features, missing/unknown values, sensitive attributes.

Dataset files are stored in:
- `data/adult.data`
- `data/adult.test`
- `data/adult.names`

## Pipeline
1. Data cleaning
2. EDA
3. Feature engineering
4. Preprocessing and train/test split
5. Baseline models
6. Model comparison
7. Hyperparameter tuning
8. Final model evaluation

Main notebook:
- `src/adult_census_income_project.ipynb`

Reusable project code:
- `src/adult_income_utils.py`
- `src/adult_visualization.py`
- `src/adult_modeling.py`

## Models
Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, plus `DummyClassifier` as a baseline.

## Metrics
Accuracy, precision, recall, F1-score, ROC-AUC, balanced accuracy.

## Final result
Based on the current run stored in `data/final_model_report.json`:

- Best model: **GradientBoosting** (selected by CV `roc_auc`)
- Test ROC-AUC: **0.9294**
- Test precision: **0.7926**
- Test recall: **0.6558**
- Test F1: **0.7177**
- Test accuracy: **0.8765**
- Test balanced accuracy: **0.8009**

Artifacts (saved to `data/`):
- `model_comparison_results.csv` - CV comparison table
- `best_models.json` - selected top models for tuning
- `best_model_gridsearch.joblib` - trained final model (pipeline)
- `final_model_report.json` - best params + CV/test metrics

## How to run
Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then run the single project notebook:

```text
src/adult_census_income_project.ipynb
```

By default, the notebook loads saved model artifacts. To recompute model comparison or GridSearchCV, set the corresponding flags in the first code cell.
