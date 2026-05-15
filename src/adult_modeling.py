import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from adult_income_utils import RANDOM_STATE, get_artifacts_dir, make_preprocessor


MODEL_COMPARISON_FILE = "model_comparison_results.csv"
BEST_MODELS_FILE = "best_models.json"
FINAL_MODEL_FILE = "best_model_gridsearch.joblib"
FINAL_REPORT_FILE = "final_model_report.json"


def make_cv(random_state=RANDOM_STATE, n_splits=5):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def get_scoring_metrics():
    return {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "balanced_accuracy": "balanced_accuracy",
        "roc_auc": "roc_auc",
    }


def make_models(random_state=RANDOM_STATE):
    return {
        "DummyClassifier": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "KNN": KNeighborsClassifier(),
        "DecisionTree": DecisionTreeClassifier(
            random_state=random_state,
            class_weight="balanced",
        ),
        "RandomForest": RandomForestClassifier(
            random_state=random_state,
            class_weight="balanced",
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=random_state,
        ),
    }


def make_pipeline(model, numerical_features, categorical_features):
    return Pipeline(steps=[
        ("preprocessor", make_preprocessor(numerical_features, categorical_features)),
        ("model", model),
    ])


def fit_model_pipelines(X_train, y_train, numerical_features, categorical_features, models=None):
    models = models or make_models()
    fitted_pipelines = {}

    for model_name, model in models.items():
        pipeline = make_pipeline(model, numerical_features, categorical_features)
        pipeline.fit(X_train, y_train)
        fitted_pipelines[model_name] = pipeline

    return fitted_pipelines


def compare_models_cv(
    X_train,
    y_train,
    numerical_features,
    categorical_features,
    models=None,
    cv=None,
    scoring=None,
    n_jobs=1,
):
    models = models or make_models()
    cv = cv or make_cv()
    scoring = scoring or get_scoring_metrics()

    rows = []
    for model_name, model in models.items():
        pipeline = make_pipeline(model, numerical_features, categorical_features)
        cv_result = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=n_jobs,
        )

        row = {"Model": model_name}
        for metric in scoring:
            scores = cv_result[f"test_{metric}"]
            row[f"{metric}_mean"] = scores.mean()
            row[f"{metric}_std"] = scores.std()
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_summary = make_results_summary(results_df)
    rankings = make_rankings(results_summary)
    top_model_names = choose_top_models(rankings)

    return {
        "results_df": results_df,
        "results_summary": results_summary,
        "rankings": rankings,
        "top_model_names": top_model_names,
    }


def make_results_summary(results_df):
    columns = {
        "accuracy_mean": "Accuracy",
        "precision_mean": "Precision",
        "recall_mean": "Recall",
        "f1_mean": "F1",
        "balanced_accuracy_mean": "Balanced Accuracy",
        "roc_auc_mean": "ROC-AUC",
    }
    available_columns = ["Model"] + [source for source in columns if source in results_df.columns]
    summary = results_df[available_columns].copy()
    summary = summary.rename(columns=columns)
    return summary.set_index("Model")


def make_rankings(results_summary):
    rankings = pd.DataFrame(index=results_summary.index)

    metric_columns = [column for column in results_summary.columns if column != "mean_rank"]
    for metric in metric_columns:
        rankings[f"{metric}_rank"] = results_summary[metric].rank(ascending=False).astype(int)

    if "mean_rank" in results_summary.columns:
        rankings["mean_rank"] = results_summary["mean_rank"]
    else:
        rankings["mean_rank"] = rankings.mean(axis=1)

    return rankings.sort_values("mean_rank")


def choose_top_models(rankings, n=2, exclude=("DummyClassifier",)):
    candidates = rankings.drop(index=list(exclude), errors="ignore")
    return candidates.sort_values("mean_rank").head(n).index.tolist()


def get_param_grids():
    return {
        "LogisticRegression": {
            "model__C": [0.1, 1.0, 3.0],
            "model__solver": ["lbfgs"],
        },
        "KNN": {
            "model__n_neighbors": [5, 15, 25],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
        "DecisionTree": {
            "model__max_depth": [None, 10, 20],
            "model__min_samples_leaf": [1, 2, 5],
            "model__criterion": ["gini", "entropy"],
        },
        "RandomForest": {
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 15, 30],
            "model__min_samples_split": [2, 10],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", "log2"],
        },
        "GradientBoosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
            "model__subsample": [0.8, 1.0],
        },
    }


def tune_models(
    top_model_names,
    X_train,
    y_train,
    numerical_features,
    categorical_features,
    models=None,
    param_grids=None,
    cv=None,
    scoring=None,
    refit_metric="roc_auc",
    n_jobs=1,
    verbose=1,
):
    models = models or make_models()
    param_grids = param_grids or get_param_grids()
    cv = cv or make_cv()
    scoring = scoring or get_scoring_metrics()

    gridsearch_results = {}
    rows = []

    for model_name in top_model_names:
        if model_name not in models or model_name not in param_grids:
            continue

        pipeline = make_pipeline(models[model_name], numerical_features, categorical_features)
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[model_name],
            scoring=scoring,
            refit=refit_metric,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=False,
        )
        grid.fit(X_train, y_train)
        gridsearch_results[model_name] = grid
        rows.append({
            "Model": model_name,
            f"best_cv_{refit_metric}": grid.best_score_,
            "n_candidates": len(grid.cv_results_["params"]),
        })

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(f"best_cv_{refit_metric}", ascending=False)

    return gridsearch_results, summary


def select_best_grid(gridsearch_results, gridsearch_summary_df, refit_metric="roc_auc"):
    if gridsearch_summary_df.empty:
        raise RuntimeError("GridSearchCV results are empty.")

    best_model_name = gridsearch_summary_df.iloc[0]["Model"]
    best_grid = gridsearch_results[best_model_name]
    return best_model_name, best_grid, best_grid.best_estimator_


def evaluate_estimator(estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)
    if hasattr(estimator, "predict_proba"):
        y_score = estimator.predict_proba(X_test)[:, 1]
    else:
        y_score = estimator.decision_function(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
    }

    return {
        "y_pred": y_pred,
        "y_score": y_score,
        "metrics": metrics,
        "classification_report": classification_report(y_test, y_pred),
    }


def get_model_explanation(estimator, top_n=20):
    feature_names = estimator.named_steps["preprocessor"].get_feature_names_out()
    model = estimator.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        explanation = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        return explanation.head(top_n), "importance"

    if hasattr(model, "coef_"):
        coef = model.coef_[0]
        explanation = pd.DataFrame({
            "feature": feature_names,
            "coef": coef,
            "abs_coef": np.abs(coef),
        }).sort_values("abs_coef", ascending=False)
        return explanation.head(top_n), "coef"

    return pd.DataFrame(), None


def make_error_frame(X_test, y_test, y_pred, y_score=None):
    error_df = X_test.copy()
    error_df["true"] = y_test.values
    error_df["pred"] = y_pred
    if y_score is not None:
        error_df["score_>50K"] = y_score

    error_df["error_type"] = "correct"
    error_df.loc[(error_df["true"] == 0) & (error_df["pred"] == 1), "error_type"] = "false_positive"
    error_df.loc[(error_df["true"] == 1) & (error_df["pred"] == 0), "error_type"] = "false_negative"

    return error_df


def error_summary_by_column(error_df, column):
    return (
        pd.crosstab(error_df[column], error_df["error_type"], normalize="index")
        .round(3)
        .sort_values("false_negative", ascending=False)
    )


def save_model_comparison(results_summary, rankings, top_model_names, artifacts_dir=None):
    artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else get_artifacts_dir()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    export = results_summary.copy()
    export["mean_rank"] = rankings["mean_rank"]
    export.to_csv(artifacts_dir / MODEL_COMPARISON_FILE)

    best_models_info = {
        "top_models": top_model_names,
        "top_1": top_model_names[0],
        "top_2": top_model_names[1],
    }
    with open(artifacts_dir / BEST_MODELS_FILE, "w", encoding="utf-8") as f:
        json.dump(best_models_info, f, indent=2)


def load_model_comparison(artifacts_dir=None):
    artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else get_artifacts_dir()
    return pd.read_csv(artifacts_dir / MODEL_COMPARISON_FILE, index_col=0)


def load_top_models(artifacts_dir=None, default=None):
    artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else get_artifacts_dir()
    path = artifacts_dir / BEST_MODELS_FILE
    if not path.exists():
        return default or ["GradientBoosting", "LogisticRegression"]

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("top_models", default or ["GradientBoosting", "LogisticRegression"])


def save_final_artifacts(
    best_estimator,
    best_model_name,
    best_grid,
    test_metrics,
    top_model_names,
    artifacts_dir=None,
    refit_metric="roc_auc",
):
    artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else get_artifacts_dir()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / FINAL_MODEL_FILE
    report_path = artifacts_dir / FINAL_REPORT_FILE

    joblib.dump(best_estimator, model_path)

    final_report = {
        "best_model": best_model_name,
        "best_params": best_grid.best_params_,
        "best_cv_metric": refit_metric,
        "best_cv_score": float(best_grid.best_score_),
        "test_metrics": test_metrics,
        "top_models_input": top_model_names,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    return model_path, report_path


def load_final_model(artifacts_dir=None):
    artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else get_artifacts_dir()
    return joblib.load(artifacts_dir / FINAL_MODEL_FILE)


def load_final_report(artifacts_dir=None):
    artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else get_artifacts_dir()
    with open(artifacts_dir / FINAL_REPORT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
