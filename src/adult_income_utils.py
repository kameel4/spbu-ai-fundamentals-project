from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42

COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


def get_data_dir():
    return Path(__file__).resolve().parents[1] / "data"


def get_repo_root():
    return Path(__file__).resolve().parents[1]


def get_artifacts_dir():
    return get_repo_root() / "data"


def load_raw_adult_data(data_dir=None):
    data_dir = Path(data_dir) if data_dir is not None else get_data_dir()

    train_raw = pd.read_csv(data_dir / "adult.data", header=None, names=COLUMN_NAMES)
    test_raw = pd.read_csv(data_dir / "adult.test", header=None, names=COLUMN_NAMES, skiprows=1)

    return train_raw, test_raw


def clean_adult_data(df):
    df = df.copy()

    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].str.strip()

    df = df.replace("?", np.nan)
    df["income"] = df["income"].str.replace(".", "", regex=False)
    df = df.drop_duplicates().reset_index(drop=True)

    return df


def load_clean_adult_data(data_dir=None):
    train_raw, test_raw = load_raw_adult_data(data_dir=data_dir)
    df_raw = pd.concat([train_raw, test_raw], ignore_index=True)

    return clean_adult_data(df_raw)


def get_cleaning_summary(data_dir=None):
    train_raw, test_raw = load_raw_adult_data(data_dir=data_dir)
    df_raw = pd.concat([train_raw, test_raw], ignore_index=True)

    stripped = df_raw.copy()
    for column in stripped.select_dtypes(include="object").columns:
        stripped[column] = stripped[column].str.strip()
    stripped = stripped.replace("?", np.nan)
    stripped["income"] = stripped["income"].str.replace(".", "", regex=False)

    missing_summary = pd.DataFrame({
        "missing_count": stripped.isna().sum(),
        "missing_percent": stripped.isna().mean() * 100,
    })
    missing_summary = missing_summary[missing_summary["missing_count"] > 0]
    missing_summary = missing_summary.sort_values("missing_count", ascending=False)

    duplicate_count = stripped.duplicated().sum()

    return {
        "train_shape": train_raw.shape,
        "test_shape": test_raw.shape,
        "raw_shape": df_raw.shape,
        "clean_shape": clean_adult_data(df_raw).shape,
        "missing_summary": missing_summary,
        "rows_with_missing": stripped.isna().any(axis=1).sum(),
        "duplicate_count": int(duplicate_count),
    }


def add_features(df):
    df = df.copy()

    married_statuses = ["Married-civ-spouse", "Married-AF-spouse"]
    df["is_married"] = df["marital_status"].isin(married_statuses).astype(int)

    df["native_country_group"] = np.where(
        df["native_country"].isna(),
        "Unknown",
        np.where(df["native_country"] == "United-States", "United-States", "Other"),
    )

    workclass_group_map = {
        "Private": "Private",
        "Federal-gov": "Government",
        "Local-gov": "Government",
        "State-gov": "Government",
        "Self-emp-not-inc": "Self-employed",
        "Self-emp-inc": "Self-employed",
        "Without-pay": "Other",
        "Never-worked": "Other",
    }
    df["workclass_group"] = df["workclass"].map(workclass_group_map).fillna("Unknown")

    df["hours_per_week_bin"] = pd.cut(
        df["hours_per_week"],
        bins=[0, 34, 40, 50, np.inf],
        labels=["part_time", "full_time_40", "over_40", "extreme"],
        include_lowest=True,
    ).astype("object")

    df["capital_net"] = df["capital_gain"] - df["capital_loss"]
    df["has_capital_gain"] = (df["capital_gain"] > 0).astype(int)
    df["has_capital_loss"] = (df["capital_loss"] > 0).astype(int)

    return df


def make_X_y(df):
    X = df.drop(columns="income")
    y = df["income"].map({"<=50K": 0, ">50K": 1})

    return X, y


def prepare_model_data(test_size=0.2, random_state=RANDOM_STATE, data_dir=None):
    df_clean = load_clean_adult_data(data_dir=data_dir)
    df_features = add_features(df_clean)
    X, y = make_X_y(df_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    numerical_features, categorical_features = get_feature_lists(X_train)

    return {
        "df_clean": df_clean,
        "df_features": df_features,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
    }


def get_feature_lists(X):
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include="object").columns.tolist()

    return numerical_features, categorical_features


def make_preprocessor(numerical_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    return preprocessor
