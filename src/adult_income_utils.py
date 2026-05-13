from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
