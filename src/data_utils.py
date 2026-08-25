from __future__ import annotations

import random

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except (ImportError, OSError):
        pass


def winsorize(s: pd.Series, lower: float, upper: float) -> pd.Series:
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def load_panel(config):
    search_df = pd.read_excel(config.search_path, sheet_name="panel")
    inno_df = pd.read_excel(config.innovation_path, sheet_name="panel")
    merge_cols = ["stkcd", "year"]
    search_cols = merge_cols + [c for c in config.search_core + config.controls if c in search_df.columns]
    inno_cols = merge_cols + [c for c in config.innovation_candidates if c in inno_df.columns]
    df = search_df[search_cols].merge(inno_df[inno_cols], on=merge_cols, how="inner").drop_duplicates(subset=merge_cols)
    if config.treatment.startswith("SVI_"):
        df[config.treatment] = np.log1p(df[config.treatment].clip(lower=0))
    for col in [c for c in config.controls + [config.treatment, config.outcome] if c in df.columns]:
        df[col] = winsorize(df[col], config.winsor_lower, config.winsor_upper)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["stkcd", "year", config.outcome, config.treatment]).copy()
    return df


def split_panel(df: pd.DataFrame, train_end: int, valid_year: int):
    train_df = df.loc[df["year"] <= train_end].copy()
    valid_df = df.loc[df["year"] == valid_year].copy()
    test_df = df.loc[df["year"] > valid_year].copy()
    return train_df, valid_df, test_df


def build_preprocessor(feature_cols, numeric_cols, categorical_cols):
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline(
        [("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer([("num", num_pipe, numeric_cols), ("cat", cat_pipe, categorical_cols)])


def prepare_dataset(config, df=None):
    df = load_panel(config) if df is None else df.copy()
    feature_cols = [c for c in config.controls if c in df.columns] + ["year"]
    numeric_cols = [c for c in config.controls if c in df.columns]
    categorical_cols = ["year"]
    train_df, valid_df, test_df = split_panel(df, config.train_end, config.valid_year)
    preprocessor = build_preprocessor(feature_cols, numeric_cols, categorical_cols)
    preprocessor.fit(train_df[feature_cols])

    def transform(frame):
        x = preprocessor.transform(frame[feature_cols])
        if hasattr(x, "toarray"):
            x = x.toarray()
        x = np.asarray(x, dtype=np.float32)
        y = frame[config.outcome].to_numpy(dtype=np.float32)
        t = frame[config.treatment].to_numpy(dtype=np.float32)
        return x, y, t

    x_train, y_train, t_train = transform(train_df)
    x_valid, y_valid, t_valid = transform(valid_df)
    x_test, y_test, t_test = transform(test_df)
    x_all = np.vstack([x_train, x_valid, x_test])
    y_all = np.concatenate([y_train, y_valid, y_test])
    t_all = np.concatenate([t_train, t_valid, t_test])
    return {
        "df": df,
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "x_train": x_train,
        "y_train": y_train,
        "t_train": t_train,
        "x_valid": x_valid,
        "y_valid": y_valid,
        "t_valid": t_valid,
        "x_test": x_test,
        "y_test": y_test,
        "t_test": t_test,
        "x_all": x_all,
        "y_all": y_all,
        "t_all": t_all,
    }
