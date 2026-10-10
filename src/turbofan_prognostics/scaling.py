import pandas as pd


def drop_constant_columns(train_df: pd.DataFrame, feature_cols: list) -> list:
    """Remove feature columns with zero variance in the training data.

    A column that never changes carries no information for a model to
    learn from, and would also cause a divide-by-zero when scaling.
    """
    stds = train_df[feature_cols].std()
    return [col for col in feature_cols if stds[col] > 0]


def fit_scaler(train_df: pd.DataFrame, feature_cols: list) -> dict:
    """Compute per-column mean and std from TRAINING data only.

    These statistics get reused (never recomputed) on validation/test data,
    so no information from those sets leaks into how we scale.
    """
    means = train_df[feature_cols].mean()
    stds = train_df[feature_cols].std()
    return {"means": means, "stds": stds}


def apply_scaler(df: pd.DataFrame, feature_cols: list, scaler: dict) -> pd.DataFrame:
    """Scale columns to zero mean, unit variance, using a previously-fit scaler."""
    df = df.copy()
    df[feature_cols] = (df[feature_cols] - scaler["means"]) / scaler["stds"]
    return df
