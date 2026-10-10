import numpy as np
import pandas as pd


def train_val_split(df: pd.DataFrame, val_fraction: float = 0.2, seed: int = 42):
    """Split a training dataframe into train/validation by ENGINE, not by row.

    Keeps every cycle of a given engine entirely on one side, so the model
    never sees part of an engine's life in training and the rest in
    validation — that would leak its exact degradation curve.
    """
    engine_ids = df["unit_number"].unique()
    rng = np.random.default_rng(seed)
    shuffled_ids = rng.permutation(engine_ids)

    n_val = int(len(shuffled_ids) * val_fraction)
    val_ids = set(shuffled_ids[:n_val])

    is_val = df["unit_number"].isin(val_ids)
    train_df = df[~is_val].copy()
    val_df = df[is_val].copy()
    return train_df, val_df
