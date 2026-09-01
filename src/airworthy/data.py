from pathlib import Path

import pandas as pd

COLUMN_NAMES = (
    ["unit_number", "time_cycles", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_raw(dataset_id: str, split: str, raw_dir: str = "data/raw") -> pd.DataFrame:
    """Load one raw C-MAPSS file and attach proper column names.

    dataset_id: one of "FD001", "FD002", "FD003", "FD004"
    split: "train" or "test"
    """
    filename = f"{split}_{dataset_id}.txt"
    filepath = Path(raw_dir) / filename
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMN_NAMES)
    return df


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Add a RUL (Remaining Useful Life) column to a *training* dataframe.

    Only valid for training data, where each engine runs until failure —
    its last recorded cycle IS the failure point.
    """
    df = df.copy()
    max_cycle_per_engine = df.groupby("unit_number")["time_cycles"].transform("max")
    df["RUL"] = max_cycle_per_engine - df["time_cycles"]
    return df
