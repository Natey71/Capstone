import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "date",
    "day",
    "period",
    "nswprice",
    "nswdemand",
    "vicprice",
    "vicdemand",
    "transfer",
    "period_sin",
    "period_cos",
    "day_sin",
    "day_cos",
]

TARGET_COLUMN = "price_direction"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical features for period and day.
    This helps the model understand repeating daily and weekly patterns.
    """
    df = df.copy()

    df["period_sin"] = np.sin(2 * np.pi * df["period"])
    df["period_cos"] = np.cos(2 * np.pi * df["period"])

    df["day_sin"] = np.sin(2 * np.pi * (df["day"] - 1) / 7)
    df["day_cos"] = np.cos(2 * np.pi * (df["day"] - 1) / 7)

    return df


def split_features_target(df: pd.DataFrame):
    """Split processed data into X and y."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def time_ordered_train_test_split(X, y, test_size: float = 0.2):
    """
    Split data without shuffling so the model is tested on later observations.
    This better matches a forecasting problem.
    """
    split_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test