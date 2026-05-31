from pathlib import Path
import pandas as pd
import numpy as np


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the raw data from a csv file."""
    return pd.read_csv(path, index_col=0)

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw data by removing rows with missing values."""
    df = df.copy()
    
    if("Unnamed: 0" in df.columns):
        df = df.sort_values("Unnamed: 0")
        df = df.drop(columns=["Unnamed: 0"])
    
    return df

    
def create_price_direction_target(
    df: pd.DataFrame,
    price_col: str = "nswprice",
    target_col: str = "price_direction"
) -> pd.DataFrame:
    """
    Create a binary target:
    1 = next interval price increased
    0 = next interval price decreased

    Rows where the next price is equal to the current price are removed.
    """
    df = df.copy()
    df["next_price"] = df[price_col].shift(-1)

    df[target_col] = np.where(
        df["next_price"] > df[price_col],
        1,
        np.where(df["next_price"] < df[price_col], 0, np.nan)
    )

    df = df.dropna(subset=[target_col])
    df[target_col] = df[target_col].astype(int)
    df = df.drop(columns=["next_price"])

    return df


def save_processed_data(df: pd.DataFrame, path: str | Path) -> None:
    """Save processed modeling data."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)