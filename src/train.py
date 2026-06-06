from pathlib import Path

from data import (
    load_raw_data_into_csv, 
    load_raw_data,
    clean_raw_data,
    create_price_direction_target,
    save_processed_data,
)
from features import (
    add_time_features,
    split_features_target,
    time_ordered_train_test_split,
)
from model import train_and_evaluate_models, save_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_data.csv"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.csv"
MODEL_DIR = PROJECT_ROOT / "models"


def main():
    load_raw_data_into_csv(RAW_DATA_PATH)
    raw_df = load_raw_data(RAW_DATA_PATH)
    clean_df = clean_raw_data(raw_df)
    target_df = create_price_direction_target(clean_df)
    feature_df = add_time_features(target_df)

    save_processed_data(feature_df, PROCESSED_DATA_PATH)

    X, y = split_features_target(feature_df)
    X_train, X_test, y_train, y_test = time_ordered_train_test_split(X, y)

    models, metrics_df = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_PATH, index=False)

    best_model_name = metrics_df.sort_values("f1_score", ascending=False).iloc[0]["model"]
    save_model(models[best_model_name], MODEL_DIR / "best_model.pkl")

    print(metrics_df)
    print(f"Best model: {best_model_name}")


if __name__ == "__main__":
    main()