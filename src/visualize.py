from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from data import (
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def prepare_output_directory() -> None:
    """Create the reports/figures directory if it does not exist."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_modeling_data() -> pd.DataFrame:
    """
    Load the raw data, clean it, create the target variable,
    add engineered features, and save the processed dataset.
    """
    raw_df = load_raw_data(RAW_DATA_PATH)
    clean_df = clean_raw_data(raw_df)
    target_df = create_price_direction_target(clean_df)
    feature_df = add_time_features(target_df)

    save_processed_data(feature_df, PROCESSED_DATA_PATH)

    return feature_df


def plot_class_distribution(df: pd.DataFrame) -> None:
    """Create a bar chart showing DOWN and UP class counts."""
    target_counts = (
        df["price_direction"]
        .map({0: "DOWN", 1: "UP"})
        .value_counts()
        .reindex(["DOWN", "UP"])
    )

    plt.figure(figsize=(8, 5))
    target_counts.plot(kind="bar")
    plt.title("Class Distribution of Electricity Price Movement")
    plt.xlabel("Price Movement Class")
    plt.ylabel("Number of Observations")
    plt.xticks(rotation=0)
    plt.tight_layout()

    output_path = FIGURES_DIR / "class_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrix(model, X_test, y_test) -> None:
    """Create a confusion matrix for the saved best model."""
    predictions = model.predict(X_test)

    plt.figure(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=[0, 1],
        display_labels=["DOWN", "UP"],
    )
    plt.title("Confusion Matrix for Best Model")
    plt.tight_layout()

    output_path = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_model_comparison() -> None:
    """Create a grouped bar chart comparing model performance metrics."""
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"{METRICS_PATH} was not found. Run python src/train.py first."
        )

    metrics_df = pd.read_csv(METRICS_PATH)

    required_columns = ["model", "accuracy", "precision", "recall", "f1_score"]
    missing_columns = [
        col for col in required_columns if col not in metrics_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"metrics.csv is missing required columns: {missing_columns}"
        )

    plot_df = metrics_df[required_columns].set_index("model")

    plt.figure(figsize=(10, 6))
    plot_df.plot(kind="bar")
    plt.title("Model Performance Comparison")
    plt.xlabel("Model")
    plt.ylabel("Metric Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Metric")
    plt.tight_layout()

    output_path = FIGURES_DIR / "model_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def get_inner_model(model):
    """
    Return the estimator that contains feature importance or coefficients.

    This handles both standalone models and sklearn Pipelines.
    """
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"]

    return model


def plot_feature_importance(model, feature_names) -> None:
    """
    Create a feature importance chart.

    Random forest models use feature_importances_.
    Logistic regression models use absolute coefficient values.
    If the selected model does not support either, a placeholder figure is saved.
    """
    inner_model = get_inner_model(model)

    if hasattr(inner_model, "feature_importances_"):
        importance_values = inner_model.feature_importances_
        importance_label = "Feature Importance"

    elif hasattr(inner_model, "coef_"):
        importance_values = abs(inner_model.coef_[0])
        importance_label = "Absolute Coefficient Value"

    else:
        plt.figure(figsize=(8, 5))
        plt.text(
            0.5,
            0.5,
            "Feature importance is not available for the selected model.",
            ha="center",
            va="center",
            wrap=True,
        )
        plt.axis("off")
        plt.tight_layout()

        output_path = FIGURES_DIR / "feature_importance.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        return

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_values,
        }
    ).sort_values("importance", ascending=False)

    top_features = importance_df.head(10).sort_values(
        "importance",
        ascending=True,
    )

    plt.figure(figsize=(9, 6))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("Top 10 Feature Importance Values")
    plt.xlabel(importance_label)
    plt.ylabel("Feature")
    plt.tight_layout()

    output_path = FIGURES_DIR / "feature_importance.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    prepare_output_directory()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} was not found. Run python src/train.py first."
        )

    feature_df = load_modeling_data()

    X, y = split_features_target(feature_df)
    _, X_test, _, y_test = time_ordered_train_test_split(X, y)

    model = joblib.load(MODEL_PATH)

    plot_class_distribution(feature_df)
    plot_confusion_matrix(model, X_test, y_test)
    plot_model_comparison()
    plot_feature_importance(model, X.columns)

    print(f"Saved figures to: {FIGURES_DIR}")
    print("Created class_distribution.png")
    print("Created confusion_matrix.png")
    print("Created model_comparison.png")
    print("Created feature_importance.png")


if __name__ == "__main__":
    main()