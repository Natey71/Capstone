from pathlib import Path
import joblib
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def build_models() -> dict:
    """Create baseline and machine learning classification models."""
    return {
        "baseline_majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
    }


def evaluate_predictions(y_true, y_pred) -> dict:
    """Calculate classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def train_and_evaluate_models(X_train, X_test, y_train, y_test) -> tuple[dict, pd.DataFrame]:
    """Train all models and return fitted models plus metrics."""
    models = build_models()
    results = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = evaluate_predictions(y_test, predictions)
        metrics["model"] = model_name
        results.append(metrics)

    metrics_df = pd.DataFrame(results)
    return models, metrics_df


def save_model(model, path: str | Path) -> None:
    """Save trained model artifact."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path):
    """Load trained model artifact."""
    return joblib.load(path)