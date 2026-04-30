"""
train_model.py — Train a Random Forest classifier to predict needs_intervention.

Pipeline:
  1. Load features from DuckDB
  2. Train/test split
  3. Train Random Forest
  4. Evaluate and print metrics
  5. Save model artifact to disk

Run directly:
    python -m src.modeling.train_model
"""

import pandas as pd
import numpy as np
import joblib
import yaml
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.build_features import build_features_from_db


# ── Config ─────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def print_evaluation(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> dict:
    """
    Print and return key evaluation metrics.

    Uses a configurable probability threshold (not just 0.5)
    so the business can tune precision vs recall tradeoff.

    Args:
        model:     Trained classifier.
        X_test:    Test feature matrix.
        y_test:    True labels.
        threshold: Probability cutoff for positive class.

    Returns:
        Dict of key metrics.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    auc     = roc_auc_score(y_test, y_proba)
    cm      = confusion_matrix(y_test, y_pred)
    report  = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n── Model Evaluation (threshold={threshold}) ────────────────")
    print(f"  ROC-AUC Score  : {auc:.4f}")
    print(f"  Accuracy       : {report['accuracy']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"              Predicted 0   Predicted 1")
    print(f"  Actual 0  :   {cm[0][0]:>8}      {cm[0][1]:>8}")
    print(f"  Actual 1  :   {cm[1][0]:>8}      {cm[1][1]:>8}")
    print(f"\n  Classification Report:")
    print(f"  {'':20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for label in ["0", "1"]:
        m = report[label]
        name = "No Intervention" if label == "0" else "Needs Intervention"
        print(f"  {name:20} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1-score']:>10.4f}")

    return {
        "roc_auc":   round(auc, 4),
        "accuracy":  round(report["accuracy"], 4),
        "precision": round(report["1"]["precision"], 4),
        "recall":    round(report["1"]["recall"], 4),
        "f1":        round(report["1"]["f1-score"], 4),
        "threshold": threshold,
    }


def print_feature_importance(
    model: RandomForestClassifier,
    feature_names: list[str],
    top_n: int = 10,
) -> None:
    """
    Print top N most important features from the trained model.

    Args:
        model:         Trained RandomForestClassifier.
        feature_names: List of feature column names.
        top_n:         Number of top features to display.
    """
    importances = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    print(f"\n── Top {top_n} Feature Importances ─────────────────────────")
    for feat, imp in importances.head(top_n).items():
        bar = "█" * int(imp * 40)
        print(f"  {feat:35} {imp:.4f}  {bar}")


# ── Main training pipeline ─────────────────────────────────────────────────────


def save_feature_names(feature_names: list[str], model_output_path: str | None = None) -> str:
    """Save feature names list alongside the model for use during scoring."""
    cfg = _load_config()
    base_path = model_output_path or cfg["model"]["model_output_path"]
    names_path = str(base_path).replace(".joblib", "_features.joblib")
    joblib.dump(feature_names, names_path)
    return names_path


def load_feature_names(model_output_path: str | None = None) -> list[str]:
    """Load saved feature names list."""
    cfg = _load_config()
    base_path = model_output_path or cfg["model"]["model_output_path"]
    names_path = str(base_path).replace(".joblib", "_features.joblib")
    if not Path(names_path).exists():
        raise FileNotFoundError(f"Feature names not found at '{names_path}'. Re-run training.")
    return joblib.load(names_path)

def train(
    table_name: str = "customer_communications",
    model_output_path: str | None = None,
) -> tuple[RandomForestClassifier, dict, list[str]]:
    """
    Full training pipeline: load → split → train → evaluate → save.

    Args:
        table_name:        DuckDB table to load data from.
        model_output_path: Where to save the .joblib model file.

    Returns:
        Tuple of (trained model, metrics dict, feature names list).
    """
    cfg = _load_config()
    model_cfg = cfg["model"]
    output_path = model_output_path or model_cfg["model_output_path"]
    threshold   = model_cfg["threshold"]

    print("── Step 1: Load features ────────────────────────────────")
    X, y = build_features_from_db(table_name)
    print(f"  Features : {X.shape[1]} columns, {X.shape[0]:,} rows")
    print(f"  Target   : {y.mean():.1%} positive rate")

    print("\n── Step 2: Train/test split ──────────────────────────────")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=model_cfg["test_size"],
        random_state=model_cfg["random_state"],
        stratify=y,   # preserve class balance in both splits
    )
    print(f"  Train : {len(X_train):,} rows")
    print(f"  Test  : {len(X_test):,} rows")

    print("\n── Step 3: Train Random Forest ───────────────────────────")
    model = RandomForestClassifier(
        n_estimators=model_cfg["n_estimators"],
        max_depth=model_cfg["max_depth"],
        random_state=model_cfg["random_state"],
        n_jobs=-1,          # use all CPU cores
        class_weight="balanced",  # handle slight class imbalance
    )
    model.fit(X_train, y_train)
    print(f"  Trained {model_cfg['n_estimators']} trees, max_depth={model_cfg['max_depth']}")

    print("\n── Step 4: Cross-validation (5-fold AUC) ─────────────────")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"  CV AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print("\n── Step 5: Evaluate on held-out test set ─────────────────")
    metrics = print_evaluation(model, X_test, y_test, threshold)

    print_feature_importance(model, X.columns.tolist())

    print("\n── Step 6: Save model ────────────────────────────────────")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"  Model saved to : {output_path}")

    names_path = save_feature_names(X.columns.tolist(), output_path)
    print(f"  Feature names  : {names_path}")

    print("\n✓ Training complete.")
    return model, metrics, X.columns.tolist()


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, metrics, features = train()
    print(f"\nFinal metrics: {metrics}")

