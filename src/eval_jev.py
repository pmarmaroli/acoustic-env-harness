"""eval_jev.py – Training and evaluation of the Jev acoustic classifier.

Usage (standalone):
    python -m src.eval_jev --runs-dir runs/

The module:
  1. Loads all ``*.json`` run files from the specified directory.
  2. Builds a labelled feature matrix from ``final_features`` + ground-truth
     ``label`` field (added manually or via the CLI ``--label`` option).
  3. Trains a Random Forest with StratifiedKFold cross-validation.
  4. Plots and prints the normalized 4×4 confusion matrix and Macro-F1 per model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

CLASSES = ["car", "bathroom", "outdoor", "office"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_runs(runs_dir: str) -> list[dict[str, Any]]:
    """Load all JSON run files from *runs_dir*.

    Each file must contain ``final_features`` (dict) and ``label`` (str).
    Files without either field are silently skipped.
    """
    records: list[dict[str, Any]] = []
    for path in sorted(Path(runs_dir).glob("*.json")):
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue

        features = data.get("final_features")
        label = data.get("label")
        if not features or not label:
            continue
        records.append({
            "model_name": data.get("model_name", path.stem),
            "label": label,
            "features": features,
        })
    return records


def build_feature_matrix(
    records: list[dict[str, Any]],
    feature_keys: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], LabelEncoder]:
    """Convert a list of records into a numeric feature matrix.

    Parameters
    ----------
    records : list[dict]
        Output of :func:`load_runs`.
    feature_keys : list[str] | None
        Ordered list of feature names to extract.  When *None*, the union of
        all keys found in the first record's features is used.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)  – integer encoded labels
    feature_keys : list[str]
    label_encoder : LabelEncoder
    """
    if feature_keys is None:
        feature_keys = list(dict.fromkeys(k for rec in records for k in rec["features"]))

    rows, labels = [], []
    for rec in records:
        row = []
        for k in feature_keys:
            val = rec["features"].get(k, 0.0)
            # Booleans and None → float
            if val is None:
                val = 0.0
            elif isinstance(val, bool):
                val = float(val)
            else:
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    val = 0.0
            row.append(val)
        rows.append(row)
        labels.append(rec["label"])

    X = np.array(rows, dtype=np.float64)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return X, y, feature_keys, le


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(
    X: np.ndarray,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    classifier: str = "random_forest",
    n_splits: int = 5,
) -> dict[str, Any]:
    """Train and cross-validate a classifier; return metrics.

    Parameters
    ----------
    X : feature matrix
    y : integer label vector
    label_encoder : fitted LabelEncoder
    classifier : ``"random_forest"`` or ``"logistic_regression"``
    n_splits : int
        Number of StratifiedKFold folds.

    Returns
    -------
    dict with keys: macro_f1, confusion_matrix, classification_report
    """
    if classifier == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=42)
    else:
        clf = RandomForestClassifier(n_estimators=200, random_state=42)

    scaler = StandardScaler()
    safe_splits = min(n_splits, int(np.min(np.bincount(y))))
    safe_splits = max(2, safe_splits)  # StratifiedKFold requires at least 2 splits
    skf = StratifiedKFold(n_splits=safe_splits, shuffle=True, random_state=42)

    all_preds, all_true = [], []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        clf.fit(X_train_s, y_train)
        all_preds.extend(clf.predict(X_test_s).tolist())
        all_true.extend(y_test.tolist())

    y_pred = np.array(all_preds)
    y_true = np.array(all_true)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_encoder.classes_))))
    report = classification_report(
        y_true, y_pred, target_names=label_encoder.classes_.tolist(), zero_division=0
    )

    return {
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": cm,
        "classification_report": report,
        "label_encoder": label_encoder,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    metrics: dict[str, Any],
    model_name: str = "",
    save_path: str | None = None,
) -> None:
    """Plot a normalized 4×4 confusion matrix.

    Falls back to a textual representation when matplotlib is unavailable.
    """
    cm = metrics["confusion_matrix"]
    le: LabelEncoder = metrics["label_encoder"]
    class_names = le.classes_.tolist()

    # Normalize
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm / row_sums

    macro_f1 = metrics["macro_f1"]
    title = f"Jev Classifier – {model_name}  |  Macro-F1 = {macro_f1:.4f}"

    if not HAS_MATPLOTLIB:
        print(title)
        print(metrics["classification_report"])
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format=".2f")
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Per-model summary
# ---------------------------------------------------------------------------

def evaluate_all_models(
    records: list[dict[str, Any]],
    classifier: str = "random_forest",
    n_splits: int = 5,
    plot: bool = True,
    output_dir: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Group records by model name, train/evaluate per model and print summary.

    Parameters
    ----------
    records : list[dict]
        Output of :func:`load_runs`.
    classifier : str
        Classifier type (see :func:`evaluate_classifier`).
    n_splits : int
        Cross-validation folds.
    plot : bool
        Whether to display confusion matrix plots.
    output_dir : str | None
        If provided, save confusion matrix figures there.

    Returns
    -------
    dict mapping model_name → metrics dict
    """
    # Group by model
    by_model: dict[str, list[dict]] = {}
    for rec in records:
        by_model.setdefault(rec["model_name"], []).append(rec)

    results: dict[str, dict[str, Any]] = {}

    for model_name, model_records in by_model.items():
        if len(model_records) < 2:
            print(f"[eval_jev] Skipping '{model_name}': need ≥ 2 labelled samples.")
            continue

        X, y, feature_keys, le = build_feature_matrix(model_records)
        metrics = evaluate_classifier(X, y, le, classifier=classifier, n_splits=n_splits)
        results[model_name] = metrics

        print(f"\n{'=' * 60}")
        print(f"Model : {model_name}   |   Macro-F1 = {metrics['macro_f1']:.4f}")
        print(metrics["classification_report"])

        if plot:
            save_path: str | None = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, f"cm_{model_name}.png")
            plot_confusion_matrix(metrics, model_name=model_name, save_path=save_path)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate Jev acoustic classifier on saved run files."
    )
    p.add_argument("--runs-dir", default="runs", help="Directory containing *.json run files.")
    p.add_argument(
        "--classifier",
        default="random_forest",
        choices=["random_forest", "logistic_regression"],
    )
    p.add_argument("--n-splits", type=int, default=5, help="StratifiedKFold splits.")
    p.add_argument("--no-plot", action="store_true", help="Suppress matplotlib figures.")
    p.add_argument("--output-dir", default=None, help="Directory to save confusion matrix PNGs.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    records = load_runs(args.runs_dir)
    if not records:
        print(f"[eval_jev] No labelled run files found in '{args.runs_dir}'.")
        return
    evaluate_all_models(
        records,
        classifier=args.classifier,
        n_splits=args.n_splits,
        plot=not args.no_plot,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
