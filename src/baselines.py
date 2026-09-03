"""
baselines.py
------------
Baseline models + probability calibration for fair comparison.

Supports the paper / portfolio claim that XGBoost is competitive
on this tabular credit dataset, and that calibrated probabilities
are used for decision confidence display.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

from logging_utils import get_logger

logger = get_logger("baselines")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_PATH = ROOT / "models" / "baseline_comparison.json"
DEFAULT_CALIBRATED_PATH = ROOT / "models" / "loan_model_calibrated.pkl"


def _metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1_default": float(f1_score(y_true, y_pred, pos_label=1)),
        "precision_default": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_default": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_good": float(f1_score(y_true, y_pred, pos_label=0)),
    }


def build_baselines(random_state: int = 42) -> dict:
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=3,
            random_state=random_state,
            eval_metric="auc",
            enable_categorical=False,
        ),
    }


def run_baseline_comparison(X_train, X_test, y_train, y_test,
                            save_path: Path | str = DEFAULT_BASELINE_PATH) -> dict:
    """Train LR / RF / XGBoost and write metrics JSON."""
    import json

    results = {}
    models = build_baselines()
    for name, model in models.items():
        logger.info("Training baseline: %s", name)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)[:, 1]
        else:
            prob = pred.astype(float)
        results[name] = _metrics(y_test, pred, prob)
        logger.info("%s AUC=%.4f", name, results[name]["roc_auc"])

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Baseline comparison saved to %s", save_path)
    return results


def calibrate_model(model, X_train, y_train, method: str = "isotonic"):
    """
    Wrap a fitted classifier with probability calibration.

    Uses a hold-out style CV calibration (sklearn CalibratedClassifierCV).
    Note: for production, prefer calibrating on a dedicated validation set.
    """
    calibrator = CalibratedClassifierCV(model, method=method, cv=3)
    calibrator.fit(X_train, y_train)
    return calibrator


def save_calibrated(model, path: Path | str = DEFAULT_CALIBRATED_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Calibrated model saved to %s", path)


def leak_feature_ablation(X_train, X_test, y_train, y_test,
                          leak_candidates: list[str] | None = None) -> dict:
    """
    Train XGBoost with/without suspected underwriting-leak features.

    loan_grade and loan_int_rate often encode prior underwriting decisions.
    Reporting this ablation is important for paper honesty / Model Cards.
    """
    leak_candidates = leak_candidates or ["loan_grade", "loan_int_rate"]
    present = [c for c in leak_candidates if c in X_train.columns]

    def _fit(cols_drop: list[str]) -> dict:
        keep = [c for c in X_train.columns if c not in cols_drop]
        m = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            scale_pos_weight=3, random_state=42, eval_metric="auc",
            enable_categorical=False,
        )
        m.fit(X_train[keep], y_train)
        prob = m.predict_proba(X_test[keep])[:, 1]
        pred = m.predict(X_test[keep])
        return _metrics(y_test, pred, prob)

    out = {
        "full_model": _fit([]),
        "without_leak_candidates": _fit(present),
        "dropped_features": present,
    }
    logger.info(
        "Leakage ablation: full AUC=%.4f vs ablated AUC=%.4f (dropped %s)",
        out["full_model"]["roc_auc"],
        out["without_leak_candidates"]["roc_auc"],
        present,
    )
    return out


if __name__ == "__main__":
    from data import DEFAULT_DATA_PATH, run_pipeline
    from model import run_training

    X_train, X_test, y_train, y_test, X, y = run_pipeline(DEFAULT_DATA_PATH)
    comparison = run_baseline_comparison(X_train, X_test, y_train, y_test)
    ablation = leak_feature_ablation(X_train, X_test, y_train, y_test)
    model, _ = run_training(X_train, X_test, y_train, y_test)
    calibrated = calibrate_model(model, X_train, y_train)
    save_calibrated(calibrated)
    print(comparison)
    print(ablation)
