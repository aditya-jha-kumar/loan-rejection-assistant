"""Ensure dataset/model exist (local or Vercel /tmp)."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from logging_utils import get_logger

logger = get_logger("runtime_assets")

ROOT = Path(__file__).resolve().parent.parent
DATASET_URL = (
    "https://raw.githubusercontent.com/"
    "sangramdedge/Credit-Risk-Analysis-Prediction-Dashboard/"
    "main/credit_risk_dataset.csv"
)


def runtime_root() -> Path:
    if os.getenv("VERCEL"):
        return Path("/tmp/loan-rejection")
    return ROOT


def data_path() -> Path:
    bundled = ROOT / "data" / "loan_dataset.csv"
    if bundled.exists() and bundled.stat().st_size > 100_000:
        return bundled
    return runtime_root() / "data" / "loan_dataset.csv"


def model_path() -> Path:
    for candidate in (
        ROOT / "models" / "loan_model.pkl",
        ROOT / "models" / "loan_model.joblib",
    ):
        if candidate.exists():
            return candidate
    return runtime_root() / "models" / "loan_model.pkl"


def ensure_dataset(path: Path | None = None) -> Path:
    dest = Path(path) if path else data_path()
    if dest.exists() and dest.stat().st_size > 100_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading dataset to %s", dest)
    urllib.request.urlretrieve(DATASET_URL, dest)
    return dest


def ensure_model(model_file: Path | None = None, dataset: Path | None = None) -> Path:
    dest = Path(model_file) if model_file else model_path()
    if dest.exists():
        return dest
    from data import run_pipeline
    from model import run_training

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Training model at %s", dest)
    csv_path = ensure_dataset(dataset)
    X_train, X_test, y_train, y_test, X, y = run_pipeline(csv_path)
    model, _ = run_training(X_train, X_test, y_train, y_test)
    if dest.resolve() != (ROOT / "models" / "loan_model.pkl").resolve():
        import joblib

        joblib.dump(model, dest)
        logger.info("Model copied to %s", dest)
    return dest


def ensure_assets() -> tuple[Path, Path]:
    csv = ensure_dataset()
    pkl = ensure_model(dataset=csv)
    return csv, pkl
