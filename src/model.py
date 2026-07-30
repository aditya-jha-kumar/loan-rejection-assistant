"""
model.py
--------
Handles model training, evaluation, and persistence for the
Explainable Loan Rejection Assistant.

Dataset context:
    loan_status: 0 = good standing, 1 = default (high risk)
    Imbalance:   3.6:1 (non-default vs default)
    Train rows:  25,217
    Test rows:    6,305

Functions:
    train_model(X_train, y_train)         - Train XGBoost classifier
    evaluate_model(model, X_test, y_test) - Full evaluation report
    save_model(model, path)               - Persist model to disk
    load_model(path)                      - Load model from disk
    run_training(X_train, X_test,         - Train + evaluate + save
                 y_train, y_test)
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "models" / "loan_model.pkl"
DEFAULT_CM_PATH = ROOT / "models" / "confusion_matrix.png"


# TRAIN MODEL

def train_model(X_train, y_train):
    """
    Train an XGBoost binary classifier on loan data.

    Why XGBoost?
    - Handles tabular data better than neural networks at this scale
    - Built-in feature importance (used by SHAP TreeExplainer)
    - Fast training, no feature scaling required
    - Industry standard for credit risk modeling

    Why scale_pos_weight=3?
    - Dataset has 3.6:1 imbalance (non-default vs default)
    - This tells XGBoost to weight the minority class (default)
      3x more during training, so it doesn't just learn to always
      predict the majority class
    - Formula: count(negative) / count(positive) ≈ 24715/6807 ≈ 3.6
      We use 3 as a slightly conservative estimate

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series):    Training labels

    Returns:
        xgb.XGBClassifier: Trained model
    """
    model = xgb.XGBClassifier(
        n_estimators=200,          # number of trees — more than default for better accuracy
        max_depth=5,               # tree depth — controls overfitting
        learning_rate=0.1,         # how much each tree contributes
        scale_pos_weight=3,        # handles 3.6:1 class imbalance
        random_state=42,           # reproducibility
        eval_metric="auc",         # optimize for AUC during training
    )

    model.fit(X_train, y_train)

    print("Model trained successfully")
    print(f"Number of trees: {model.n_estimators}")
    print(f"Features used:   {model.n_features_in_}")

    return model


# EVALUATE MODEL

def evaluate_model(model, X_test, y_test, cm_path=DEFAULT_CM_PATH):
    """
    Run full evaluation on the test set.

    Metrics reported:
    - Classification report (precision, recall, F1 per class)
    - ROC-AUC score (overall discrimination ability)
    - Confusion matrix (visual breakdown of errors)

    Why ROC-AUC for this dataset?
    With 3.6:1 imbalance, accuracy is misleading — a model that
    always predicts 0 would score 78% accuracy. ROC-AUC measures
    how well the model separates the two classes regardless of
    the decision threshold, making it imbalance-robust.

    Args:
        model: Trained XGBoost model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series):    Test labels
        cm_path: Path to save confusion matrix image

    Returns:
        dict: {predictions, probabilities, auc_score}
    """
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probabilities)

    print("=" * 50)
    print("MODEL EVALUATION REPORT")
    print("=" * 50)
    print(f"\nROC-AUC Score: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Good Standing (0)", "Default (1)"]
    ))

    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Good Standing", "Default"]
    )
    disp.plot(cmap="Blues")
    plt.title("XGBoost Confusion Matrix — Credit Risk")
    plt.tight_layout()

    cm_path = Path(cm_path)
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "auc_score": auc,
    }


# SAVE AND LOAD MODEL

def save_model(model, path=DEFAULT_MODEL_PATH):
    """
    Persist trained model to disk using joblib.

    Why joblib over pickle?
    joblib is optimized for large numpy arrays — much faster
    for ML models than Python's built-in pickle.

    Args:
        model: Trained XGBoost model
        path (str): Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path=DEFAULT_MODEL_PATH):
    """
    Load a previously trained model from disk.

    Use this instead of retraining — training takes time,
    loading takes milliseconds.

    Args:
        path (str): Path to saved model file

    Returns:
        Trained XGBoost model
    """
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model


# FULL TRAINING PIPELINE

def run_training(X_train, X_test, y_train, y_test):
    """
    Run the complete training pipeline:
    train → evaluate → save

    Args:
        X_train, X_test: Feature splits
        y_train, y_test: Label splits

    Returns:
        tuple: (model, evaluation_results)
    """
    model = train_model(X_train, y_train)
    results = evaluate_model(model, X_test, y_test)
    save_model(model)

    return model, results


# QUICK TEST

if __name__ == "__main__":
    from data import DEFAULT_DATA_PATH, run_pipeline

    X_train, X_test, y_train, y_test, X, y = run_pipeline(DEFAULT_DATA_PATH)
    model, results = run_training(X_train, X_test, y_train, y_test)

    print(f"\nFinal AUC: {results['auc_score']:.4f}")
    print("Ready for SHAP explainability.")
