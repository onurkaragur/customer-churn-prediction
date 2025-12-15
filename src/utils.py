import os
import pickle
from typing import Any, Dict, Optional

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_pickle(obj: Any, path: str) -> None:
    """Save an object to `path` using pickle. Creates parent directories as needed."""
    _ensure_parent_dir(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    """Load and return a pickle object from `path`.

    Raises FileNotFoundError if the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pickle file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model: Any, path: str) -> None:
    """Alias for `save_pickle` for semantic clarity when saving models."""
    save_pickle(model, path)


def load_model(path: str) -> Any:
    """Alias for `load_pickle` for semantic clarity when loading models."""
    return load_pickle(path)


def metrics_dict(y_true, y_pred, y_prob: Optional[float] = None) -> Dict[str, Optional[float]]:
    """Compute basic classification metrics and return as dict."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = None
    return metrics


def print_metrics(y_true, y_pred, y_prob: Optional[float] = None, model_name: Optional[str] = None) -> None:
    """Prints a human-friendly classification report and summary metrics."""
    header = f"=== Metrics: {model_name} ===" if model_name else "=== Metrics ==="
    print(header)
    try:
        print(classification_report(y_true, y_pred))
    except Exception:
        pass
    print(metrics_dict(y_true, y_pred, y_prob))
