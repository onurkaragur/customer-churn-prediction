"""Helpers to load preprocessor and models for the app.

This module uses the artifacts saved under `src/config.py` and `src/utils.py`.
"""
from typing import Any, Dict, Optional

import pandas as pd

from src.config import MODEL_LOGREG, MODEL_RF, MODEL_XGB, PREPROCESSOR
from src.utils import load_model, load_pickle


def load_preprocessor() -> Optional[Any]:
    try:
        return load_pickle(PREPROCESSOR)
    except FileNotFoundError:
        return None


def load_models() -> Dict[str, Any]:
    models = {}
    for name, path in ("logreg", MODEL_LOGREG), ("rf", MODEL_RF), ("xgb", MODEL_XGB):
        try:
            models[name] = load_model(path)
        except FileNotFoundError:
            continue
    return models


def get_model(models: Dict[str, Any], name: str):
    return models.get(name)


def predict(model: Any, preprocessor: Optional[Any], X: pd.DataFrame):
    """Return (preds, probs) for given pandas DataFrame X.

    - If a `preprocessor` with `transform` is provided, apply it to `X` first.
    - `probs` will be None if the model doesn't implement `predict_proba`.
    """
    if preprocessor is not None:
        try:
            X_proc = preprocessor.transform(X)
        except Exception:
            # If preprocessor expects raw arrays and fails, try passing X values
            X_proc = preprocessor.transform(X.values)
    else:
        X_proc = X

    preds = model.predict(X_proc)
    probs = None
    try:
        probs = model.predict_proba(X_proc)[:, 1]
    except Exception:
        pass

    return preds, probs
