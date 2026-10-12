"""Test-set metrics, following LLM4Mat-Bench (MAD:MAE) plus R2 and AUC."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

from .data import CLS_THRESHOLD


def regression_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y, p)
    mad = np.mean(np.abs(y - y.mean()))
    labels = y > CLS_THRESHOLD
    return {
        "mae": mae,
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": r2_score(y, p),
        "mad_mae": mad / mae,
        # Predicted PCE is the classification score, so one regressor gives both.
        "auc": roc_auc_score(labels, p) if 0 < labels.sum() < len(labels) else np.nan,
    }


def bootstrap_ci(y, p, metric="mae", n=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    vals = [regression_metrics(y[i], p[i])[metric] for i in idx]
    return tuple(np.quantile(vals, [alpha / 2, 1 - alpha / 2]))


def paired_bootstrap_p(y, p_a, p_b, n=10000, seed=0) -> float:
    """One-sided p-value that model A's MAE is not lower than model B's."""
    rng = np.random.default_rng(seed)
    err_a, err_b = np.abs(y - p_a), np.abs(y - p_b)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    diff = err_a[idx].mean(1) - err_b[idx].mean(1)
    return float((diff >= 0).mean())
