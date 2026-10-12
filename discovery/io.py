"""Uniform prediction/metadata files so every model is scored by the same code."""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from .data import REPO

PRED_DIR = REPO / "results" / "predictions"


def save_predictions(model: str, seed: int, rec: pd.DataFrame, preds: dict[str, np.ndarray],
                     meta: dict) -> None:
    """preds maps split name -> predictions aligned with rec[rec.split == name]."""
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for split, p in preds.items():
        part = rec.loc[rec["split"] == split, ["row_id", "split", "pce"]].copy()
        part["pred"] = np.asarray(p, dtype=float)
        frames.append(part)
    stem = f"{model}__seed{seed}"
    pd.concat(frames).to_csv(PRED_DIR / f"{stem}.csv", index=False)
    (PRED_DIR / f"{stem}.json").write_text(json.dumps({"model": model, "seed": seed, **meta}, indent=2))


@contextmanager
def gpu_timer(meta: dict, n_gpus: int = 1):
    t0 = time.time()
    yield
    meta["wall_min"] = round((time.time() - t0) / 60, 2)
    meta["gpu_min"] = round(meta["wall_min"] * n_gpus, 2)
