"""Tabular baselines: mean predictor, LightGBM (paper 18 / all fields), TabPFN."""
import argparse

import lightgbm as lgb
import numpy as np

from discovery.data import load_processed, split_masks
from discovery.io import gpu_timer, save_predictions
from discovery.metrics import regression_metrics

LGBM_PARAMS = dict(n_estimators=5000, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def fit_lgbm(X, y, m, seed):
    model = lgb.LGBMRegressor(random_state=seed, **LGBM_PARAMS)
    model.fit(X[m["train"]], y[m["train"]], eval_set=[(X[m["val"]], y[m["val"]])],
              eval_metric="l1", callbacks=[lgb.early_stopping(200, verbose=False)])
    return model, {"best_iteration": int(model.best_iteration_)}


def fit_tabpfn(X, y, m, seed):
    from tabpfn import TabPFNRegressor
    X = X.copy()
    for c in X.columns[X.dtypes == "category"]:
        X[c] = X[c].cat.codes.replace(-1, np.nan).astype(float)
    model = TabPFNRegressor(device="cuda", random_state=seed, ignore_pretraining_limits=True)
    model.fit(X[m["train"]], y[m["train"]])
    return _Frame(model, X), {}


class _Frame:
    """Applies the categorical->code conversion done at fit time."""
    def __init__(self, model, X):
        self.model, self.X = model, X

    def predict(self, X):
        return self.model.predict(self.X.loc[X.index])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mean", "lgbm18", "lgbmall", "tabpfn"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    rec = load_processed("records")
    y = rec["pce"].to_numpy()
    m = split_masks(rec)
    X = load_processed("tab18" if args.model == "lgbm18" else "taball")

    for seed in args.seeds:
        meta = {"input": "tabular", "n_features": X.shape[1]}
        with gpu_timer(meta, n_gpus=1 if args.model == "tabpfn" else 0):
            if args.model == "mean":
                preds = {s: np.full(m[s].sum(), y[m["train"]].mean()) for s in ("val", "test")}
            else:
                fit = fit_lgbm if args.model.startswith("lgbm") else fit_tabpfn
                model, extra = fit(X, y, m, seed)
                meta.update(extra)
                preds = {s: model.predict(X[m[s]]) for s in ("val", "test")}
        meta["val"] = regression_metrics(y[m["val"]], preds["val"])
        save_predictions(args.model, seed, rec, preds, meta)
        print(args.model, seed, regression_metrics(y[m["test"]], preds["test"]))
        if args.model == "mean":
            break


if __name__ == "__main__":
    main()
