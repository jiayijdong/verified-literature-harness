"""Aggregate saved predictions into the main table (LaTeX + Markdown), metrics JSON, best-model figure."""
import json

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from discovery.data import CLS_THRESHOLD, REPO
from discovery.io import PRED_DIR
from discovery.metrics import bootstrap_ci, paired_bootstrap_p, regression_metrics

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = REPO / "results"
# key: (display name, input, trainable params, group)
ROWS = {
    "mean":          ("Training-set mean", "--", "--", "Reference"),
    "lgbm18":        ("LightGBM, expert descriptors", "Tabular", "--", "Tabular"),
    "lgbmall":       ("LightGBM, all recipe fields", "Tabular", "--", "Tabular"),
    "tabpfn":        ("TabPFN v2, all recipe fields", "Tabular", "--", "Tabular"),
    "matscibert":    ("MatSciBERT, fine-tuned", "Text (512 tok)", "110M", "Fine-tuned LM"),
    "t5enc":         ("T5-base encoder, fine-tuned", "Text (1024 tok)", "110M", "Fine-tuned LM"),
    "llama8b_embed": ("Llama-3.1-8B embeddings + ridge", "Text", "4K (head)", "Frozen LLM"),
    "llama8b_0shot": ("Llama-3.1-8B-Instruct, 0-shot", "Text", "0", "Prompted LLM"),
    "llama8b_5shot": ("Llama-3.1-8B-Instruct, 5-shot", "Text", "0", "Prompted LLM"),
}
METRICS = [("mae", "MAE", False), ("mad_mae", "MAD:MAE", True), ("r2", "$R^2$", True), ("auc", "AUC", True)]

# Figure palette: categorical slots 1-2 of the reference palette + a neutral for references.
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"


def load():
    runs = {}
    for meta_path in sorted(PRED_DIR.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        df = pd.read_csv(meta_path.with_suffix(".csv"))
        test = df[df["split"] == "test"].sort_values("row_id")
        runs.setdefault(meta["model"], []).append((meta, test))
    return runs


def summarize(runs):
    rows, ens = {}, {}
    for key in ROWS:
        if key not in runs:
            continue
        per_seed = [regression_metrics(t["pce"].to_numpy(), t["pred"].to_numpy()) for _, t in runs[key]]
        y = runs[key][0][1]["pce"].to_numpy()
        p = np.mean([t["pred"].to_numpy() for _, t in runs[key]], axis=0)   # seed-averaged
        ens[key] = (y, p)
        metas = [mt for mt, _ in runs[key]]
        rows[key] = {
            "n_seeds": len(per_seed),
            **{f"{k}_mean": float(np.mean([r[k] for r in per_seed])) for k, *_ in METRICS},
            **{f"{k}_std": float(np.std([r[k] for r in per_seed])) for k, *_ in METRICS},
            "mae_ci95_seedavg": [float(v) for v in bootstrap_ci(y, p, "mae")],
            "gpu_min_per_run": float(np.mean([mt.get("gpu_min", 0.0) for mt in metas])),
            "invalid_rate": metas[0].get("invalid_rate"),
        }
    return rows, ens


def significance(rows, ens):
    """Best (lowest mean MAE) vs. every model, plus each fine-tuned LM vs. the best tabular model."""
    ranked = sorted((k for k in rows if k != "mean"), key=lambda k: rows[k]["mae_mean"])
    best = ranked[0]
    y, pb = ens[best]
    p_best = {ref: paired_bootstrap_p(y, pb, ens[ref][1]) for ref in ranked[1:]}
    tab = min((k for k in rows if ROWS[k][1] == "Tabular"), key=lambda k: rows[k]["mae_mean"])
    p_lm_vs_tab = {k: paired_bootstrap_p(y, ens[k][1], ens[tab][1])
                   for k in rows if ROWS[k][3] == "Fine-tuned LM"}
    return best, p_best, tab, p_lm_vs_tab


DECIMALS = {"mae": 2, "mad_mae": 2, "r2": 2, "auc": 3}   # AUCs differ in the third decimal


def fmt(rows, key, k, best_val):
    mu, sd, d = rows[key][f"{k}_mean"], rows[key][f"{k}_std"], DECIMALS[k]
    s = f"{mu:.{d}f}" if rows[key]["n_seeds"] == 1 else f"{mu:.{d}f}$\\pm${sd:.{d}f}"
    s = s.replace("-", "$-$", 1) if s.startswith("-") else s
    return f"\\textbf{{{s}}}" if np.isclose(mu, best_val) else s


def latex(rows, n_test):
    best_vals = {k: (max if hi else min)(rows[r][f"{k}_mean"] for r in rows) for k, _, hi in METRICS}
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{PCE prediction from device recipes on a temporally held-out split "
        r"(train: papers up to 2023; validation: 2024; test: 2025, $n=" + str(n_test) + r"$). "
        r"Mean$\pm$std over three seeds for stochastically trained models; prompted models use "
        r"greedy decoding. AUC uses the predicted PCE as the score for PCE $>" + f"{CLS_THRESHOLD:.0f}" +
        r"$\%. Invalid: unparseable answers, scored as the training mean. GPU-min: per run on one "
        r"A100-40GB. Best value per column in bold.}",
        r"\label{tab:pce_benchmark}",
        r"\resizebox{0.95\textwidth}{!}{%",
        r"\begin{tabular}{lllcccccc}", r"\toprule",
        r"\textbf{Model} & \textbf{Input} & \textbf{Params} & \textbf{MAE} $\downarrow$ & "
        r"\textbf{MAD:MAE} $\uparrow$ & $\boldsymbol{R^2}$ $\uparrow$ & \textbf{AUC} $\uparrow$ & "
        r"\textbf{Invalid (\%)} & \textbf{GPU-min} \\", r"\midrule",
    ]
    group = None
    for key, (name, inp, params, grp) in ROWS.items():
        if key not in rows:
            continue
        if group is not None and grp != group:
            lines.append(r"\midrule")
        group = grp
        r = rows[key]
        cells = [fmt(rows, key, k, best_vals[k]) for k, *_ in METRICS]
        inv = "--" if r["invalid_rate"] is None else f"{100 * r['invalid_rate']:.1f}"
        gpu = "0 (CPU)" if r["gpu_min_per_run"] == 0 else f"{r['gpu_min_per_run']:.1f}"
        lines.append(" & ".join([name, inp, params, *cells, inv, gpu]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines)


def markdown(rows, best, p_best):
    head = "| Model | Input | MAE ↓ | MAD:MAE ↑ | R² ↑ | AUC ↑ | Invalid % | GPU-min | p (best vs. this) |"
    out = [head, "|" + "---|" * 9]
    for key, (name, inp, *_rest) in ROWS.items():
        if key not in rows:
            continue
        r = rows[key]
        cell = lambda k: f"{r[k + '_mean']:.3f} ± {r[k + '_std']:.3f}"
        inv = "–" if r["invalid_rate"] is None else f"{100 * r['invalid_rate']:.1f}"
        p = "best" if key == best else (f"{p_best[key]:.4f}" if key in p_best else "–")
        out.append(f"| {name} | {inp} | {cell('mae')} | {cell('mad_mae')} | {cell('r2')} | {cell('auc')} | "
                   f"{inv} | {r['gpu_min_per_run']:.1f} | {p} |")
    return "\n".join(out)


def median_seed(runs, key):
    """The run whose test MAE is the median across seeds, so the figure matches a real run."""
    scored = sorted(runs[key], key=lambda r: regression_metrics(r[1]["pce"].to_numpy(),
                                                                r[1]["pred"].to_numpy())["mae"])
    meta, test = scored[len(scored) // 2]
    return meta["seed"], test["pce"].to_numpy(), test["pred"].to_numpy()


def _style(ax):
    ax.grid(color="#e4e3df", lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelcolor=INK, width=0.6, length=2.5)


def best_model_figure(runs, best, ref="lgbm18"):
    """One panel per metric for the best model, with the expert-descriptor baseline for reference."""
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
                         "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    seed, y, p = median_seed(runs, best)
    ref_seed, y_ref, p_ref = median_seed(runs, ref)
    assert np.array_equal(y, y_ref)
    mb, mr = regression_metrics(y, p), regression_metrics(y, p_ref)
    name, ref_name = ROWS[best][0].split(",")[0], ROWS[ref][0]

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.15), gridspec_kw={"wspace": 0.42})
    for ax in axes:
        _style(ax)

    # (a) MAE: cumulative distribution of absolute errors.
    ax = axes[0]
    dev = np.abs(y - y.mean())
    for err, color in ((np.abs(y - p), BLUE), (np.abs(y - p_ref), ORANGE), (dev, GRAY)):
        xs = np.sort(err)
        ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=color, lw=1.4,
                ls="--" if color == GRAY else "-")
    ax.set_xlim(0, 10)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Absolute error (PCE %)")
    ax.set_ylabel("Fraction of test devices")
    ax.set_title(f"(a) MAE = {mb['mae']:.2f}", loc="left")

    # (b) MAD:MAE: model errors vs. spread of the targets around their mean.
    ax = axes[1]
    bins = np.linspace(0, 15, 31)
    ax.hist(dev, bins=bins, color=GRAY, alpha=0.45, edgecolor="white", lw=0.4)
    ax.hist(np.abs(y - p), bins=bins, color=BLUE, alpha=0.75, edgecolor="white", lw=0.4)
    top = ax.get_ylim()[1]
    for v, txt, yfrac in ((dev.mean(), "MAD", 0.75), (mb["mae"], "MAE", 0.95)):
        ax.axvline(v, color=MUTED, lw=0.9, ls=":")
        ax.text(v + 0.2, top * yfrac, f"{txt}\n{v:.2f}", color=INK, fontsize=5.6, va="top")
    ax.set_xlabel("Absolute deviation (PCE %)")
    ax.set_ylabel("Devices")
    ax.set_title(f"(b) MAD:MAE = {mb['mad_mae']:.2f}", loc="left")

    # (c) R^2: parity.
    ax = axes[2]
    ax.plot([0, 30], [0, 30], color=MUTED, lw=0.8, ls="--")
    ax.scatter(p, y, s=4, color=BLUE, alpha=0.45, edgecolors="none")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_xlabel("Predicted PCE (%)")
    ax.set_ylabel("Reported PCE (%)")
    ax.set_title(f"(c) $R^2$ = {mb['r2']:.2f}", loc="left")

    # (d) AUC: ROC for PCE > threshold.
    ax = axes[3]
    lab = y > CLS_THRESHOLD
    for pred, color in ((p, BLUE), (p_ref, ORANGE)):
        fpr, tpr, _ = roc_curve(lab, pred)
        ax.plot(fpr, tpr, color=color, lw=1.4)
    ax.plot([0, 1], [0, 1], color=GRAY, lw=0.8, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"(d) AUC = {mb['auc']:.2f}", loc="left")

    handles = [plt.Line2D([], [], color=BLUE, lw=1.6, label=f"{name} (best model)"),
               plt.Line2D([], [], color=ORANGE, lw=1.6, label=ref_name),
               plt.Line2D([], [], color=GRAY, lw=1.6, ls="--", label="Deviation from test-set mean")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=6.5,
               labelcolor=INK, bbox_to_anchor=(0.5, 1.0), handlelength=2.2)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.19, top=0.80)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_best_model.{ext}", dpi=300)
    err = np.abs(y - p)
    return {"model": best, "seed": int(seed), "ref": ref, "ref_seed": int(ref_seed),
            "best": mb, "reference": mr,
            "share_abs_err_le_2": float((err <= 2).mean()),
            "ref_share_abs_err_le_2": float((np.abs(y - p_ref) <= 2).mean()),
            "mae_pce_le_15": float(err[y <= 15].mean()),
            "mae_pce_gt_15": float(err[y > 15].mean()),
            "share_pce_le_15": float((y <= 15).mean())}


def main():
    runs = load()
    rows, ens = summarize(runs)
    best, p_best, tab, p_lm_vs_tab = significance(rows, ens)
    fig_stats = best_model_figure(runs, best)
    (OUT / "metrics.json").write_text(json.dumps(
        {"rows": rows, "best": best, "p_best_vs": p_best, "best_tabular": tab,
         "p_finetuned_lm_vs_best_tabular": p_lm_vs_tab, "figure": fig_stats}, indent=2))
    (OUT / "table_main.tex").write_text(latex(rows, len(ens[best][0])))
    md = markdown(rows, best, p_best)
    (OUT / "table_main.md").write_text(md + "\n")
    print(md)
    print(json.dumps({"p_best_vs": p_best, "best_tabular": tab, "p_lm_vs_tab": p_lm_vs_tab,
                      "figure": fig_stats}, indent=1))


if __name__ == "__main__":
    main()
