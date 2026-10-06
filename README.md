# Predicting perovskite solar cell efficiency from literature-extracted recipes

Code for the held-out prediction benchmark in *Harnessing Verified Literature for AI Scientists*
(anonymous submission). Models predict the power conversion efficiency (PCE) of a device from
the recipe fields of the released device records
([anonymous dataset mirror](https://huggingface.co/datasets/ANON_HF_NAMESPACE/psc-device-records)),
training on papers up to 2023 and testing on papers from 2025. The benchmark compares
tabular models, fine-tuned language models, and prompted LLMs. No literature extraction is
re-run; everything uses the released records.

## Task and protocol

| | |
|---|---|
| Target | `JV_measured_reverse_scan_PCE` (%), records with 0 < PCE <= 30 |
| Classification | PCE > 20%, scored by the predicted PCE (AUC); no separate classifier |
| Split | train <= 2023 (5,422), val 2024 (2,170), **test 2025 (858)**; val is used only for early stopping / model selection |
| Inputs | 71 recipe fields (cell definition, stack, synthesis, fabrication). Excluded: `Ref_*` (titles often state the PCE), all `JV_*` (PCE = Voc·Jsc·FF/P_in; forward/stabilized/certified PCE), all `Stability*`, provenance columns |
| Metrics | MAE, MAD:MAE (LLM4Mat-Bench convention; MAD of the test targets), R², AUC, invalid-output rate for prompted LLMs, GPU-minutes per run |
| Seeds | 3 for stochastic training; prompted LLMs use greedy decoding (1 run) |
| Significance | paired bootstrap (10,000 resamples) on seed-averaged predictions |

## Models

| Key | Model | Input |
|---|---|---|
| `mean` | training-set mean | – |
| `lgbm18` | LightGBM on 18 expert descriptors: A-site FA/MA/Cs, B-site Pb/Sn, X-site I/Br/Cl presence flags; bandgap; absorber thickness; cell area; has perovskite additive / surface treatment / ETL additive; substrate (ITO/FTO/other); ETL deposition (spin/CBD/PVD-ALD/other); HTL atmosphere (N2/air/other); sequential spin-coating | tabular |
| `lgbmall` | LightGBM on all recipe fields, auto-parsed (numeric if >=60% of train values hold one number, else train values with count >= 10 as categories) + composition flags | tabular |
| `tabpfn` | TabPFN v2 (2.1.3) on the same all-field table | tabular |
| `matscibert` | MatSciBERT + mean-pool + linear head, fine-tuned (L1 loss, lr 5e-5, 10 epochs, bs 16, 512 tokens) | text |
| `t5enc` | T5-base encoder + mean-pool + linear head (LLM-Prop style; lr 1e-4, 10 epochs, bs 16, 1024 tokens) | text |
| `llama8b_embed` | frozen Llama-3.1-8B-Instruct mean-pooled hidden states (layer 16 or 32) + ridge; layer and alpha chosen on val | text |
| `llama8b_{0,5}shot` | Llama-3.1-8B-Instruct, greedy, 48 new tokens; 5-shot uses the 5 most TF-IDF-similar training records | text |

Text input is a deterministic `field: value` serialization of the non-null recipe
fields (absorber first), built in `pscbench/data.py:record_text`; no LLM is involved.
Llama-3.1's knowledge cutoff (Dec 2023) predates the 2025 test papers.

A prompted answer is valid only if it is a bare number or ends in "... be/is/of [about] X%".
Invalid answers are replaced by the training mean and reported in the table. Raw
generations are saved, so the parser can be changed and re-applied with
`python scripts/run_llm_prompt.py --rescore`, with no GPU needed.

## Reproduce

Tested with Python 3.11, torch 2.10 (CUDA 12.8), transformers 4.57, vLLM 0.19, scikit-learn 1.8,
LightGBM 4.7, and TabPFN 2.1.3 on A100-40GB GPUs (see `requirements.txt`). Llama-3.1-8B-Instruct
is a gated model on Hugging Face and requires accepting its license.

```bash
export PSC_ENV=/path/to/env           # optional: venv or conda env to activate
source scripts/env.sh
python scripts/prepare.py             # downloads the records, writes data/processed/*
python scripts/run_tabular.py --model mean
python scripts/run_tabular.py --model lgbm18
python scripts/run_tabular.py --model lgbmall
python scripts/run_tabular.py --model tabpfn            # GPU
python scripts/run_encoder.py --model matscibert        # GPU
python scripts/run_encoder.py --model t5enc --seeds 0   # GPU; seeds can run on separate GPUs
python scripts/run_llm_embed.py                         # GPU
python scripts/run_llm_prompt.py                        # GPU (vLLM)
python scripts/make_report.py         # results/table_main.{tex,md}, metrics.json, fig_best_model.{pdf,png}
```

On a SLURM allocation, `scripts/on_gpu.sh <gpu> <command>` runs a command on one GPU of the
allocated node (`SLURM_JOBID`, or `PSC_JOBID` from outside the allocation). Set `PSC_DATA_URL`
to read the records from another location.

## Layout

```
pscbench/data.py      loading, target, split, leakage exclusions, features, text
pscbench/metrics.py   MAE / MAD:MAE / R2 / AUC, bootstrap CI, paired bootstrap
pscbench/io.py        prediction + metadata files shared by every model
scripts/             one script per model family, plus prepare / report
results/             predictions/ (per run), table_main.*, metrics.json, fig_best_model.*
```
