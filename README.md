# Harnessing Verified Literature for AI Scientists

Data and downstream-experiment code for the anonymous submission
*Harnessing Verified Literature for AI Scientists*.

AI scientists can automate discovery when experiments run as code, but in experimental
sciences the evidence they would reason over sits in unstructured papers. Before an agent can
use it, the literature has to become data whose reliability is measured. The paper argues that
this reliability is decided by the **harness** around the language model: the schema that fixes
the output contract, the agent roles, the verification stages, and the reliability signals that
travel with the data.

This repository releases what the harness produced and what an AI scientist can do with it:

- **The verified records** ([`dataset/`](dataset/)): one record per paper for more than ten
  thousand experimental perovskite solar cell papers, each describing the paper's best-performing
  device in a fixed schema, together with the judged reliability of every field category.
- **Reliability-scoped discovery** ([`discovery/`](discovery/), [`scripts/`](scripts/)): code that
  consumes the records the way the paper recommends (trusting reliable categories, excluding the
  least reliable one) and tests whether they carry knowledge that transfers to devices reported
  after the data an agent learns from.

## The harness

| Stage | Role |
|---|---|
| Schema as contract | Fixed field names, value conventions (units, ordered layer stacks), and explicit `Null` semantics. The same definitions serve as the evaluation rubric. |
| Annotator agent | Reads the full paper and supporting information, identifies the best-performing device, and emits a schema-conforming record. |
| Field-wise judge agents | Independently verify every field against the source paper; two judges of different capability, whose disagreement is part of the reliability estimate. |
| Reliability signal | Category-level accuracy ships with the data, so downstream consumers know which fields to automate on. |

The extraction itself ran on commercial API models whose versions change over time, so the
released records, not a re-run of the harness, are the reference artifact. Judged accuracy by
field category is listed in the [data card](dataset/README.md#reliability): near ceiling for
local, lexically anchored fields (reference data, cell definition, fabrication, J–V metrics),
lower for procedural synthesis fields, and lowest for stability.

## Reliability-scoped discovery

An AI scientist that learns from literature-derived data is only useful if what it learns
transfers to devices it has not seen. The experiment here asks exactly that, with a temporal
hold-out:

- **Learn** from records of papers published up to 2023 (5,422 devices); 2024 papers (2,170)
  are used only for early stopping and model selection.
- **Anticipate** the reported power conversion efficiency of devices from 2025 papers (858),
  given only their recipes: the 71 fields describing cell definition, device stack, synthesis,
  and fabrication.
- **Respect the reliability signal**: stability fields, the least reliable category, are never
  used. Every J–V field, every stability field, and all bibliographic fields are also kept out of
  the inputs, because efficiency follows from Voc, Jsc, and FF, and several hundred titles state it.

Three kinds of agents-in-the-loop consume the same records: tabular learners, language models
fine-tuned on a deterministic text rendering of the recipe, and a general-purpose LLM used
through frozen embeddings or prompted with zero or five retrieved examples (Llama-3.1, whose
training data end before the 2025 papers appeared).

| Model | Input | MAE ↓ | MAD:MAE ↑ | R² ↑ | AUC ↑ | Invalid % |
|---|---|---|---|---|---|---|
| Training-set mean | – | 5.69 | 0.80 | −0.22 | 0.500 | – |
| LightGBM, expert descriptors | tabular | 2.87 ± 0.01 | 1.59 ± 0.01 | 0.60 ± 0.00 | 0.904 ± 0.001 | – |
| LightGBM, all recipe fields | tabular | 2.48 ± 0.01 | 1.84 ± 0.00 | 0.69 ± 0.00 | 0.930 ± 0.001 | – |
| TabPFN v2, all recipe fields | tabular | 2.49 ± 0.01 | 1.84 ± 0.00 | 0.70 ± 0.00 | 0.932 ± 0.001 | – |
| MatSciBERT, fine-tuned | text | 2.24 ± 0.09 | 2.04 ± 0.08 | **0.71 ± 0.03** | **0.935 ± 0.002** | – |
| T5-base encoder, fine-tuned | text | **2.20 ± 0.06** | **2.08 ± 0.05** | 0.68 ± 0.02 | 0.934 ± 0.002 | – |
| Llama-3.1-8B embeddings + ridge | text | 2.70 | 1.69 | 0.63 | 0.912 | – |
| Llama-3.1-8B-Instruct, 0-shot | text | 4.10 | 1.12 | −0.02 | 0.642 | 3.6 |
| Llama-3.1-8B-Instruct, 5-shot | text | 2.82 | 1.62 | 0.50 | 0.867 | 4.7 |

Mean ± std over three seeds where training is stochastic. AUC scores the predicted efficiency for
devices above 20%. Full numbers, per-run predictions, and raw LLM generations are in
[`results/`](results/).

What this shows for an AI scientist:

- **The verified records transfer.** Every trained model clearly beats the mean on devices from
  a later year, even though reported efficiencies drift upward over time.
- **The full schema is worth extracting.** Using all recipe fields beats a hand-picked set of
  expert descriptors, and language models that read the record as text, with no feature
  engineering, do best.
- **A general-purpose LLM does not replace learning from the data.** Zero-shot prompting is no
  better than the mean; the five-shot gain comes from the retrieved records themselves.

## Reproduce

Tested with Python 3.11, torch 2.10 (CUDA 12.8), transformers 4.57, vLLM 0.19, scikit-learn 1.8,
LightGBM 4.7, and TabPFN 2.1.3 on A100-40GB GPUs (see `requirements.txt`). Llama-3.1-8B-Instruct
is a gated model on Hugging Face and requires accepting its license.

```bash
export HARNESS_ENV=/path/to/env       # optional: venv or conda env to activate
source scripts/env.sh
python scripts/prepare.py             # reads dataset/, writes data/processed/*
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
allocated node (`SLURM_JOBID`, or `HARNESS_JOBID` from outside the allocation). Set
`HARNESS_RECORDS_URL` to read the records from another location.

Protocol details: the target is `JV_measured_reverse_scan_PCE` for records with 0 < value ≤ 30%;
text inputs are a deterministic `field: value` rendering of the non-null recipe fields, built in
`discovery/data.py:record_text`, with no LLM involved; a prompted answer counts as valid only if it
is a bare number or ends in "… to be X%", and invalid answers are scored as the training mean.
Raw generations are saved, so `python scripts/run_llm_prompt.py --rescore` re-applies the parser
without a GPU. Significance uses a paired bootstrap (10,000 resamples) on seed-averaged
predictions.

## Layout

```
dataset/               verified records (Parquet + gzipped JSONL) and their data card
discovery/data.py      reliability-scoped loading: target, temporal split, leakage exclusions, features, text
discovery/metrics.py   MAE / MAD:MAE / R2 / AUC, bootstrap CI, paired bootstrap
discovery/io.py        prediction + metadata files shared by every model
scripts/               one script per model family, plus prepare / report
results/               predictions/ (per run), table_main.*, metrics.json, fig_best_model.*
```

## License

The records are released under the MIT license. They contain structured factual values with
provenance to each source DOI, not article text.
