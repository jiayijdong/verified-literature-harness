"""Zero-shot and retrieval k-shot PCE prediction with an instruct LLM (vLLM, greedy).

Llama-3.1 has a Dec-2023 knowledge cutoff, so the 2025 test papers postdate its
pretraining data. k-shot examples are the k most TF-IDF-similar *training*
(<=2023) records. Unparseable or out-of-range answers count as invalid and are
replaced by the training mean for scoring.
"""
import argparse
import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from discovery.data import load_processed, split_masks
from discovery.io import PRED_DIR, gpu_timer, save_predictions
from discovery.metrics import regression_metrics

MODELS = {"llama8b": "meta-llama/Llama-3.1-8B-Instruct"}
SYSTEM = ("You are an expert in perovskite solar cells. Given a device recipe extracted from a "
          "paper, predict the power conversion efficiency (PCE, %) of that champion device under "
          "a reverse J-V scan. Reply with a single number only.")
SHOT_CHARS = 2500                # keeps 5-shot prompts well inside the context window
# Valid answer = a bare number, or a reply that *ends* in "... (be|is|of) [about] X[%]".
# Anything else (reasoning cut off at max_tokens, quoting a bandgap or an example's
# PCE mid-argument) is invalid rather than guessed from the first number in sight.
BARE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%?\s*\.?\s*$")
FINAL = re.compile(r"\b(?:be|is|of|at)\s+(?:around|approximately|about|roughly|~)?\s*"
                   r"(\d+(?:\.\d+)?)\s*%?\s*\.?\s*$", re.IGNORECASE)


def build_prompts(rec, m, k):
    test_idx = np.flatnonzero(m["test"])
    train_idx = np.flatnonzero(m["train"])
    texts = rec["text"].to_numpy()
    pce = rec["pce"].to_numpy()
    if k:
        tfidf = TfidfVectorizer(sublinear_tf=True).fit(texts[train_idx])
        sims = linear_kernel(tfidf.transform(texts[test_idx]), tfidf.transform(texts[train_idx]))
        nbrs = train_idx[np.argsort(-sims, axis=1)[:, :k]]
    convs = []
    for n, i in enumerate(test_idx):
        shots = ""
        if k:
            shots = "Similar devices with their measured PCE:\n\n" + "\n\n".join(
                f"Example {e + 1}:\n{texts[j][:SHOT_CHARS]}\nPCE (%): {pce[j]:.2f}" for e, j in enumerate(nbrs[n][::-1]))
            shots += "\n\nNow predict for this device.\n\n"
        convs.append([{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": f"{shots}Device recipe:\n{texts[i]}\n\nPCE (%):"}])
    return convs


def parse(text):
    hit = BARE.match(text) or FINAL.search(text)
    v = float(hit.group(1)) if hit else np.nan
    return v if 0 < v <= 40 else np.nan


def score(name, rec, m, raw, meta):
    y = rec["pce"].to_numpy()
    vals = np.array([parse(t) for t in raw])
    meta["invalid_rate"] = float(np.isnan(vals).mean())
    meta["raw_examples"] = raw[:5]
    pred = np.where(np.isnan(vals), y[m["train"]].mean(), vals)   # invalid -> training mean
    save_predictions(name, 0, rec, {"test": pred}, meta)
    print(name, meta["invalid_rate"], regression_metrics(y[m["test"]], pred), flush=True)


def rescore(args, rec, m):
    """Re-apply parse() to saved raw generations; no GPU needed."""
    for k in args.shots:
        name = f"{args.model}_{k}shot"
        meta = json.loads((PRED_DIR / f"{name}__seed0.json").read_text())
        raw = pd.read_json(PRED_DIR / f"{name}__raw.jsonl", lines=True)["raw"].tolist()
        score(name, rec, m, raw, {k_: v for k_, v in meta.items() if k_ not in ("model", "seed")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama8b", choices=list(MODELS))
    ap.add_argument("--shots", type=int, nargs="+", default=[0, 5])
    ap.add_argument("--rescore", action="store_true", help="re-parse saved raw outputs only")
    args = ap.parse_args()
    rec = load_processed("records")
    m = split_masks(rec)
    if args.rescore:
        return rescore(args, rec, m)
    from vllm import LLM, SamplingParams

    llm = LLM(model=MODELS[args.model], dtype="bfloat16", max_model_len=12288,
              gpu_memory_utilization=0.85, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=48)   # room for "I predict ... to be X%"

    for k in args.shots:
        meta = {"input": "text", "hf_id": MODELS[args.model], "shots": k, "decoding": "greedy"}
        with gpu_timer(meta):
            outs = llm.chat(build_prompts(rec, m, k), params, use_tqdm=False)
        raw = [o.outputs[0].text for o in outs]
        name = f"{args.model}_{k}shot"
        pd.DataFrame({"row_id": rec.loc[m["test"], "row_id"].to_numpy(), "raw": raw}).to_json(
            PRED_DIR / f"{name}__raw.jsonl", orient="records", lines=True)
        score(name, rec, m, raw, meta)


if __name__ == "__main__":
    main()
