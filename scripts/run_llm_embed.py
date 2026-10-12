"""Frozen LLM embeddings of recipe text + ridge regression.

Mean-pooled hidden states from a middle and the last layer; the layer and the
ridge penalty are chosen on 2024 val. One forward pass over the corpus.
"""
import argparse

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from discovery.data import DATA_DIR, load_processed, split_masks
from discovery.io import gpu_timer, save_predictions
from discovery.metrics import regression_metrics

MODELS = {"llama8b": "meta-llama/Llama-3.1-8B-Instruct"}
ALPHAS = np.logspace(0, 5, 11)


@torch.no_grad()
def embed(hf_id, texts, layers, max_len=2048, bs=8):
    tok = AutoTokenizer.from_pretrained(hf_id)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "right"
    model = AutoModel.from_pretrained(hf_id, torch_dtype=torch.bfloat16).cuda().eval()
    order = np.argsort([len(t) for t in texts])          # length-bucketed batches
    out = {l: np.zeros((len(texts), model.config.hidden_size), np.float32) for l in layers}
    for i in range(0, len(order), bs):
        j = order[i:i + bs]
        enc = tok([texts[k] for k in j], truncation=True, max_length=max_len, padding=True,
                  return_tensors="pt").to("cuda")
        hs = model(**enc, output_hidden_states=True).hidden_states
        mask = enc["attention_mask"].unsqueeze(-1).float()
        for l in layers:
            out[l][j] = ((hs[l].float() * mask).sum(1) / mask.sum(1)).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama8b", choices=list(MODELS))
    args = ap.parse_args()
    hf_id = MODELS[args.model]

    rec = load_processed("records")
    m = split_masks(rec)
    y = rec["pce"].to_numpy()
    cache = DATA_DIR / "cache" / f"emb_{args.model}.npz"
    meta = {"input": "text", "hf_id": hf_id, "head": "ridge"}

    with gpu_timer(meta):
        if cache.exists():
            embs = dict(np.load(cache))
        else:
            embs = {str(k): v for k, v in embed(hf_id, rec["text"].tolist(), layers=[16, 32]).items()}
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, **embs)
        best = None
        for layer, E in embs.items():
            sc = StandardScaler().fit(E[m["train"]])
            Etr, Eva = sc.transform(E[m["train"]]), sc.transform(E[m["val"]])
            for a in ALPHAS:
                r = Ridge(alpha=a).fit(Etr, y[m["train"]])
                mae = np.abs(r.predict(Eva) - y[m["val"]]).mean()
                if best is None or mae < best[0]:
                    best = (mae, layer, a, r, sc)
        _, layer, alpha, ridge, sc = best
        E = embs[layer]
        preds = {s: ridge.predict(sc.transform(E[m[s]])) for s in ("val", "test")}

    meta.update(layer=int(layer), alpha=float(alpha), val=regression_metrics(y[m["val"]], preds["val"]))
    save_predictions(f"{args.model}_embed", 0, rec, preds, meta)
    print(args.model, layer, alpha, regression_metrics(y[m["test"]], preds["test"]), flush=True)


if __name__ == "__main__":
    main()
