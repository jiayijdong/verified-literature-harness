"""Fine-tune a text encoder + linear head on recipe text (LLM-Prop-style regression).

MatSciBERT is capped at 512 tokens; the T5 encoder (relative positions) reads 1024.
Mean pooling, L1 loss on the standardized target, best epoch chosen on 2024 val.
"""
import argparse
import copy

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, get_linear_schedule_with_warmup

from discovery.data import load_processed, split_masks
from discovery.io import gpu_timer, save_predictions
from discovery.metrics import regression_metrics

MODELS = {
    "matscibert": dict(hf="m3rg-iitd/matscibert", max_len=512, lr=5e-5),
    "t5enc": dict(hf="google-t5/t5-base", max_len=1024, lr=1e-4),
}


class Regressor(nn.Module):
    def __init__(self, hf_id):
        super().__init__()
        cls = T5EncoderModel if "t5" in hf_id else AutoModel
        self.encoder = cls.from_pretrained(hf_id)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.encoder.config.hidden_size, 1))

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * mask).sum(1) / mask.sum(1)).squeeze(-1)


def batches(enc, y, idx, bs, shuffle, rng):
    idx = rng.permutation(idx) if shuffle else idx
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        yield (enc["input_ids"][j], enc["attention_mask"][j], torch.tensor(y[j], dtype=torch.float32))


@torch.no_grad()
def predict(model, enc, idx, bs=64):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(enc["input_ids"][j].cuda(), enc["attention_mask"][j].cuda()).float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()
    cfg = MODELS[args.model]

    rec = load_processed("records")
    m = split_masks(rec)
    idx = {s: np.flatnonzero(m[s]) for s in m}
    y = rec["pce"].to_numpy()
    mu, sd = y[m["train"]].mean(), y[m["train"]].std()
    z = (y - mu) / sd

    tok = AutoTokenizer.from_pretrained(cfg["hf"])
    enc = tok(rec["text"].tolist(), truncation=True, max_length=cfg["max_len"], padding="max_length",
              return_tensors="pt")
    n_trunc = int((enc["attention_mask"].sum(1) == cfg["max_len"]).sum())

    for seed in args.seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        meta = {"input": "text", "hf_id": cfg["hf"], "max_len": cfg["max_len"], "lr": cfg["lr"],
                "epochs": args.epochs, "batch_size": args.batch_size, "n_truncated": n_trunc}
        with gpu_timer(meta):
            model = Regressor(cfg["hf"]).cuda()
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
            steps = args.epochs * int(np.ceil(len(idx["train"]) / args.batch_size))
            sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
            best = (np.inf, None, -1)
            for epoch in range(args.epochs):
                model.train()
                for ids, att, t in batches(enc, z, idx["train"], args.batch_size, True, rng):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = nn.functional.l1_loss(model(ids.cuda(), att.cuda()).float(), t.cuda())
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()
                val_mae = np.abs(predict(model, enc, idx["val"]) * sd + mu - y[m["val"]]).mean()
                print(f"{args.model} seed={seed} epoch={epoch} val_mae={val_mae:.3f}", flush=True)
                if val_mae < best[0]:
                    best = (val_mae, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}), epoch)
            model.load_state_dict(best[1])
            preds = {s: predict(model, enc, idx[s]) * sd + mu for s in ("val", "test")}
        meta["best_epoch"] = best[2]
        meta["val"] = regression_metrics(y[m["val"]], preds["val"])
        save_predictions(args.model, seed, rec, preds, meta)
        print(args.model, seed, regression_metrics(y[m["test"]], preds["test"]), flush=True)
        del model, opt


if __name__ == "__main__":
    main()
