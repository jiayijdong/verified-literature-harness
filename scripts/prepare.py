"""Build row-aligned records/text, paper (18) features, and all-field features."""
import json

import pandas as pd

from discovery.data import (PROC_DIR, REPO, all_field_features, build_records, input_fields,
                          load_raw, paper_features, record_text)


def main():
    raw = load_raw()
    rec = build_records(raw)
    sub = raw.set_index("row_id").loc[rec["row_id"]].reset_index()
    rec["text"] = record_text(sub).to_numpy()

    tab18 = paper_features(sub)
    taball = all_field_features(sub, train_mask=(rec["split"] == "train").to_numpy())

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    rec.to_parquet(PROC_DIR / "records.parquet", index=False)
    tab18.to_parquet(PROC_DIR / "tab18.parquet", index=False)
    taball.to_parquet(PROC_DIR / "taball.parquet", index=False)

    stats = {
        "n_raw": len(raw),
        "n_with_target": len(rec),
        "n_input_fields": len(input_fields(raw)),
        "n_taball_features": taball.shape[1],
        "n_taball_numeric": int((taball.dtypes != "category").sum()),
        "split_sizes": rec["split"].value_counts().to_dict(),
        "share_pce_gt20": rec.groupby("split")["pce"].apply(lambda s: round((s > 20).mean(), 3)).to_dict(),
        "pce_mean": rec.groupby("split")["pce"].mean().round(2).to_dict(),
        "text_chars_median": int(rec["text"].str.len().median()),
    }
    (REPO / "results" / "data_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(rec["text"].iloc[0])


if __name__ == "__main__":
    main()
