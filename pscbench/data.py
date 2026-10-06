"""Device-record loading, PCE target, temporal split, and the model inputs.

Every model in the benchmark sees only *recipe* fields (cell definition, device
stack, synthesis, fabrication). Identifier/provenance fields and every J-V and
stability field are excluded: PCE = Voc*Jsc*FF/P_in, forward/stabilized/certified
PCE are near-copies of the target, and ~600 titles state the efficiency.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
RAW_PATH = DATA_DIR / "raw" / "psc_device_records.parquet"
PROC_DIR = DATA_DIR / "processed"
# Released records (anonymous mirror for review); override with PSC_DATA_URL.
DATA_URL = os.environ.get("PSC_DATA_URL", "https://huggingface.co/datasets/ANON_HF_NAMESPACE/psc-device-records/"
                                          "resolve/main/psc_device_records.parquet")

TARGET_FIELD = "JV_measured_reverse_scan_PCE"
PCE_RANGE = (0.0, 30.0)          # single-junction devices; drops tandem/garbled values
CLS_THRESHOLD = 20.0             # PCE > 20% classification, as in the paper
LEAK_PREFIXES = ("Ref_", "_", "JV_", "Stabilized_", "Stability_")
SPLIT_YEARS = {"train": (0, 2023), "val": (2024, 2024), "test": (2025, 2025)}

# Text order: absorber first so that truncation at 512 tokens drops the least
# informative layers (contacts, encapsulation) first.
TEXT_PREFIX_ORDER = ("Cell_", "Perovskite_", "ETL_", "HTL_", "Back", "Beck", "Encapsulation_")

NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def load_raw() -> pd.DataFrame:
    if not RAW_PATH.exists():
        import urllib.request
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, RAW_PATH)
    df = pd.read_parquet(RAW_PATH)
    df = df.astype(object).astype(str)
    df = df.mask(df.isin(["Null", "None", "nan", ""]))
    return df.reset_index(drop=True).rename_axis("row_id").reset_index()


def input_fields(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "row_id" and not c.startswith(LEAK_PREFIXES)]


def first_number(s) -> float:
    if not isinstance(s, str):
        return np.nan
    m = NUM_RE.search(s.replace(",", ""))
    return float(m.group()) if m else np.nan


def build_records(df: pd.DataFrame) -> pd.DataFrame:
    """row_id, year, split, pce for every record with a usable target and year."""
    year = pd.to_numeric(df["Ref_publication_year"].str[:4], errors="coerce")
    pce = df[TARGET_FIELD].map(first_number)
    rec = pd.DataFrame({"row_id": df["row_id"], "year": year, "pce": pce})
    rec = rec[rec["year"].notna() & rec["pce"].between(*PCE_RANGE, inclusive="right")]
    rec["year"] = rec["year"].astype(int)
    rec["split"] = None
    for name, (lo, hi) in SPLIT_YEARS.items():
        rec.loc[rec["year"].between(lo, hi), "split"] = name
    return rec.dropna(subset=["split"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Paper Sec. 5.2 features (rebuilt from Fig. 3 / Fig. S3): 18 descriptors.
# ---------------------------------------------------------------------------
SITE_PATTERNS = {
    "A_FA": r"FA", "A_MA": r"MA", "A_Cs": r"Cs",
    "B_Pb": r"Pb", "B_Sn": r"Sn",
    "X_I": r"I(?![a-z])", "X_Br": r"Br", "X_Cl": r"Cl",
}


def composition_flags(comp: pd.Series) -> pd.DataFrame:
    out = {}
    for name, pat in SITE_PATTERNS.items():
        hit = comp.str.contains(pat, regex=True, na=False).astype(float)
        out[name] = hit.where(comp.notna())
    return pd.DataFrame(out, index=comp.index)


def _number_in(s: pd.Series, lo: float, hi: float, scale: dict[str, float] | None = None) -> pd.Series:
    def parse(v):
        x = first_number(v)
        if scale and isinstance(v, str):
            for unit, f in scale.items():
                if unit in v:
                    x *= f
                    break
        return x if lo <= x <= hi else np.nan
    return s.map(parse)


def _first_layer_category(s: pd.Series, rules: list[tuple[str, str]]) -> pd.Series:
    def cat(v):
        if not isinstance(v, str):
            return np.nan
        head = v.split("/")[0].lower()
        for pat, label in rules:
            if re.search(pat, head):
                return label
        return "Other"
    return s.map(cat)


def _substrate(stack):
    if not isinstance(stack, str):
        return np.nan
    layers = [x.strip().lower() for x in stack.split("/")[:2]]
    for layer in layers:
        if "fto" in layer:
            return "FTO"
        if "ito" in layer:
            return "ITO"
    return "Other"


def paper_features(df: pd.DataFrame) -> pd.DataFrame:
    has = lambda c: df[c].notna().astype(float)
    feats = composition_flags(df["Perovskite_composition_long_form"])
    feats["bandgap_eV"] = _number_in(df["Perovskite_bandgap"], 1.0, 3.5)
    feats["thickness_nm"] = _number_in(df["Perovskite_thickness"], 10, 5000, {"µm": 1e3, "μm": 1e3, "um": 1e3})
    feats["cell_area_cm2"] = _number_in(df["Cell_area_measured"], 1e-3, 1e3, {"mm": 1e-2})
    feats["perovskite_additive"] = has("Perovskite_additive_compound")
    feats["surface_treatment"] = has("Perovskite_surface_treatment_material")
    feats["etl_additive"] = has("ETL_additive_compound")
    feats["substrate"] = df["Cell_stack_sequence"].map(_substrate)
    feats["etl_deposition"] = _first_layer_category(df["ETL_deposition_method"], [
        (r"spin", "Spin"), (r"chemical bath|cbd", "CBD"),
        (r"evapor|sputter|atomic layer|ald|pvd|vapou?r", "PVD/ALD")])
    feats["htl_atmosphere"] = _first_layer_category(df["HTL_deposition_synthesis_atmosphere"], [
        (r"n2|nitrogen|glove", "N2"), (r"air", "Air")])
    seq = df["Perovskite_deposition_sequential_spin_coating"].str.lower()
    feats["sequential_spin"] = seq.map({"true": 1.0, "false": 0.0})
    for c in ("substrate", "etl_deposition", "htl_atmosphere"):
        feats[c] = feats[c].astype("category")
    assert feats.shape[1] == 18
    return feats


# ---------------------------------------------------------------------------
# Automatic featurization of all recipe fields (no hand-picking).
# ---------------------------------------------------------------------------
def _is_numeric_field(s: pd.Series, min_share: float = 0.6) -> bool:
    v = s.dropna()
    if len(v) == 0:
        return False
    n_nums = v.map(lambda x: len(NUM_RE.findall(x)))
    return (n_nums == 1).mean() >= min_share


def all_field_features(df: pd.DataFrame, train_mask: pd.Series, min_count: int = 10) -> pd.DataFrame:
    """Numeric fields -> first number; text fields -> frequent values (train) else 'other'."""
    cols = {}
    for c in input_fields(df):
        s = df[c]
        if s[train_mask].notna().sum() < min_count:   # e.g. an all-empty schema field
            continue
        if _is_numeric_field(s[train_mask]):
            cols[c] = s.map(first_number).astype(float)
        else:
            norm = s.str.strip().str.lower()
            counts = norm[train_mask].value_counts()
            keep = set(counts[counts >= min_count].index)
            cols[c] = norm.where(norm.isna() | norm.isin(keep), "other").astype("category")
    out = pd.DataFrame(cols, index=df.index)
    flags = composition_flags(df["Perovskite_composition_long_form"]).add_prefix("comp_")
    return pd.concat([out, flags], axis=1)


# ---------------------------------------------------------------------------
# Deterministic text serialization (no LLM involved).
# ---------------------------------------------------------------------------
def _ordered_text_fields(df: pd.DataFrame) -> list[str]:
    fields = input_fields(df)
    rank = lambda c: next((i for i, p in enumerate(TEXT_PREFIX_ORDER) if c.startswith(p)), len(TEXT_PREFIX_ORDER))
    return sorted(fields, key=lambda c: (rank(c), fields.index(c)))


def record_text(df: pd.DataFrame) -> pd.Series:
    fields = _ordered_text_fields(df)
    names = {c: c.replace("_", " ").replace("Beckcontact", "Backcontact") for c in fields}

    def render(row):
        return "\n".join(f"{names[c]}: {row[c]}" for c in fields if isinstance(row[c], str))
    return df.apply(render, axis=1)


def load_processed(name: str = "records") -> pd.DataFrame:
    """records (row_id, year, split, pce, text), tab18, or taball; all row-aligned."""
    return pd.read_parquet(PROC_DIR / f"{name}.parquet")


def split_masks(rec: pd.DataFrame) -> dict[str, np.ndarray]:
    return {s: (rec["split"] == s).to_numpy() for s in SPLIT_YEARS}
