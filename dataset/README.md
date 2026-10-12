# Verified device records

The output of the literature-to-data harness in *Harnessing Verified Literature for AI Scientists*
(anonymous submission): structured records of perovskite solar cell devices extracted from the
peer-reviewed literature by a schema-constrained annotator agent and verified field by field by
independent judge agents. Each category of fields ships with its judged reliability, so that AI
scientists consuming the records know which fields they can automate on.

## Files

| File | Format | Size |
|---|---|---|
| `device_records.parquet` | Parquet, 10,258 rows × 102 string columns | 3.6 MB |
| `device_records.jsonl.gz` | gzipped JSON Lines, same rows and columns | 4.2 MB (53 MB uncompressed) |

```python
import pandas as pd
df = pd.read_parquet("dataset/device_records.parquet")
```

## What a record is

One record per paper: the paper's **best-performing device**, as identified by the annotator
agent from the main text and supporting information. The corpus covers experimental perovskite solar cell
papers indexed in Web of Science; publication years are mostly 2021–2025 (2025 up to April), with a few
earlier papers and 13 records without a parsed year.

## Columns

All values are strings. Numbers keep their units (e.g. `"545 nm"`, `"23.87 mA/cm2"`), layer
stacks are ordered from substrate to top contact with `/` (e.g. `FTO/SnO2/perovskite/Spiro-OMeTAD/Au`),
multiple entries are separated by `;`, and a field with no support in the paper is the literal
string `"Null"`.

| Prefix | Columns | Content |
|---|---|---|
| `Ref_` | 4 | DOI, journal, publication year, title |
| `Cell_` | 3 | architecture (NIP/PIN), full stack, measured area |
| `ETL_` | 15 | electron-transport layer: stack, thickness, additives, deposition and annealing |
| `Perovskite_` | 34 | absorber: composition, dimension, bandgap, thickness, additives, deposition, annealing, surface treatment |
| `HTL_` | 14 | hole-transport layer: stack, thickness, additives, deposition and annealing |
| `Backcontact_` | 3 | back contact thickness, deposition method and pressure |
| `Encapsulation_` | 2 | encapsulation method and atmosphere |
| `JV_` | 15 | reverse/forward-scan Voc, Jsc, FF, PCE; illumination; scan settings; certified PCE |
| `Stabilized_` | 2 | stabilized measured and certified PCE |
| `Stability_` | 8 | stability-test conditions and PCE after a given time |
| `_doi_folder`, `_source_file` | 2 | provenance: DOI-based identifiers of the source extraction |

These are the 97 schema fields of the paper (Appendix A; `Beckcontact_deposition_method` there is
`Backcontact_deposition_method` here), plus three auxiliary columns that are empty or nearly so
(`JV_measured_inverse_scan_Voc`, `ETL_deposition_spin_coating_score`,
`Perovskite_deposition_final_solutions_concentration`), plus the two provenance columns.

## Reliability

Extraction quality was judged field by field by two LLM judges (gpt-5-mini and o3). Mean accuracy
by field category:

| Category | Accuracy (%) |
|---|---|
| Reference data | 99.50 |
| Cell definition | 93.33 |
| Device fabrication | 93.05 |
| J–V metrics | 93.12 |
| Device stack | 89.95 |
| Material synthesis | 85.31 |
| Stability metrics | 70.31 |
| **Overall** | **89.71** |

Errors are mostly incorrect values rather than omissions. Treat stability fields as exploratory,
and check synthesis fields before relying on them.

## Intended use and limitations

The records describe reported champion devices, not the full experimental distribution, and
reflect publication bias toward positive results. Values are LLM extractions and can be wrong;
consult the source DOI for any value that matters. Trends found in these records are associations,
not causal or experimentally validated conclusions.

## License

MIT. Only structured factual values with provenance to the source DOI are released, not article text.
