# FDA Product Code Classifier — TF‑IDF (Dot‑Product) Edition

This repo contains the **early, local‑only** classifier that maps a short device
description to likely **FDA Product Codes** using TF‑IDF and cosine similarity
(a normalized dot product). No API keys, no network calls.

---

## What it does
- Builds a TF‑IDF index from your product‑code CSV (`product_code`, `device_name`, `definition`).
- For a query string, computes cosine similarity between the query vector and each code’s text.
- Returns top‑k suggestions. Also supports **batch** classification from a CSV of queries.

## Quick start

```bash
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Build the model (one-time, local):
```bash
python classifier.py build --csv "PPC Classification.csv" --model_dir ./model
```

Single query:
```bash
python classifier.py predict --model_dir ./model   --query "AI software for detecting lung nodules on CT scans" --top_k 5
```

Batch file (CSV with a column named `query`):
```bash
# sample provided in examples/queries.csv
python classifier.py predict-batch --model_dir ./model   --in_csv examples/queries.csv --query_column query   --out_csv results.csv --append
```

## CSV schema

Your main catalog CSV must include these **columns** (case‑insensitive):
- `product_code` — three‑letter code (e.g., QIH)
- `device_name`  — device type name/designation
- `definition`   — device type definition/description

Additional columns are ignored in this edition.

## Notes

- Cosine similarity = normalized dot product of TF‑IDF vectors.
- This edition is fully local and reproducible — great as a baseline repo.
- For a more accurate semantic version and hybrid LLM re‑ranker, see your second repo.

## License
Choose a license (e.g., MIT) and put it in `LICENSE`.
