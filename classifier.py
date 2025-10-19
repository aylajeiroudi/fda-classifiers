#!/usr/bin/env python3
"""
classifier.py — TF‑IDF (dot‑product) FDA Product Code Classifier

Local, no‑API baseline. Builds a TF‑IDF model from your product‑code catalog
and returns the most similar codes for a text query.

Expected CSV columns (case‑insensitive):
- product_code
- device_name
- definition

Commands:
- build         Build TF‑IDF model artifacts
- predict       Predict top‑k codes for a single query
- predict-batch Predict many queries from a CSV

Usage examples:
  python classifier.py build --csv "PPC Classification.csv" --model_dir ./model
  python classifier.py predict --model_dir ./model --query "AFib via PPG" --top_k 5
  python classifier.py predict-batch --model_dir ./model --in_csv examples/queries.csv --out_csv results.csv --append
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _normalize_text(s: str) -> str:
    return " ".join(str(s).lower().split()) if pd.notnull(s) else ""

def _read_catalog(csv_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    colmap = {c.lower(): c for c in df.columns}
    required = ["product_code", "device_name", "definition"]
    missing = [c for c in required if c not in colmap]
    if missing:
        raise ValueError(f"CSV missing required column(s): {', '.join(missing)}")
    df = df[[colmap[c] for c in required]].copy()
    df = df.drop_duplicates("product_code").reset_index(drop=True)
    df["document"] = (
        df[colmap["device_name"]].apply(_normalize_text)
        + " "
        + df[colmap["definition"]].apply(_normalize_text)
    )
    return df

def build(csv_path: pathlib.Path, model_dir: pathlib.Path) -> None:
    print(f"[+] Loading CSV from {csv_path} …")
    df = _read_catalog(csv_path)

    print(f"[+] Building TF‑IDF for {len(df)} product codes …")
    vectorizer = TfidfVectorizer(stop_words="english", min_df=2)
    matrix = vectorizer.fit_transform(df["document"])

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, model_dir / "vectorizer.joblib")
    joblib.dump(matrix, model_dir / "tfidf_matrix.joblib")
    df[["product_code", "device_name", "definition"]].to_csv(model_dir / "metadata.csv", index=False)
    print(f"[✓] Saved artifacts to {model_dir.resolve()}")

def _load(model_dir: pathlib.Path):
    try:
        vectorizer = joblib.load(model_dir / "vectorizer.joblib")
        matrix = joblib.load(model_dir / "tfidf_matrix.joblib")
        meta = pd.read_csv(model_dir / "metadata.csv")
    except FileNotFoundError as e:
        raise SystemExit(f"[!] Model not found in {model_dir}. Run 'build' first.") from e
    return vectorizer, matrix, meta

def predict(query: str, model_dir: pathlib.Path, top_k: int = 5) -> List[Tuple[str, str, float]]:
    vectorizer, matrix, meta = _load(model_dir)
    vec = vectorizer.transform([_normalize_text(query)])
    sims = cosine_similarity(vec, matrix).flatten()
    idx = sims.argsort()[::-1][:top_k]
    return [(meta.iloc[i]["product_code"], meta.iloc[i]["device_name"], float(sims[i])) for i in idx]

def predict_batch(in_csv: pathlib.Path, query_column: str, model_dir: pathlib.Path, out_csv: str | None, append: bool, top_k: int = 5) -> None:
    vectorizer, matrix, meta = _load(model_dir)
    dfq = pd.read_csv(in_csv)
    if query_column not in dfq.columns:
        raise SystemExit(f"[!] Column '{query_column}' not found in {in_csv}.")

    rows = []
    for q in dfq[query_column].astype(str).tolist():
        vec = vectorizer.transform([_normalize_text(q)])
        sims = cosine_similarity(vec, matrix).flatten()
        idx = sims.argsort()[::-1][:top_k]
        for rank, i in enumerate(idx, 1):
            rows.append({
                "query": q,
                "rank": rank,
                "product_code": meta.iloc[i]["product_code"],
                "device_name": meta.iloc[i]["device_name"],
                "score": float(sims[i]),
            })

    out_df = pd.DataFrame(rows)
    if out_csv:
        if append and pathlib.Path(out_csv).exists():
            out_df.to_csv(out_csv, mode="a", header=False, index=False)
            print(f"[✓] Appended {len(out_df)} rows to {out_csv}")
        else:
            out_df.to_csv(out_csv, index=False)
            print(f"[✓] Wrote {len(out_df)} rows to {out_csv}")
    else:
        # print to stdout
        print(out_df.to_csv(index=False))

def _add_build(sp):
    p = sp.add_parser("build", help="Build TF‑IDF model")
    p.add_argument("--csv", type=pathlib.Path, required=True)
    p.add_argument("--model_dir", type=pathlib.Path, default=pathlib.Path("./model"))

def _add_predict(sp):
    p = sp.add_parser("predict", help="Single query predict")
    p.add_argument("--model_dir", type=pathlib.Path, default=pathlib.Path("./model"))
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--top_k", type=int, default=5)

def _add_predict_batch(sp):
    p = sp.add_parser("predict-batch", help="Batch classify from CSV")
    p.add_argument("--model_dir", type=pathlib.Path, default=pathlib.Path("./model"))
    p.add_argument("--in_csv", type=pathlib.Path, required=True)
    p.add_argument("--query_column", type=str, default="query")
    p.add_argument("--out_csv", type=str, default=None)
    p.add_argument("--append", action="store_true")
    p.add_argument("--top_k", type=int, default=5)

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description="TF‑IDF (dot‑product) FDA Product Code Classifier")
    sp = parser.add_subparsers(dest="cmd")
    _add_build(sp); _add_predict(sp); _add_predict_batch(sp)

    if not argv:
        parser.print_help(sys.stderr); sys.exit(1)
    args = parser.parse_args(argv)

    if args.cmd == "build":
        build(args.csv, args.model_dir)
    elif args.cmd == "predict":
        res = predict(args.query, args.model_dir, args.top_k)
        print("\nTop suggestions:\n----------------")
        for code, name, score in res:
            print(f"{code}\t{score:.3f}\t{name}")
    elif args.cmd == "predict-batch":
        predict_batch(args.in_csv, args.query_column, args.model_dir, args.out_csv, args.append, args.top_k)
    else:
        parser.print_help(sys.stderr); sys.exit(1)

if __name__ == "__main__":
    main()
