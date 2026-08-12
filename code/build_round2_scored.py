"""Build data/analysis_corrected/round2_scored.csv from committed inputs.

This file previously had no generator, which is how it came to carry an adopted-instrument
column that no reported number derived from. The 2026-08-12 handoff review found that
`llm8_in`, a separate re-judge of the 201 labeled rows, disagrees with the deployment
verdicts in `<catalog>/<llm>/llm_match.jsonl` on 24 of 201 rows. Every other number in the
paper comes from those deployment verdicts, so scoring Table 1 on anything else describes
an instrument the paper does not use.

The adopted-instrument column here is therefore `llm_dep`, joined directly from the
released verdict files. `llm8_in` is kept for the record and is not scored.

The likely cause of the disagreement is batch composition, not decoding noise: the
deployment pass batches ten titles from a single generator, while the ad-hoc re-judge
batched ten drawn from the mixed 201-row sample.

Inputs, all committed:
  label_round2_labeled.csv           the annotator's returns, 205 rows
  <catalog>/<llm>/llm_match.jsonl    the deployment verdicts
Output:
  round2_scored.csv

Run: code/.venv/bin/python code/build_round2_scored.py
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, "code")
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from surface_match import score_pair
from utils import normalize_title

OUT = Path("data/analysis_corrected")
DATASETS = ["movielens", "yelp", "amazon"]
LLMS = ["mistral", "llama", "gpt-oss", "claude"]
NCAND = 8

# Two labels were provably wrong and are corrected here rather than in the raw returns,
# so the annotator's file stays as it was submitted. Verified by exact catalog lookup.
LABEL_FIXES = {
    ("movielens", "The Witch (2015)"): "MovieLens item 140267, 1764 interactions, Q1",
    ("yelp", "Clearwater Beach"): "Yelp tQIw_BZzfjh8UC4Eqhd0XQ, 601 interactions, Q1",
}


def cands(r):
    return [str(r[f"cand_{i}"]) for i in range(1, NCAND + 1)
            if str(r[f"cand_{i}"]) not in ("", "nan")]


def main():
    L = pd.read_csv(OUT / "label_round2_labeled.csv")
    n_drawn = len(L)
    L = L[pd.to_numeric(L.same_item, errors="coerce").notna()].copy()
    L["same_item"] = L.same_item.astype(int)
    L["human_in"] = L.same_item > 0
    print(f"drawn {n_drawn}, decided {len(L)}, dropped {n_drawn - len(L)} undecided")

    for (cat, title), why in LABEL_FIXES.items():
        m = (L.catalog == cat) & (L.generated_title.astype(str) == title)
        if m.any() and not L.loc[m, "human_in"].all():
            L.loc[m, "human_in"] = True
            print(f"  label corrected: {cat} / {title} -> in catalog ({why})")

    # adopted instrument: the verdicts that produced every other number in the paper
    dep = {}
    for d in DATASETS:
        for l in LLMS:
            p = OUT / d / l / "llm_match.jsonl"
            if not p.exists():
                continue
            for raw in p.open():
                r = json.loads(raw)
                dep[(d, str(r["title"]))] = bool(r["in_catalog"])
    key = list(zip(L.catalog, L.generated_title.astype(str)))
    missing = [k for k in key if k not in dep]
    if missing:
        raise SystemExit(f"{len(missing)} labeled rows absent from the deployment "
                         f"verdicts, e.g. {missing[:3]}")
    L["llm_dep"] = [dep[k] for k in key]

    # rule-based instruments, scored on the same eight candidates the annotator saw
    tset, tsort, surf = [], [], []
    for _, r in L.iterrows():
        cs = cands(r)
        q = normalize_title(str(r.generated_title))
        tset.append(any(fuzz.token_set_ratio(q, normalize_title(c)) >= 90 for c in cs))
        tsort.append(any(fuzz.token_sort_ratio(q, normalize_title(c)) >= 90 for c in cs))
        surf.append(any((score_pair(str(r.generated_title), c, r.catalog) or 0) >= 90
                        for c in cs))
    L["tset_in"], L["tsort_in"], L["surf_in"] = tset, tsort, surf

    prev = OUT / "round2_scored.csv"
    if prev.exists():
        old = pd.read_csv(prev)
        for col in ("llm_in", "llm8_in"):
            if col in old.columns:
                m = dict(zip(zip(old.catalog, old.generated_title.astype(str)), old[col]))
                L[col] = [m.get(k, "") for k in key]
        for col in ("tset_in", "tsort_in", "surf_in"):
            m = dict(zip(zip(old.catalog, old.generated_title.astype(str)), old[col]))
            same = sum(bool(m.get(k)) == bool(v) for k, v in zip(key, L[col]))
            print(f"  {col}: reproduces {same}/{len(L)} of the previous file")

    cols = (["catalog", "model", "generated_title", "llm_says", "llm_pick"]
            + [f"cand_{i}" for i in range(1, NCAND + 1)]
            + ["same_item", "human_in", "llm_dep", "llm_in", "llm8_in",
               "tset_in", "tsort_in", "surf_in"])
    L[[c for c in cols if c in L.columns]].to_csv(prev, index=False)
    print(f"wrote {prev} ({len(L)} rows); adopted-instrument column is llm_dep")
    d = int((L.llm_dep.values != L.llm8_in.values.astype(bool)).sum())
    print(f"  llm_dep vs the retired llm8_in re-judge: {d} rows differ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
