"""Restore the withheld MovieLens and Yelp catalog titles from your own copy of those
datasets.

The MovieLens and Yelp licenses forbid redistributing their data, so this release refers
to their catalog entries by item_id instead of by title. Everything else is here in full.
Once you have built the three catalogs (see DATA_SOURCES.md), this script joins the ids
back to titles and writes a `rehydrated/` copy of every affected file.

    python rehydrate.py --catalogs /path/to/data/processed

`--catalogs` is the directory holding `<dataset>/catalog.parquet` for movielens, yelp and
amazon, exactly as `code/prep_*.py` write them. Amazon titles are already present and are
passed through unchanged.

Nothing is overwritten: the released files stay as they are and the restored ones land in
`rehydrated/`.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

WITHHELD = {"movielens", "yelp"}
NCAND = 8
HERE = Path(__file__).parent


def load(catalogs: Path) -> tuple:
    titles, quart = {}, {}
    for d in ("movielens", "yelp", "amazon"):
        p = catalogs / d / "catalog.parquet"
        if not p.exists():
            sys.exit(f"missing {p}\nBuild it first; see DATA_SOURCES.md.")
        cat = pd.read_parquet(p)
        ids = cat.item_id.astype(str)
        titles[d] = dict(zip(ids, cat.title.astype(str)))
        quart[d] = dict(zip(ids, cat.popularity_quartile.astype(str)))
    return titles, quart


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogs", required=True, type=Path)
    a = ap.parse_args()
    T, Q = load(a.catalogs)
    dest = HERE / "rehydrated"
    (dest / "verdicts").mkdir(parents=True, exist_ok=True)
    (dest / "labels").mkdir(parents=True, exist_ok=True)
    n = 0

    for src in sorted((HERE / "verdicts").glob("*.jsonl")):
        d = src.stem.split("_")[0]
        lines = []
        for raw in src.open():
            r = json.loads(raw)
            if r.get("match_title") is None and r.get("match_id"):
                r["match_title"] = T[d].get(str(r["match_id"]), "")
                r["match_quartile"] = Q[d].get(str(r["match_id"]), "")
                n += 1
            lines.append(json.dumps(r, ensure_ascii=False))
        (dest / "verdicts" / src.name).write_text("\n".join(lines) + "\n")

    for name in ("human_labels_201.csv", "human_labels_blind_60.csv"):
        src = HERE / "labels" / name
        if not src.exists():
            continue
        df = pd.read_csv(src, keep_default_na=False)
        for i in range(1, NCAND + 1):
            tcol, icol = f"cand_{i}", f"cand_{i}_id"
            if tcol not in df.columns or icol not in df.columns:
                continue
            fill = df.catalog.isin(WITHHELD) & df[tcol].eq("") & df[icol].ne("")
            df.loc[fill, tcol] = [T[c].get(str(k), "") for c, k in
                                  zip(df.catalog[fill], df[icol][fill])]
            n += int(fill.sum())
        df.to_csv(dest / "labels" / name, index=False)

    src = HERE / "labels" / "sampling_frame_1200.csv"
    if src.exists():
        df = pd.read_csv(src, keep_default_na=False)
        fill = (df.catalog.isin(WITHHELD) & df.llm_match_title.eq("")
                & df.llm_match_id.ne(""))
        df.loc[fill, "llm_match_title"] = [T[c].get(str(k), "") for c, k in
                                           zip(df.catalog[fill], df.llm_match_id[fill])]
        df.loc[fill, "llm_match_quartile"] = [Q[c].get(str(k), "") for c, k in
                                              zip(df.catalog[fill], df.llm_match_id[fill])]
        n += int(fill.sum())
        df.to_csv(dest / "labels" / src.name, index=False)

    print(f"restored {n} catalog titles -> {dest}")
    print("Those titles are MovieLens and Yelp data, under their licenses, not ours.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
