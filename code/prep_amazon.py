"""Prep Amazon Reviews 2023 (Toys_and_Games subset) for the LLM-rec audit.

Downloads directly from McAuley Lab's UCSD-hosted gzipped JSONL files.
Bypasses HuggingFace `datasets` because v4.5+ no longer supports loading
scripts, and McAuley-Lab/Amazon-Reviews-2023 ships a loading script.

Run from Papers/CIKM/code:
    python prep_amazon.py [--force]
"""
import argparse
import gzip
import json
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from config import (
    AMAZON_CATEGORY,
    MIN_HISTORY,
    N_AUDIT_USERS,
    N_CALIB_USERS,
    PROCESSED_DIR,
    RAW_DIR,
    SEED,
)
from utils import (
    already_prepped,
    assign_popularity_quartile,
    normalize_title,
    sample_users_with_holdout,
    write_manifest,
)

OUT_DIR = PROCESSED_DIR / "amazon"
RAW_AMAZON = RAW_DIR / "amazon"

# McAuley Lab direct downloads (Amazon Reviews 2023)
# Reference: https://amazon-reviews-2023.github.io/
BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
REVIEW_URL = f"{BASE_URL}/review_categories/{AMAZON_CATEGORY}.jsonl.gz"
META_URL = f"{BASE_URL}/meta_categories/meta_{AMAZON_CATEGORY}.jsonl.gz"


def stream_download(url: str, out_path: Path, desc: str) -> None:
    """Download a file with a progress bar; skip if already on disk."""
    if out_path.exists():
        return
    print(f"Downloading {desc} from {url}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=900)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with out_path.open("wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            pbar.update(len(chunk))


def load_jsonl_gz(path: Path, fields: list[str], desc: str) -> pd.DataFrame:
    """Stream gzipped JSONL; retain only `fields` per record."""
    rows = []
    with gzip.open(path, "rt") as f:
        for line in tqdm(f, desc=desc, unit=" lines"):
            obj = json.loads(line)
            rows.append({k: obj.get(k) for k in fields})
    return pd.DataFrame(rows)


def main(force: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_AMAZON.mkdir(parents=True, exist_ok=True)

    if not force and already_prepped(OUT_DIR):
        print(f"Amazon already prepped at {OUT_DIR}; pass --force to re-run.")
        return

    meta_path = RAW_AMAZON / f"meta_{AMAZON_CATEGORY}.jsonl.gz"
    review_path = RAW_AMAZON / f"{AMAZON_CATEGORY}.jsonl.gz"

    stream_download(META_URL, meta_path, desc=f"meta_{AMAZON_CATEGORY}")
    stream_download(REVIEW_URL, review_path, desc=f"{AMAZON_CATEGORY} reviews")

    print("Loading item metadata")
    items_df = load_jsonl_gz(meta_path, ["parent_asin", "title"], desc="meta")
    items_df = items_df.rename(columns={"parent_asin": "item_id"})
    items_df["item_id"] = items_df["item_id"].astype(str)
    items_df["normalized_title"] = items_df["title"].map(normalize_title)
    items_df = items_df.drop_duplicates("item_id").reset_index(drop=True)
    print(f"Items: {len(items_df):,}")

    print("Loading reviews (this may take a few minutes)")
    reviews_df = load_jsonl_gz(
        review_path, ["user_id", "parent_asin", "timestamp"], desc="reviews"
    )
    reviews_df = reviews_df.rename(columns={"parent_asin": "item_id"})
    reviews_df["user_id"] = reviews_df["user_id"].astype(str)
    reviews_df["item_id"] = reviews_df["item_id"].astype(str)
    reviews_df["timestamp"] = pd.to_numeric(reviews_df["timestamp"], errors="coerce")
    reviews_df = reviews_df.dropna(subset=["timestamp"])
    print(f"Reviews: {len(reviews_df):,}")

    print("Building catalog with popularity")
    counts = reviews_df.groupby("item_id").size().rename("interaction_count")
    catalog = items_df.merge(counts.to_frame(), how="left", on="item_id")
    catalog["interaction_count"] = catalog["interaction_count"].fillna(0).astype(int)
    catalog["popularity_quartile"] = assign_popularity_quartile(
        catalog["interaction_count"]
    )
    catalog = catalog[
        ["item_id", "title", "normalized_title", "interaction_count", "popularity_quartile"]
    ]
    catalog.to_parquet(OUT_DIR / "catalog.parquet", index=False)
    print(f"Catalog: {len(catalog):,} items")

    print("Sampling users")
    audit_df, calib_df = sample_users_with_holdout(
        reviews_df,
        user_col="user_id",
        item_col="item_id",
        timestamp_col="timestamp",
        catalog=catalog,
        min_history=MIN_HISTORY,
        n_audit=N_AUDIT_USERS,
        n_calib=N_CALIB_USERS,
        seed=SEED,
    )
    audit_df.to_parquet(OUT_DIR / "audit_users.parquet", index=False)
    calib_df.to_parquet(OUT_DIR / "calib_users.parquet", index=False)
    print(f"Audit: {len(audit_df):,} | Calib: {len(calib_df):,}")

    write_manifest(
        OUT_DIR,
        {
            "dataset": f"amazon-reviews-2023/{AMAZON_CATEGORY}",
            "source_review": REVIEW_URL,
            "source_meta": META_URL,
            "category": AMAZON_CATEGORY,
            "seed": SEED,
            "min_history": MIN_HISTORY,
            "n_audit": N_AUDIT_USERS,
            "n_calib": N_CALIB_USERS,
            "catalog_size": int(len(catalog)),
        },
    )
    print(f"Done. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
