"""Prep MovieLens-25M for the LLM-rec calibration audit.

Auto-downloads the zip from grouplens.org, extracts, builds the catalog,
samples audit + calibration users with chronological holdout.

Run from Papers/CIKM/code:
    python prep_movielens.py [--force]
"""
import argparse
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from config import (
    MIN_HISTORY,
    MOVIELENS_URL,
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

OUT_DIR = PROCESSED_DIR / "movielens"
RAW_ZIP = RAW_DIR / "ml-25m.zip"
RAW_EXTRACTED = RAW_DIR / "ml-25m"


def download() -> None:
    """Download and extract MovieLens-25M if not already on disk."""
    if (RAW_EXTRACTED / "movies.csv").exists():
        return
    print(f"Downloading MovieLens-25M from {MOVIELENS_URL}")
    resp = requests.get(MOVIELENS_URL, stream=True, timeout=600)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    RAW_ZIP.parent.mkdir(parents=True, exist_ok=True)
    with RAW_ZIP.open("wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="ml-25m.zip"
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            pbar.update(len(chunk))
    print(f"Extracting to {RAW_DIR}")
    with zipfile.ZipFile(RAW_ZIP) as zf:
        zf.extractall(RAW_DIR)


def build_catalog(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    movies = movies.rename(columns={"movieId": "item_id"}).copy()
    movies["item_id"] = movies["item_id"].astype(str)
    movies["normalized_title"] = movies["title"].map(normalize_title)

    counts = ratings.groupby("movieId").size().rename("interaction_count")
    counts.index = counts.index.astype(str)

    catalog = movies.merge(
        counts.to_frame(), how="left", left_on="item_id", right_index=True
    )
    catalog["interaction_count"] = catalog["interaction_count"].fillna(0).astype(int)
    catalog["popularity_quartile"] = assign_popularity_quartile(
        catalog["interaction_count"]
    )
    return catalog[
        ["item_id", "title", "normalized_title", "interaction_count", "popularity_quartile"]
    ]


def main(force: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not force and already_prepped(OUT_DIR):
        print(f"MovieLens already prepped at {OUT_DIR}; pass --force to re-run.")
        return

    download()

    print("Loading ratings + movies CSVs")
    ratings = pd.read_csv(RAW_EXTRACTED / "ratings.csv")
    movies = pd.read_csv(RAW_EXTRACTED / "movies.csv")
    print(f"Ratings: {len(ratings):,} | Movies: {len(movies):,}")

    print("Building catalog")
    catalog = build_catalog(ratings, movies)
    catalog.to_parquet(OUT_DIR / "catalog.parquet", index=False)
    print(f"Catalog: {len(catalog):,} items written to catalog.parquet")

    print("Sampling users")
    ratings["movieId"] = ratings["movieId"].astype(str)
    ratings["userId"] = ratings["userId"].astype(str)
    audit_df, calib_df = sample_users_with_holdout(
        ratings,
        user_col="userId",
        item_col="movieId",
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
            "dataset": "movielens-25m",
            "source": MOVIELENS_URL,
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
