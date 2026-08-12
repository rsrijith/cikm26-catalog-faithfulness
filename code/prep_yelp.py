"""Prep Yelp Open Dataset for the LLM-rec audit.

Requires manual download (Yelp EULA acceptance). Place the files
    yelp_academic_dataset_business.json
    yelp_academic_dataset_review.json
in `data/raw/yelp/` and then run this script.

Run from Papers/CIKM/code:
    python prep_yelp.py [--force]
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import (
    MIN_HISTORY,
    N_AUDIT_USERS,
    N_CALIB_USERS,
    PROCESSED_DIR,
    RAW_DIR,
    SEED,
    YELP_INSTRUCTIONS_URL,
)
from utils import (
    already_prepped,
    assign_popularity_quartile,
    normalize_title,
    sample_users_with_holdout,
    write_manifest,
)

OUT_DIR = PROCESSED_DIR / "yelp"

BUSINESS_FILE = "yelp_academic_dataset_business.json"
REVIEW_FILE = "yelp_academic_dataset_review.json"


def find_yelp_dir() -> Path:
    """Locate the directory containing the Yelp JSONL files.

    Searches under data/raw/ for `yelp_academic_dataset_business.json`.
    Handles the case where the user extracted the Yelp archive into a
    differently-named subdirectory (e.g., "Yelp JSON/yelp_dataset/").
    """
    matches = list(RAW_DIR.rglob(BUSINESS_FILE))
    if matches:
        return matches[0].parent
    # Fallback canonical location for the error message
    return RAW_DIR / "yelp"


def check_raw_files(yelp_dir: Path) -> None:
    """Fail with actionable instructions if Yelp raw files are missing."""
    missing = [
        f for f in (BUSINESS_FILE, REVIEW_FILE) if not (yelp_dir / f).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Yelp raw files missing under {RAW_DIR} (looked in {yelp_dir}): {missing}\n"
            f"Download manually from {YELP_INSTRUCTIONS_URL} (requires EULA),\n"
            f"extract the .tar archive, and place these JSON files anywhere "
            f"under data/raw/:\n"
            f"  - {BUSINESS_FILE}\n"
            f"  - {REVIEW_FILE}"
        )


def load_jsonl(path: Path, fields: list[str], desc: str) -> pd.DataFrame:
    """Load a line-delimited JSON file streaming, retaining only `fields`."""
    rows = []
    with path.open() as f:
        for line in tqdm(f, desc=desc):
            obj = json.loads(line)
            rows.append({k: obj.get(k) for k in fields})
    return pd.DataFrame(rows)


def main(force: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not force and already_prepped(OUT_DIR):
        print(f"Yelp already prepped at {OUT_DIR}; pass --force to re-run.")
        return

    yelp_dir = find_yelp_dir()
    check_raw_files(yelp_dir)
    print(f"Found Yelp files at {yelp_dir}")

    print("Loading businesses")
    businesses = load_jsonl(
        yelp_dir / BUSINESS_FILE,
        ["business_id", "name", "city", "review_count"],
        desc="businesses",
    )
    print(f"Businesses: {len(businesses):,}")

    print("Building catalog")
    catalog = businesses.rename(
        columns={"business_id": "item_id", "name": "title", "review_count": "interaction_count"}
    ).copy()
    catalog["item_id"] = catalog["item_id"].astype(str)
    catalog["normalized_title"] = catalog["title"].map(normalize_title)
    catalog["interaction_count"] = catalog["interaction_count"].fillna(0).astype(int)
    catalog["popularity_quartile"] = assign_popularity_quartile(
        catalog["interaction_count"]
    )
    catalog = catalog[
        ["item_id", "title", "normalized_title", "interaction_count", "popularity_quartile"]
    ]
    catalog.to_parquet(OUT_DIR / "catalog.parquet", index=False)
    print(f"Catalog: {len(catalog):,} businesses written to catalog.parquet")

    print("Loading reviews (large file; ~5-10 minutes first time)")
    reviews = load_jsonl(
        yelp_dir / REVIEW_FILE,
        ["user_id", "business_id", "date"],
        desc="reviews",
    )
    reviews = reviews.rename(columns={"business_id": "item_id"})
    reviews["user_id"] = reviews["user_id"].astype(str)
    reviews["item_id"] = reviews["item_id"].astype(str)
    reviews["timestamp"] = pd.to_datetime(reviews["date"]).astype("int64") // 10**9
    reviews = reviews[["user_id", "item_id", "timestamp"]]
    print(f"Reviews: {len(reviews):,}")

    print("Sampling users")
    audit_df, calib_df = sample_users_with_holdout(
        reviews,
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
            "dataset": "yelp-open-dataset",
            "source": YELP_INSTRUCTIONS_URL,
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
