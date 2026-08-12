"""Shared utilities for dataset prep.

normalize_title: canonical-form for fuzzy item matching.
assign_popularity_quartile: Q1 (head) ... Q4 (tail) by interaction-count percentile.
sample_users_with_holdout: chronologically split a user's history, sample
    N_AUDIT + N_CALIB users with sufficient history.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

YEAR_SUFFIX_RE = re.compile(r"\s*\([12]\d{3}\)\s*$")
NON_WORD_RE = re.compile(r"[^\w\s]")


def normalize_title(title: object) -> str:
    """Canonicalize an item title for fuzzy matching.

    Lowercase, strip parenthetical year suffix (e.g. "Toy Story (1995)"),
    drop punctuation, collapse whitespace.
    """
    if not isinstance(title, str) or not title:
        return ""
    s = title.lower()
    s = YEAR_SUFFIX_RE.sub("", s)
    s = NON_WORD_RE.sub(" ", s)
    return " ".join(s.split())


def assign_popularity_quartile(counts: pd.Series) -> pd.Series:
    """Q1 (head, top 25% by count) ... Q4 (tail, bottom 25%).

    Items tied at zero counts all land in Q4. We rank descending so highest
    count gets rank 1 and falls in Q1.
    """
    ranks = counts.rank(method="first", ascending=False)
    quartiles = pd.qcut(ranks, q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    return quartiles.astype(str)


def sample_users_with_holdout(
    interactions: pd.DataFrame,
    *,
    user_col: str,
    item_col: str,
    timestamp_col: str,
    catalog: pd.DataFrame,
    min_history: int,
    n_audit: int,
    n_calib: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample audit + calibration users with chronological holdout.

    For each sampled user, the most recent interaction by timestamp becomes
    the target; everything before becomes the history.

    Returns (audit_df, calib_df) with columns:
        user_id, history (list[str]), target_item_id, target_quartile
    """
    user_counts = interactions.groupby(user_col).size()
    eligible = user_counts[user_counts >= min_history].index.to_numpy()

    if len(eligible) < n_audit + n_calib:
        raise ValueError(
            f"Only {len(eligible):,} users have >={min_history} interactions; "
            f"need {n_audit + n_calib:,}"
        )

    rng = np.random.default_rng(seed)
    sampled = rng.choice(eligible, size=n_audit + n_calib, replace=False)
    audit_set = set(sampled[:n_audit].tolist())
    calib_set = set(sampled[n_audit:].tolist())
    keep = audit_set | calib_set

    # Filter to sampled users, then sort once for cheap groupby
    sub = interactions[interactions[user_col].isin(keep)].copy()
    sub = sub.sort_values([user_col, timestamp_col]).reset_index(drop=True)

    quartile_map = catalog.set_index("item_id")["popularity_quartile"].to_dict()

    audit_records, calib_records = [], []
    for user_id, group in sub.groupby(user_col, sort=False):
        items = group[item_col].astype(str).tolist()
        if len(items) < 2:
            continue
        target = items[-1]
        history = items[:-1]
        record = {
            "user_id": str(user_id),
            "history": history,
            "target_item_id": target,
            "target_quartile": quartile_map.get(target, "Q4"),
        }
        if user_id in audit_set:
            audit_records.append(record)
        else:
            calib_records.append(record)

    return pd.DataFrame(audit_records), pd.DataFrame(calib_records)


def write_manifest(out_dir: Path, payload: dict) -> None:
    """Write a JSON manifest documenting the prep run."""
    payload = dict(payload)
    payload.setdefault("prepped_at", datetime.now(timezone.utc).isoformat())
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2, default=str))


def already_prepped(out_dir: Path) -> bool:
    """Whether a dataset's expected outputs all exist."""
    return all(
        (out_dir / f"{name}.parquet").exists()
        for name in ("catalog", "audit_users", "calib_users")
    )
