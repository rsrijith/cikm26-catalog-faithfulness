"""Recall-oriented candidate retrieval, independent of every matcher under test.

The v1 validation drew the annotator's candidates from the same scorer being validated,
so when retrieval missed the true entry the item was silently recorded as a
hallucination and recall was measured conditional on the instrument's own blind spot
(Amazon recall came out at exactly 1.000). Retrieval here uses an IDF-weighted rare-token
inverted index, which shares no logic with score_pair or with any fuzzy scorer, and is
tuned for recall rather than precision: its only job is to avoid missing the true entry.

Candidates are then unioned with a fuzzy top-k so neither method's blind spot is fatal.
"""
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, "code")
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from surface_match import fold, sortkey

PROC = Path("data/processed")
CACHE = Path("data/analysis_corrected/retrieval_cache")
CACHE.mkdir(parents=True, exist_ok=True)


class CatalogIndex:
    def __init__(self, dataset: str):
        self.dataset = dataset
        cat = pd.read_parquet(PROC / dataset / "catalog.parquet").reset_index(drop=True)
        self.titles = cat["title"].astype(str).to_numpy()
        self.ids = cat["item_id"].astype(str).to_numpy()
        self.quart = cat["popularity_quartile"].astype(str).to_numpy()
        self.norm = [fold(t) for t in self.titles]
        self.sortkeys = [sortkey(n) for n in self.norm]
        df = defaultdict(int)
        toks = []
        for n in self.norm:
            ts = set(n.split())
            toks.append(ts)
            for t in ts:
                df[t] += 1
        N = len(self.norm)
        self.idf = {t: math.log(N / (1 + c)) for t, c in df.items()}
        self.post = defaultdict(list)
        for i, ts in enumerate(toks):
            for t in ts:
                # skip ultra-common tokens; they add cost without discriminating
                if df[t] <= N * 0.02:
                    self.post[t].append(i)
        self.ntok = np.array([len(t) for t in toks])

    def retrieve(self, query: str, k: int = 12) -> list[int]:
        """Union of IDF-weighted token overlap and fuzzy top-k."""
        qn = fold(query)
        if not qn:
            return []
        qt = set(qn.split())
        scores = defaultdict(float)
        for t in qt:
            if t not in self.post:
                continue
            w = self.idf.get(t, 0.0)
            for i in self.post[t]:
                scores[i] += w
        top = sorted(scores.items(), key=lambda x: -x[1])[:k]
        out = [i for i, _ in top]
        # fuzzy supplement, so a query with no rare tokens still gets candidates
        res = process.extract(sortkey(qn), self.sortkeys, scorer=fuzz.ratio,
                              limit=k, processor=None)
        out += [i for _, _, i in res]
        seen, uniq = set(), []
        for i in out:
            if i not in seen:
                seen.add(i); uniq.append(i)
        # NOTE: returns up to 2k (IDF-top-k unioned with fuzzy-top-k). Callers that
        # need an exact budget must slice. Validating at one depth and deploying at
        # another is how the round-2 numbers were briefly wrong.
        return uniq[: k * 2]


def get_index(dataset: str) -> CatalogIndex:
    p = CACHE / f"{dataset}.pkl"
    if p.exists():
        with p.open("rb") as f:
            return pickle.load(f)
    idx = CatalogIndex(dataset)
    with p.open("wb") as f:
        pickle.dump(idx, f)
    return idx
