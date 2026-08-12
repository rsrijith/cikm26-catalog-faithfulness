"""Test whether any published OUT verdict was fabricated by a failed or partly-parsed batch.

The deployment pass wrote `in_catalog: false` both when the judge considered an item and
chose no candidate, and when no answer came back for that item at all. `parse_reply` drops
any line it cannot parse without reporting it, so a partial answer produced silent OUTs
with nothing in the logs. `llm_match_all.py` now records a `judged` flag, but the published
verdicts predate that and the distinction cannot be recovered from them.

Offline tests do not settle it. Aligned all-OUT runs of ten are enriched at batch
boundaries, but heavy title deduplication means index position is only loosely tied to
user position, and per-user clusters of genuine hallucinations produce the same signature.
So measure it directly: re-judge and compare flip rates.

  arm A   every record inside an aligned all-OUT run of ten, the batch-failure signature
  arm B   a random sample of OUT records outside those runs, which catches partial parses
  arm C   a random sample of IN records, the nondeterminism baseline

If arm A and arm B flip to IN at arm C's rate, no verdicts were fabricated. If either
exceeds it materially, the affected verdicts are missing data and the published rates need
recomputing without them.

Needs ANTHROPIC_API_KEY. Roughly 150 cached-prefix calls, well under a dollar.

Run: ANTHROPIC_API_KEY=... code/.venv/bin/python code/verify_verdicts.py
"""
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, "code")
import numpy as np
import pandas as pd
from llm_matcher import judge, DEPLOY_TOPN

OUT = Path("data/analysis_corrected")
DATASETS = ["movielens", "yelp", "amazon"]
LLMS = ["mistral", "llama", "gpt-oss", "claude"]
B = 10
N_PER_ARM = 500
RNG = np.random.default_rng(20260812)


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set")
    rows = []
    for d in DATASETS:
        for l in LLMS:
            recs = [json.loads(x) for x in (OUT / d / l / "llm_match.jsonl").open()]
            v = [r["in_catalog"] for r in recs]
            inblock = set()
            for s in range(0, len(v) - B + 1, B):
                if not any(v[s:s + B]):
                    inblock.update(range(s, s + B))
            for i, r in enumerate(recs):
                arm = ("A" if i in inblock else "B") if not r["in_catalog"] else "C"
                rows.append({"catalog": d, "model": l, "generated_title": r["title"],
                             "was": bool(r["in_catalog"]), "arm": arm})
    D = pd.DataFrame(rows)
    print("population: " + "  ".join(f"{a}={int((D.arm == a).sum())}" for a in "ABC"))

    samp = pd.concat([g if len(g) <= N_PER_ARM else g.sample(N_PER_ARM, random_state=7)
                      for _, g in D.groupby("arm")], ignore_index=True)
    print(f"re-judging {len(samp)} records with the current code\n")
    # topn reproduces the slice llm_match_all.py applies. Re-judging at a
    # different depth measures depth, not fabrication: a first run of this script
    # used the full ~20 and inflated OUT->IN flips to 19% against a 9.6% baseline.
    out = judge(samp.to_dict("records"), k=12, topn=DEPLOY_TOPN, verbose=True)
    # Partial parses happen at roughly a tenth of batches even on a healthy run, so
    # retry the unanswered items rather than discard the pass. Only give up if some
    # record still has no answer after several rounds: an unjudged record is missing
    # data, and counting it as a non-match is the defect this script exists to measure.
    # A batch is ten consecutive records, so an unanswered record's index // 10 names
    # the batch that dropped it.
    first_pass_unanswered = sum(1 for r in out if not r.get("judged"))
    partial_batches = len({i // 10 for i, r in enumerate(out) if not r.get("judged")})
    for attempt in range(4):
        miss = [r for r in out if not r.get("judged")]
        if not miss:
            break
        print(f"  retry {attempt + 1}: {len(miss)} records unanswered", flush=True)
        judge(miss, k=12, topn=DEPLOY_TOPN, verbose=False)
    R = pd.DataFrame(out)
    unjudged = int((~R.judged.astype(bool)).sum())
    if unjudged:
        by_arm = R[~R.judged.astype(bool)].arm.value_counts().to_dict()
        sys.exit(f"ABORT: {unjudged}/{len(R)} records still unjudged after retries "
                 f"{by_arm}. The comparison is invalid; do not report it.")
    print(f"  all {len(R)} records answered")
    R["flipped"] = R.llm_in_catalog.astype(bool) != R.was.astype(bool)

    print("\narm  n     was      flipped   rate")
    res = {}
    for a in "ABC":
        s = R[R.arm == a]
        res[a] = float(s.flipped.mean()) if len(s) else float("nan")
        lab = "OUT (in an aligned all-OUT run)" if a == "A" else (
              "OUT (elsewhere)" if a == "B" else "IN (baseline)")
        print(f" {a}  {len(s):4d}  {lab:32s} {int(s.flipped.sum()):4d}  {res[a]:.3%}")

    print()
    for a, lab in (("A", "batch-failure signature"), ("B", "partial-parse signature")):
        excess = res[a] - res["C"]
        verdict = ("no evidence of fabricated verdicts" if excess < 0.02
                   else "EXCESS FLIPS: treat these verdicts as missing data and recompute")
        print(f"  arm {a} ({lab}): {res[a]:.2%} against baseline {res['C']:.2%}, "
              f"excess {excess:+.2%} -> {verdict}")
    npop = sum(int((D.arm == a).sum()) for a in "ABC")
    unans_pct = first_pass_unanswered / len(R) * 100
    (OUT / "verify_verdicts.json").write_text(json.dumps(
        {"rates": res, "n": {a: int((R.arm == a).sum()) for a in "ABC"},
         "population": {a: int((D.arm == a).sum()) for a in "ABC"},
         "judged_total": int(len(R)),
         "unanswered_pct": round(unans_pct, 3),
         "partial_batches": partial_batches,
         "total_batches": int(-(-len(R) // 10)),
         "affected_estimate": int(round(npop * unans_pct / 100))}, indent=1))
    print(f"\nwrote {OUT / 'verify_verdicts.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
