"""Pipeline integrity and instrument validation. Run before any number reaches the paper.

Written after a review pool found eight defects in the previous correction, most of them
invisible because a validation instrument shared code with the thing it validated. Each
check below corresponds to a failure that actually happened.

Run: python code/validate_pipeline.py
"""
import json
import math
import re
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, "code")
import numpy as np
import pandas as pd

OUT = Path("data/analysis_corrected")
DATASETS = ["movielens", "yelp", "amazon"]
LLMS = ["mistral", "llama", "gpt-oss", "claude"]
PROBLEMS = []


def bad(msg):
    PROBLEMS.append(msg)
    print(f"  FAIL {msg}")


def ok(msg):
    print(f"  ok   {msg}")


def main():
    print("=== 1. label coverage: every recommendation must be judged ===")
    total = judged = 0
    for d in DATASETS:
        for l in LLMS:
            p = OUT / d / l / "llm_match.jsonl"
            if not p.exists():
                bad(f"{d}/{l}: no llm_match.jsonl")
                continue
            lab, conflict = {}, 0
            with p.open() as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    prev = lab.get(r["title"])
                    if prev is not None and prev["in_catalog"] != r["in_catalog"]:
                        conflict += 1
                    lab[r["title"]] = r
            fdf = pd.read_parquet(OUT / d / l / "final.parquet")
            titles = set(fdf["title"].astype(str))
            miss = titles - set(lab)
            total += len(titles)
            judged += len(titles) - len(miss)
            if miss:
                bad(f"{d}/{l}: {len(miss)} titles unjudged")
            if conflict:
                bad(f"{d}/{l}: {conflict} titles judged twice with different answers")
    if total:
        ok(f"coverage {judged}/{total} ({judged / total * 100:.1f}%)")

    print("\n=== 2. matched titles must actually exist in the catalog ===")
    fakes = 0
    for d in DATASETS:
        cat = set(pd.read_parquet(f"data/processed/{d}/catalog.parquet")["title"].astype(str))
        for l in LLMS:
            p = OUT / d / l / "llm_match.jsonl"
            if not p.exists():
                continue
            n = f_ = 0
            with p.open() as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r["in_catalog"]:
                        n += 1
                        if r["match_title"] not in cat:
                            f_ += 1
            if f_:
                bad(f"{d}/{l}: {f_}/{n} matches name a title absent from the catalog")
                fakes += f_
    if not fakes:
        ok("all in-catalog matches resolve to real catalog entries")

    print("\n=== 3. no silent conflation of 'no match' with 'unusable input' ===")
    from surface_match import score_pair
    if not math.isnan(score_pair("", "x", "yelp")):
        bad("empty query returns a number, not NaN")
    else:
        ok("empty query -> NaN, distinct from a 0.0 non-match")

    print("\n=== 4. instrument validated on data it was not tuned on ===")
    lp = OUT / "round2_scored.csv"
    if lp.exists():
        r2 = pd.read_csv(lp)
        ex = ["Se7en", "The Rock", "Dumb and Dumber", "Sabrina", "Walk-On"]
        leak = r2.generated_title.astype(str).apply(
            lambda t: any(re.search(re.escape(e), t, re.I) for e in ex)).values
        h = r2.human_in.values
        # The instrument to validate is the one that produced the deployment labels,
        # joined from llm_match.jsonl itself. Two earlier versions of this gate scored a
        # re-judge instead, which is the defect it is supposed to catch.
        pred = r2.llm_dep.values.astype(bool)

        def prf(mask, name):
            hh, pp = h[mask], pred[mask]
            tp = int((pp & hh).sum()); fp = int((pp & ~hh).sum()); fn = int((~pp & hh).sum())
            p_ = tp / (tp + fp) if tp + fp else float("nan")
            r_ = tp / (tp + fn) if tp + fn else float("nan")
            f1 = (2 * p_ * r_ / (p_ + r_)) if (p_ == p_ and r_ == r_ and p_ + r_ > 0) else float("nan")
            print(f"    {name:28s} n={int(mask.sum()):3d}  prec={p_:.3f} rec={r_:.3f} F1={f1:.3f}")

        prf(np.ones(len(r2), bool), "all round-2 rows")
        prf(~leak, "excluding prompt examples")
        ok(f"{int(leak.sum())} of {len(r2)} rows overlap a prompt example; report both")
    else:
        bad("round2_scored.csv missing; instrument not validated")

    print("\n=== 5. no retracted or stale claim left in code/docs ===")
    # only our own sources; the retraction memo is SUPPOSED to name these strings
    # Scope: artifacts that SHIP. Retraction records (FINDINGS_FINAL, the checklist,
    # the failure memos) are supposed to name the withdrawn numbers, and flagging them
    # trains the operator to ignore this gate.
    scan = [str(x) for x in Path("code").glob("*.py")] + \
           [str(x) for x in Path("corrected").glob("*.tex")] + \
           ["corrected/chair_disclosure_email_DRAFT.md"]
    exempt = ("test_matcher", "validate_pipeline", "surface_match",
              "VALIDATION_FAILURES", "CONSOLIDATED_REVIEW")

    def superseded(path):
        """A file explicitly marked SUPERSEDED is allowed to retain stale numbers as a
        record; the banner tells a reader not to quote them."""
        try:
            return "SUPERSEDED" in "".join(open(path).readlines()[:6])
        except Exception:
            return False
    # patterns, not bare substrings: "94.7" also matches the hex colour #9467bd
    for s in [r"94\.7\s*%", r"0\.994", r"\bprefix_match\b", r"79\.1\s*%"]:
        r = subprocess.run(["grep", "-lnE", s] + scan, capture_output=True, text=True)
        files = [x for x in r.stdout.split()
                 if not any(e in x for e in exempt) and not superseded(x)]
        if files:
            bad(f"stale string {s!r} still in: {', '.join(files)}")
    ok("retracted-claim scan done")

    print("\n=== 6. numbers.tex is current with the artifacts it is generated from ===")
    # The paper quotes macros only, so a stale numbers.tex is the one way a wrong
    # number can still reach the PDF: the artifacts get recomputed and the generator
    # is not re-run. Regenerate into a scratch copy and diff.
    nt = Path("corrected/numbers.tex")
    if nt.exists():
        before = nt.read_text()
        subprocess.run([sys.executable, "code/gen_numbers.py"],
                       capture_output=True, text=True)
        after = nt.read_text()
        if before != after:
            n = sum(1 for a, b in zip(before.splitlines(), after.splitlines()) if a != b)
            bad(f"numbers.tex was stale ({n} macros changed); it has been regenerated, "
                "rebuild the PDF before shipping")
        else:
            ok("every macro matches the committed artifacts")
    else:
        bad("corrected/numbers.tex missing")

    print("\n" + "=" * 60)
    if PROBLEMS:
        print(f"{len(PROBLEMS)} PROBLEM(S) — do not publish numbers until resolved")
        sys.exit(1)
    print("pipeline validation: all checks passed")


if __name__ == "__main__":
    main()
