"""Score every candidate membership instrument against the human labels, and write
data/analysis_corrected/instrument_table.json.

Why this file exists. The earlier draft selected the membership instrument on F1, and F1
turns out to be the wrong criterion: a constant "always in catalog" predictor scores F1
0.78 on this sample, and F1 varies by 0.13 across instruments whose implied Amazon
hallucination rate varies by a factor of 16. The quantity a prevalence estimate is
sensitive to is net bias, the difference between the rate an instrument reports and the
rate humans report on the same items. This script computes both, per catalog and in
aggregate, plus the design-reweighted version, so the paper can argue on the criterion
that matters.

Everything is derived from two committed artifacts:
  round2_scored.csv     201 human-labeled items with every instrument's verdict
  llm_pool_judged.csv   the 1,200-item frame the labeled sample was drawn from

The adopted-instrument column is `llm_dep`, joined straight from the deployment verdict
files, because those verdicts produced every other number in the paper. `llm_says` is the
20-candidate judge that defined the sampling strata and is reported only as a disclosure.
A third column, `llm8_in`, was a separate re-judge of the labeled rows; it disagreed with
deployment on 24 of 201 and is not scored. Validating one instrument and deploying another
is the defect this file exists to keep out of the paper.

Run: code/.venv/bin/python code/instrument_table.py
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, "code")
import numpy as np
import pandas as pd
from surface_match import fold

OUT = Path("data/analysis_corrected")
DEST = OUT / "instrument_table.json"
DATASETS = ["movielens", "yelp", "amazon"]
LLMS = ["mistral", "llama", "gpt-oss", "claude"]
RNG = np.random.default_rng(20260812)
NBOOT = 20000


def scores(pred: np.ndarray, human: np.ndarray) -> dict:
    tp = int((pred & human).sum()); fp = int((pred & ~human).sum())
    fn = int((~pred & human).sum()); tn = int((~pred & ~human).sum())
    P = tp / (tp + fp) if tp + fp else float("nan")
    R = tp / (tp + fn) if tp + fn else float("nan")
    F = 2 * P * R / (P + R) if P == P and R == R and P + R > 0 else float("nan")
    return {"prec": P, "rec": R, "f1": F, "acc": (tp + tn) / len(pred),
            "in_rate": float(pred.mean()), "human_rate": float(human.mean()),
            "bias": float(pred.mean() - human.mean()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": int(len(pred))}


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def boot_ci(fn, *arrs, n=NBOOT):
    N = len(arrs[0])
    vals = []
    for _ in range(n):
        i = RNG.integers(0, N, N)
        v = fn(*[a[i] for a in arrs])
        if v == v:
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar(p1: np.ndarray, p2: np.ndarray, human: np.ndarray) -> float:
    """Exact two-sided binomial test on the discordant pairs."""
    from math import comb
    c1 = (p1 == human) & (p2 != human)
    c2 = (p2 == human) & (p1 != human)
    b, c = int(c1.sum()), int(c2.sum())
    n = b + c
    if n == 0:
        return float("nan")
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def perm_p(vals: np.ndarray, group: np.ndarray, n=20000) -> float:
    """Two-sided permutation test on a difference of group means."""
    obs = abs(vals[group].mean() - vals[~group].mean())
    g = group.copy()
    hits = 0
    for _ in range(n):
        RNG.shuffle(g)
        if abs(vals[g].mean() - vals[~g].mean()) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (n + 1)


def cands_of(r) -> list:
    out = []
    for i in range(1, 9):
        c = str(r[f"cand_{i}"])
        if c and c != "nan":
            out.append(c)
    return out


def exact_predictions(r2: pd.DataFrame) -> np.ndarray:
    """Exact normalized-string match, the trivial matcher a reviewer will ask about.
    Scored on the same eight candidates as every other row of the table."""
    return np.array([fold(r.generated_title) in {fold(c) for c in cands_of(r)}
                     for _, r in r2.iterrows()])


def fullcatalog_rates(r2: pd.DataFrame) -> dict:
    """Every instrument in the table is scored on the eight candidates the annotator
    saw, which is the only information set all of them share. The published token-set
    rule, though, runs over the entire catalog. Measure what that costs, so the table's
    bias column can be read as a lower bound on the published configuration's bias.

    Cached: this is ~70M string comparisons on Amazon.
    """
    cache = OUT / "fullcatalog_preds.csv"
    if cache.exists():
        return pd.read_csv(cache)
    from rapidfuzz import fuzz, process
    from retrieval import get_index
    from utils import normalize_title
    out = r2[["catalog", "generated_title"]].copy()
    out["tokenset_full"] = False
    out["tokensort_full"] = False
    for d in DATASETS:
        m = (r2.catalog == d).values
        titles = [normalize_title(t) for t in get_index(d).titles]
        q = [normalize_title(str(t)) for t in r2.generated_title[m]]
        for nm, sc in (("tokenset_full", fuzz.token_set_ratio),
                       ("tokensort_full", fuzz.token_sort_ratio)):
            M = process.cdist(q, titles, scorer=sc, score_cutoff=90,
                              dtype=np.uint8, workers=-1)
            out.loc[m, nm] = (M.max(axis=1) >= 90)
        print(f"  full-catalog {d}: " + "  ".join(
            f"{nm}={out.loc[m, nm].mean():.3f}"
            for nm in ("tokenset_full", "tokensort_full")), flush=True)
    out.to_csv(cache, index=False)
    return out


def cost_estimate(r2: pd.DataFrame) -> dict:
    """Approximate API cost of the deployment pass, from the actual prompt text on a
    sample of items scaled by the verdict count. An estimate, and labeled as one."""
    from llm_matcher import SYSTEM, build_user
    n_verdicts = sum(1 for d in DATASETS for l in LLMS
                     for _ in (OUT / d / l / "llm_match.jsonl").open()
                     if (OUT / d / l / "llm_match.jsonl").exists())
    samp = r2.sample(min(120, len(r2)), random_state=5)
    chars = []
    for _, r in samp.iterrows():
        cands = [str(r[f"cand_{i}"]) for i in range(1, 9) if str(r[f"cand_{i}"]) != "nan"]
        chars.append(len(build_user([(r.generated_title, cands)])))
    per_item = float(np.mean(chars)) / 4.0          # ~4 chars/token
    sys_tok = len(SYSTEM) / 4.0
    n_calls = int(np.ceil(n_verdicts / 10))
    in_tok = n_verdicts * per_item + n_calls * sys_tok
    out_tok = n_verdicts * 8                        # "<n>: <n>\n"
    usd = in_tok / 1e6 * 3.0 + out_tok / 1e6 * 15.0
    return {"n_verdicts": n_verdicts, "n_calls": n_calls,
            "input_tokens_est": int(in_tok), "usd_est": round(usd, 2)}


def main():
    r2 = pd.read_csv(OUT / "round2_scored.csv")
    human = r2.human_in.values.astype(bool)

    preds = {
        "llm8": r2.llm_dep.values.astype(bool),
        "llm20": (r2.llm_says.astype(str).str.strip() == "IN").values,
        "tokenset": r2.tset_in.values.astype(bool),
        "tokensort": r2.tsort_in.values.astype(bool),
        "surface": r2.surf_in.values.astype(bool),
        "alwaysin": np.ones(len(r2), dtype=bool),
        "exact": exact_predictions(r2),
    }

    res = {"n_labels": int(len(r2)), "n_drawn": 205,
           "human_in_rate": float(human.mean()),
           "instruments": {}, "per_catalog": {}, "reweighted": {}}

    for nm, p in preds.items():
        s = scores(p, human)
        s["kappa"] = kappa(p, human)
        s["kappa_lo"], s["kappa_hi"] = boot_ci(kappa, p, human)
        s["bias_lo"], s["bias_hi"] = boot_ci(
            lambda a, b: float(a.mean() - b.mean()), p, human)
        res["instruments"][nm] = s

    for d in DATASETS:
        m = (r2.catalog == d).values
        res["per_catalog"][d] = {nm: scores(p[m], human[m]) for nm, p in preds.items()}
        res["per_catalog"][d]["n"] = int(m.sum())

    # ---- design-reweighted in-catalog rate ---------------------------------
    # The labeled sample is stratified on the 20-candidate judge's verdict, so a raw
    # mean over it is not a population rate. Reweight each row by the frame share of
    # its stratum. This is the population-level version of the bias column.
    P = pd.read_csv(OUT / "llm_pool_judged.csv")
    w = np.zeros(len(r2))
    for d in DATASETS:
        pool = P[P.catalog == d]
        for v, flag in (("IN", True), ("OUT", False)):
            share = float((pool.llm_in_catalog == flag).mean()) / len(DATASETS)
            m = ((r2.catalog == d) & (r2.llm_says == v)).values
            if m.sum():
                w[m] = share / m.sum()
    w = w / w.sum()
    res["reweighted"] = {"human": float((w * human).sum()),
                         **{nm: float((w * p).sum()) for nm, p in preds.items()}}
    res["reweighted_by_catalog"] = {}
    for d in DATASETS:
        m = (r2.catalog == d).values
        wd = w[m] / w[m].sum()
        res["reweighted_by_catalog"][d] = {
            "human": float((wd * human[m]).sum()),
            **{nm: float((wd * p[m]).sum()) for nm, p in preds.items()}}

    # ---- significance of the adopted instrument against the incumbent -------
    res["mcnemar_llm8_vs_tokenset"] = mcnemar(preds["llm8"], preds["tokenset"], human)
    res["mcnemar_discordant"] = {
        "llm8_only_right": int(((preds["llm8"] == human) &
                                (preds["tokenset"] != human)).sum()),
        "tokenset_only_right": int(((preds["tokenset"] == human) &
                                    (preds["llm8"] != human)).sum())}
    res["llm8_vs_llm20_disagree"] = int((preds["llm8"] != preds["llm20"]).sum())

    # ---- instrument error by catalog and by generating model ---------------
    err = (preds["llm8"] != human)
    res["error_by_catalog"] = {d: float(err[(r2.catalog == d).values].mean())
                               for d in DATASETS}
    res["error_by_catalog_p"] = perm_p(err.astype(float),
                                       (r2.catalog == "amazon").values)
    res["error_by_model"] = {m: float(err[(r2.model == m).values].mean())
                             for m in sorted(r2.model.unique())}

    # ---- self-preference ---------------------------------------------------
    # The objection is that a Claude judge is permissive toward Claude-generated titles,
    # so the quantity to test is the difference in net bias, not in error rate.
    is_claude = (r2.model == "claude").values
    signed = preds["llm8"].astype(float) - human.astype(float)
    res["self_preference"] = {
        "err_claude": float(err[is_claude].mean()),
        "err_other": float(err[~is_claude].mean()),
        "p_err": perm_p(err.astype(float), is_claude),
        "bias_claude": float(signed[is_claude].mean()),
        "bias_other": float(signed[~is_claude].mean()),
        "p_bias": perm_p(signed, is_claude),
        "n_claude": int(is_claude.sum()),
        "n_claude_human_out": int((~human & is_claude).sum()),
    }

    # Two labeled items appear as examples in the judge's system prompt. Report the
    # adopted instrument's F1 with them removed.
    leak_terms = ["se7en", "dumb and dumber", "the rock", "sabrina"]
    leaked = np.array([any(t in str(g).lower() for t in leak_terms)
                       for g in r2.generated_title])
    res["leakage"] = {"n": int(leaked.sum()),
                      "f1_excl": scores(preds["llm8"][~leaked], human[~leaked])["f1"]}

    # ---- blind relabel -----------------------------------------------------
    B = pd.read_csv(OUT / "label_blind60_labeled.csv")
    B["human_in"] = B.same_item.fillna(0).astype(int) > 0
    key = ["catalog", "generated_title"]
    ref = r2[key + ["human_in", "llm8_in", "tset_in", "llm_says"]].drop_duplicates(key)
    J = B.drop_duplicates(key).merge(ref, on=key, how="inner",
                                     suffixes=("_blind", "_hint"))
    hb, hh = J.human_in_blind.values.astype(bool), J.human_in_hint.values.astype(bool)
    shown = (J.llm_says.astype(str).str.strip() == "IN").values
    res["blind"] = {
        "n_relabeled": int(len(B)), "n_matched": int(len(J)),
        "agreement": float((hb == hh).mean()),
        "hinted_agrees_with_shown": float((hh == shown).mean()),
        "blind_agrees_with_shown": float((hb == shown).mean()),
        "anchoring": float((hh == shown).mean() - (hb == shown).mean()),
        "llm8": scores(J.llm8_in.values.astype(bool), hb),
        "tokenset": scores(J.tset_in.values.astype(bool), hb),
        "alwaysin": scores(np.ones(len(J), dtype=bool), hb),
    }

    # ---- design disclosures ------------------------------------------------
    res["strata"] = {d: {v: int(((r2.catalog == d) & (r2.llm_says == v)).sum())
                         for v in ("IN", "OUT")} for d in DATASETS}
    FC = fullcatalog_rates(r2)
    res["fullcatalog"] = {}
    for d in DATASETS:
        m = (r2.catalog == d).values
        wd = w[m] / w[m].sum()
        res["fullcatalog"][d] = {
            "human": float((wd * human[m]).sum()),
            "tokenset_shortlist": float((wd * preds["tokenset"][m]).sum()),
            "tokenset_full": float((wd * FC.tokenset_full.values[m]).sum()),
            "tokensort_full": float((wd * FC.tokensort_full.values[m]).sum()),
            "tokenset_full_raw": float(FC.tokenset_full.values[m].mean())}
    res["cost"] = cost_estimate(r2)

    # ---- human-anchored OOD, recomputed here so the estimator is committed --
    ood = {}
    for d in DATASETS:
        pool = P[P.catalog == d]
        wIN = float((pool.llm_in_catalog == True).mean())  # noqa: E712
        sub = r2[r2.catalog == d]
        pin = sub[sub.llm_says == "IN"].human_in
        pout = sub[sub.llm_says == "OUT"].human_in
        g = wIN * pin.mean() + (1 - wIN) * pout.mean()
        # Jeffreys posterior draws on each stratum rate and on the frame weight, so a
        # saturated stratum gets a proper interval instead of zero variance. One draw of
        # w serves both strata: w and 1-w are the same quantity, not two free ones.
        nIN = int((pool.llm_in_catalog == True).sum())   # noqa: E712
        wd = RNG.beta(nIN + .5, len(pool) - nIN + .5, 200000)
        dr = (wd * RNG.beta(pin.sum() + .5, len(pin) - pin.sum() + .5, 200000)
              + (1 - wd) * RNG.beta(pout.sum() + .5, len(pout) - pout.sum() + .5, 200000))
        ood[d] = {"grounded": float(g), "ood": float(1 - g),
                  "ood_lo": float(np.percentile(1 - dr, 2.5)),
                  "ood_hi": float(np.percentile(1 - dr, 97.5)),
                  "n_in": int(len(pin)), "n_out": int(len(pout))}
    res["ood"] = ood
    # ood_estimates.json is what the paper's per-catalog interval reads from; write it
    # from the same estimator so there is one implementation, not two.
    (OUT / "ood_estimates.json").write_text(json.dumps(ood, indent=1))

    DEST.write_text(json.dumps(res, indent=1))
    print(f"wrote {DEST}")
    for nm, s in res["instruments"].items():
        print(f"  {nm:10s} P={s['prec']:.3f} R={s['rec']:.3f} F1={s['f1']:.3f} "
              f"bias={s['bias']:+.3f} kappa={s['kappa']:.3f}")
    print(f"  reweighted: human {res['reweighted']['human']:.3f}  " +
          "  ".join(f"{k} {v:+.3f}" for k, v in
                    ((k, res['reweighted'][k] - res['reweighted']['human'])
                     for k in ("llm8", "tokenset", "surface", "tokensort"))))
    print(f"  McNemar llm8 vs token-set p={res['mcnemar_llm8_vs_tokenset']:.4f}")
    print(f"  OOD: " + "  ".join(f"{d} {v['ood']*100:.1f} "
                                 f"[{v['ood_lo']*100:.1f},{v['ood_hi']*100:.1f}]"
                                 for d, v in ood.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
