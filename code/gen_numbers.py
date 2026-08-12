"""Generate corrected/numbers.tex: every number the paper quotes, as a LaTeX macro
computed from a committed artifact.

Most defects the review pool found in the previous draft were prose/data drift: the paper
said the median conformal reduction was 0.22 pp when the artifact gives 0.080, said "one
cell gets worse" when four do, said the exact-match band carries 55% when it carries
57.5%. Those are not analysis errors, they are transcription errors, and they recur
because numbers get typed into LaTeX by hand and then the analysis is re-run.

With this file the paper references macros only, so the prose cannot disagree with the
data, and a recompute updates every figure at once. A macro used but never generated is a
hard LaTeX error rather than a stale number, which is the behaviour we want.

Run: python code/gen_numbers.py    (after any recompute)
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, "code")
import numpy as np
import pandas as pd

OUT = Path("data/analysis_corrected")
DEST = Path("corrected/numbers.tex")
DATASETS = ["movielens", "yelp", "amazon"]
LLMS = ["mistral", "llama", "gpt-oss", "claude"]
ROMAN = {"movielens": "ML", "yelp": "Yelp", "amazon": "Amz"}
LROMAN = {"mistral": "Mistral", "llama": "Llama", "gpt-oss": "GptOss", "claude": "Claude"}

lines, prov, missing = [], [], []


def emit(name, value, source, fmt="{:.1f}"):
    """Define \\name and record where the value came from."""
    v = fmt.format(value) if isinstance(value, (int, float, np.floating)) else str(value)
    lines.append(f"\\newcommand{{\\{name}}}{{{v}}}")
    prov.append((name, v, source))


def main():
    # ---- human-anchored OOD per catalog -------------------------------------
    p = OUT / "ood_estimates.json"
    if p.exists():
        est = json.load(p.open())
        for d, r in est.items():
            k = ROMAN[d]
            emit(f"ood{k}", r["ood"] * 100, "ood_estimates.json")
            emit(f"ood{k}Lo", r["ood_lo"] * 100, "ood_estimates.json")
            emit(f"ood{k}Hi", r["ood_hi"] * 100, "ood_estimates.json")
            emit(f"grounded{k}", r["grounded"] * 100, "ood_estimates.json")
    else:
        missing.append("ood_estimates.json")

    # ---- per-cell metrics under the adopted matcher --------------------------
    p = OUT / "llm_metrics.parquet"
    if p.exists():
        M = pd.read_parquet(p)
        for _, r in M.iterrows():
            k = ROMAN[r.dataset] + LROMAN[r.llm]
            emit(f"grnd{k}", r.grounded * 100, "llm_metrics.parquet")
            emit(f"conf{k}", r.conf * 100, "llm_metrics.parquet")
            emit(f"gap{k}", r.gap * 100, "llm_metrics.parquet", "{:+.1f}")
            emit(f"ece{k}", r.ece, "llm_metrics.parquet", "{:.3f}")
            emit(f"brier{k}", r.brier, "llm_metrics.parquet", "{:.3f}")
            emit(f"n{k}", int(r.n), "llm_metrics.parquet", "{:d}")
        g = M.gap
        emit("nUnder", int((g > 0.02).sum()), "llm_metrics.parquet", "{:d}")
        emit("nOver", int((g < -0.02).sum()), "llm_metrics.parquet", "{:d}")
        emit("nFlat", int(((g >= -0.02) & (g <= 0.02)).sum()), "llm_metrics.parquet", "{:d}")
        emit("meanECE", M.ece.mean(), "llm_metrics.parquet", "{:.3f}")
        emit("meanBrier", M.brier.mean(), "llm_metrics.parquet", "{:.3f}")
        emit("totalRecs", int(M.n.sum()), "llm_metrics.parquet", "{:d}")
        emit("confBandLo", M.conf.min() * 100, "llm_metrics.parquet", "{:.0f}")
        emit("confBandHi", M.conf.max() * 100, "llm_metrics.parquet", "{:.0f}")
        emit("confBandWidth", (M.conf.max() - M.conf.min()) * 100,
             "llm_metrics.parquet", "{:.0f}")
        emit("grndRatioApprox", M.grounded.max() / M.grounded.min(),
             "llm_metrics.parquet", "{:.1f}")
        for d in DATASETS:
            s = M[M.dataset == d]
            emit(f"grndLo{ROMAN[d]}", s.grounded.min() * 100, "llm_metrics.parquet")
            emit(f"grndHi{ROMAN[d]}", s.grounded.max() * 100, "llm_metrics.parquet")
        # ECE collapses to the absolute mean gap only where confidence is concentrated
        # in bins of one sign. Count it rather than asserting "all twelve".
        eq = (np.abs(M.ece - np.abs(M.gap)) < 5e-4)
        emit("eceEqCells", int(eq.sum()), "llm_metrics.parquet", "{:d}")
        # Whose property is the confidence level: the model's or the catalog's?
        bm = M.groupby("llm").conf.mean() * 100
        bc = M.groupby("dataset").conf.mean() * 100
        emit("varBetweenModel", bm.var(ddof=0), "llm_metrics.parquet")
        emit("varBetweenCatalog", bc.var(ddof=0), "llm_metrics.parquet")
        for l, v in bm.items():
            emit(f"confModel{LROMAN[l]}", v, "llm_metrics.parquet")
        emit("grndSwing", (M.grounded.max() - M.grounded.min()) * 100,
             "llm_metrics.parquet", "{:.0f}")
    else:
        missing.append("llm_metrics.parquet")

    # ---- instrument comparison against the round-2 human labels --------------
    # Source is instrument_table.json, written by code/instrument_table.py. The adopted
    # instrument is the 8-candidate judge that produced the deployment labels, not the
    # 20-candidate judge that defined the sampling strata; the latter is emitted under
    # the LLMval suffix and appears only in the depth disclosure.
    p = OUT / "instrument_table.json"
    SUF = {"llm8": "LLM", "llm20": "LLMval", "tokenset": "TokenSet",
           "tokensort": "TokenSort", "surface": "Surface",
           "alwaysin": "AlwaysIn", "exact": "Exact"}
    if p.exists():
        I = json.load(p.open())
        src = "instrument_table.json"
        for key, s in SUF.items():
            m = I["instruments"][key]
            emit(f"prec{s}", m["prec"], src, "{:.3f}")
            emit(f"rec{s}", m["rec"], src, "{:.3f}")
            emit(f"fone{s}", m["f1"], src, "{:.3f}")
            emit(f"acc{s}", m["acc"], src, "{:.3f}")
            emit(f"bias{s}", m["bias"], src, "{:+.3f}")
            emit(f"kappa{s}", m["kappa"], src, "{:.3f}")
            emit(f"kappa{s}Lo", m["kappa_lo"], src, "{:.3f}")
            emit(f"kappa{s}Hi", m["kappa_hi"], src, "{:.3f}")
            for d in DATASETS:
                c = I["per_catalog"][d][key]
                emit(f"bias{s}{ROMAN[d]}", c["bias"], src, "{:+.3f}")
                emit(f"fone{s}{ROMAN[d]}", c["f1"], src, "{:.3f}")
                emit(f"prec{s}{ROMAN[d]}", c["prec"], src, "{:.3f}")
                emit(f"rec{s}{ROMAN[d]}", c["rec"], src, "{:.3f}")
        emit("nLabels", I["n_labels"], src, "{:d}")
        emit("nDrawn", I["n_drawn"], src, "{:d}")
        emit("nMLOut", I["strata"]["movielens"]["OUT"], src, "{:d}")
        emit("nAmzVal", I["per_catalog"]["amazon"]["n"], src, "{:d}")
        emit("nLeak", I["leakage"]["n"], src, "{:d}")
        emit("foneLLMexLeak", I["leakage"]["f1_excl"], src, "{:.3f}")
        emit("mcnemarP", I["mcnemar_llm8_vs_tokenset"], src, "{:.2f}")
        emit("depthDisagree", I["llm8_vs_llm20_disagree"], src, "{:d}")

        # design-reweighted in-catalog rate, and each instrument's gap against it
        rw = I["reweighted"]
        emit("rwHuman", rw["human"] * 100, src)
        for key, s in SUF.items():
            emit(f"rw{s}", rw[key] * 100, src)
            emit(f"rwD{s}", (rw[key] - rw["human"]) * 100, src, "{:+.1f}")

        for d in DATASETS:
            emit(f"errCat{ROMAN[d]}", I["error_by_catalog"][d], src, "{:.3f}")
        emit("errCatP", I["error_by_catalog_p"], src, "{:.4f}")

        sp = I["self_preference"]
        emit("spBiasClaude", sp["bias_claude"], src, "{:+.3f}")
        emit("spBiasOther", sp["bias_other"], src, "{:+.3f}")
        emit("spPbias", sp["p_bias"], src, "{:.2f}")
        emit("spErrClaude", sp["err_claude"], src, "{:.3f}")
        emit("spErrOther", sp["err_other"], src, "{:.3f}")
        emit("spPerr", sp["p_err"], src, "{:.2f}")
        emit("spNClaudeOut", sp["n_claude_human_out"], src, "{:d}")

        b = I["blind"]
        emit("blindN", b["n_relabeled"], src, "{:d}")
        emit("blindMatched", b["n_matched"], src, "{:d}")
        emit("blindAgree", b["agreement"], src, "{:.3f}")
        emit("blindAgreePct", b["agreement"] * 100, src)
        emit("blindAnchor", b["anchoring"], src, "{:+.3f}")
        emit("blindFoneLLM", b["llm8"]["f1"], src, "{:.3f}")
        emit("blindBiasLLM", b["llm8"]["bias"], src, "{:+.3f}")
        emit("blindFoneTokenSet", b["tokenset"]["f1"], src, "{:.3f}")
        emit("blindBiasTokenSet", b["tokenset"]["bias"], src, "{:+.3f}")

        # in-catalog rate under the published full-catalog scan, against the annotator
        for d in DATASETS:
            f = I["fullcatalog"][d]
            emit(f"fcHuman{ROMAN[d]}", f["human"] * 100, src)
            emit(f"fcTset{ROMAN[d]}", f["tokenset_full"] * 100, src)
            emit(f"fcTsetShort{ROMAN[d]}", f["tokenset_shortlist"] * 100, src)
            emit(f"fcTsort{ROMAN[d]}", f["tokensort_full"] * 100, src)
        # confusion counts the caption quotes
        az = I["per_catalog"]["amazon"]
        emit("tsetFpAmz", az["tokenset"]["fp"], src, "{:d}")
        emit("humanOutAmz", az["tokenset"]["fp"] + az["tokenset"]["tn"], src, "{:d}")
        emit("recExactPct", I["instruments"]["exact"]["rec"] * 100, src)
        # the whole selection argument in two numbers: how far F1 moves across the
        # instruments, against how far the quantity they are chosen to estimate moves
        real = [k for k in SUF if k != "llm20"]
        f1s = [I["instruments"][k]["f1"] for k in real]
        bz = [I["per_catalog"]["amazon"][k]["bias"] for k in real]
        emit("foneSpan", max(f1s) - min(f1s), src, "{:.2f}")
        emit("biasSpanAmz", max(bz) - min(bz), src, "{:.2f}")

        c = I["cost"]
        emit("costVerdicts", c["n_verdicts"], src, "{:,d}")
        emit("costCalls", c["n_calls"], src, "{:,d}")
        emit("costUSD", c["usd_est"], src, "{:.0f}")
    else:
        missing.append("instrument_table.json")

    # ---- D1 table inputs ------------------------------------------------------
    p = OUT / "d1_tables.json"
    if p.exists():
        T = json.load(p.open())
        for cell, v in T["ood"].items():
            d, l = cell.split("/")
            k = ROMAN[d] + LROMAN[l]
            emit(f"oodC{k}", v["ood"] * 100, "d1_tables.json", "{:.1f}")
            emit(f"oodC{k}Lo", v["lo"] * 100, "d1_tables.json", "{:.1f}")
            emit(f"oodC{k}Hi", v["hi"] * 100, "d1_tables.json", "{:.1f}")
            emit(f"nUsers{k}", v["users"], "d1_tables.json", "{:d}")
        for key, v in T["strata"].items():
            d, l, st = key.split("/")
            k = ROMAN[d] + LROMAN[l] + st.capitalize()
            emit(f"ece{k}", v["ece"], "d1_tables.json", "{:.3f}")
            emit(f"brier{k}", v["brier"], "d1_tables.json", "{:.3f}")
            emit(f"grnd{k}", v["grounded"] * 100, "d1_tables.json", "{:.1f}")
        for cell, v in T["conformal"].items():
            d, l = cell.split("/")
            k = ROMAN[d] + LROMAN[l]
            # LaTeX control sequences are letters only: no digits in macro names
            ANAME = {0.05: "Afive", 0.1: "Aten", 0.15: "Afifteen", 0.2: "Atwenty"}
            for a in (0.05, 0.1, 0.15, 0.2):
                x = v.get(f"ood_{a}")
                if x is not None:
                    emit(f"cf{k}{ANAME[a]}", x * 100, "d1_tables.json", "{:.1f}")
                c = v.get(f"cov_{a}")
                if c is not None:
                    emit(f"cov{k}{ANAME[a]}", c, "d1_tables.json", "{:.3f}")
        # largest / median conformal reduction, computed not typed
        red = []
        for cell, v in T["conformal"].items():
            for a in (0.05, 0.1, 0.15, 0.2):
                x = v.get(f"ood_{a}")
                if x is not None:
                    red.append(v["base"] - x)
        if red:
            emit("cfMaxRed", max(red) * 100, "d1_tables.json", "{:.2f}")
            emit("cfMedRed", float(np.median(red)) * 100, "d1_tables.json", "{:.2f}")
            emit("cfWorseCells", int(sum(1 for r in red if r < 0)), "d1_tables.json", "{:d}")
            emit("cfEntries", len(red), "d1_tables.json", "{:d}")
        for cell, v in T["ablation"].items():
            d, l = cell.split("/")
            k = ROMAN[d] + LROMAN[l]
            emit(f"ablEceBase{k}", v["ece_base"], "d1_tables.json", "{:.3f}")
            emit(f"ablEcePert{k}", v["ece_pert"], "d1_tables.json", "{:.3f}")
            emit(f"ablConfBase{k}", v["conf_base"], "d1_tables.json", "{:.1f}")
            emit(f"ablConfPert{k}", v["conf_pert"], "d1_tables.json", "{:.1f}")
            emit(f"ablGrndBase{k}", v["grounded_base"] * 100, "d1_tables.json", "{:.1f}")
            emit(f"ablGrndPert{k}", v["grounded_pert"] * 100, "d1_tables.json", "{:.1f}")
    else:
        missing.append("d1_tables.json")

    DEST.parent.mkdir(parents=True, exist_ok=True)
    hdr = ["% GENERATED by code/gen_numbers.py -- do not edit by hand.",
           "% Every value below is computed from a committed artifact; see",
           "% corrected/numbers_provenance.txt for the source of each macro.", ""]
    DEST.write_text("\n".join(hdr + sorted(lines)) + "\n")
    Path("corrected/numbers_provenance.txt").write_text(
        "\n".join(f"{n:24s} = {v:>10s}   <- {s}" for n, v, s in sorted(prov)) + "\n")
    print(f"wrote {len(lines)} macros -> {DEST}")
    if missing:
        # Exiting 0 here would let a caller believe an empty numbers.tex was a success,
        # and the paper would then fail to build with no indication why. A check that
        # cannot report failure is the defect class this whole correction is about.
        print(f"  MISSING artifacts, macros NOT generated: {', '.join(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
