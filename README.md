# Catalog-membership measurement for LLM recommender audits

Data and scoring code for *Do LLM Recommenders Know When They're Hallucinating? Auditing
Confidence Calibration in Catalog Faithfulness* (CIKM 2026, short paper).

The paper's methodological claim is that the string matcher deciding whether a generated
title exists in the catalog is a measurement instrument, that its choice moves a published
hallucination rate by more than a factor of ten, and that the criterion normally used to
pick one cannot tell the candidates apart. Everything needed to check that is here.

On Amazon Toys, the token-set rule used throughout this literature puts 93.8% of the
labeled items in catalog. A human annotator puts 48.9% there.

## What is in the box

```
verdicts/<catalog>_<model>.jsonl   18,833 membership verdicts, one per recommendation
labels/human_labels_201.csv        201 hand-labeled items, with every instrument's verdict
labels/human_labels_blind_60.csv   60 of those relabeled with the model's verdict hidden
labels/sampling_frame_1200.csv     the 1,200-item frame the labeled sample was drawn from
labels/annotation_guideline.md     the instructions the annotator worked from
judge_prompt.txt                   the verbatim system prompt given to the judge
code/                              scoring, regeneration, and the three catalog prep scripts
rehydrate.py                       restores the withheld MovieLens and Yelp fields
DATA_SOURCES.md                    where to get the three catalogs, and their licenses
```

**Read `DATA_SOURCES.md` first if you want the catalog titles.** MovieLens and Yelp both
forbid redistribution, so their entries are referred to by `item_id` here and restored from
your own licensed copy with `rehydrate.py`. Amazon titles are included directly. The round
trip is exact: all 18,833 verdict records and all 2,088 candidate cells come back
identical, so nothing about reproducibility is lost.

### verdicts/

One JSON object per line:

```json
{"title": "Barbie Collector Holiday 2007 Doll", "in_catalog": true,
 "match_title": "Barbie 2007 Holiday Collector Doll",
 "match_id": "B000OYYDSW", "match_quartile": "Q1"}
```

`title` is what the recommender emitted, and is ours. `in_catalog` is the judge's verdict.
`match_id` names the catalog entry it selected, empty when it selected none.
`match_title` and `match_quartile` come from the catalog, so on MovieLens and Yelp they are
`null`, meaning withheld, as distinct from `""`, meaning the judge found no match.

### labels/human_labels_201.csv

The gold standard. `generated_title` is the recommendation; `cand_1_id` … `cand_8_id` are
the eight catalog entries the annotator was shown, retrieved by an index whose first block
is independent of the matchers under test and whose second is not, with `cand_1` … `cand_8` carrying their titles where we may publish
them; `same_item` is the candidate number the annotator picked, or 0 for none; `human_in`
is `same_item > 0`. The remaining columns are each instrument's verdict on the same item:
`llm_dep` (the adopted instrument, joined from the deployment verdicts in `verdicts/`),
`llm_says` (a 20-candidate judge, used only to define the sampling strata), `tset_in`,
`tsort_in`, `surf_in`. A retired column `llm8_in` records a separate re-judge of these rows
that disagreed with deployment on 24 of 201; it is kept for the record and is not scored.

Four of the 205 drawn items were left undecided and dropped, two of them from the MovieLens
out-of-catalog stratum, which is why that stratum holds only three rows and its interval in
the paper is wide.

## Reproducing the paper's numbers

`code/instrument_table.py` scores every instrument against the labels and writes
`instrument_table.json`; `code/gen_numbers.py` turns that into the LaTeX macros the
manuscript quotes, so no number in the paper is typed by hand. What is here regenerates the
instrument table and the §3.3 statistics. The per-cell calibration tables and the reliability
figure derive from recommendation-level files that are not in this release, so those are not
independently rebuildable from it. The scripts expect `round2_scored.csv` and
`llm_pool_judged.csv`; those are `labels/human_labels_201.csv` and
`labels/sampling_frame_1200.csv` here. `code/test_matcher.py` is a
regression suite with one case per matcher bug found during this work, and
`code/validate_pipeline.py` is the pre-publication gate: label coverage, matched titles
resolving to real catalog entries, no conflation of "no match" with "unusable input",
validation at the deployed retrieval depth, and a check that the generated macros are
current with the artifacts.

Running the judge again needs `ANTHROPIC_API_KEY` and the three catalogs. Hosted decoding
at temperature 0 is not bit-reproducible, so a re-run will not reproduce every verdict; the
files here are the ones the paper's numbers come from.

## Caveats worth reading before reusing this

The human labels come from a single annotator, with no second-rater agreement statistic.
The annotator chose among eight retrieved candidates, so the gold standard inherits that
retrieval's ceiling, and we do not have a clean measurement of where that ceiling sits. An
earlier version of this README claimed the independent half of the retriever rescued 43 of
43 surface-matcher misses. That check was a tautology: the candidates it searched were the
same list it drew the answer from, so it could not have returned any other number. It is
withdrawn.

The judge shares a vendor with one of the four audited recommenders. It never sees which
model produced a title, and its net bias on that vendor's output is not distinguishable
from its bias elsewhere, but the test rests on only seven items from that vendor the
annotator marked out-of-catalog, so it is weak rather than a clean acquittal. It is also
pooled across catalogs in a setting where judge error is known to be catalog-dependent.

About 1% of items go unanswered per pass, spread over roughly one batch in twenty, because
a reply occasionally omits an item. Under the code that produced these files an unanswered
item was written as a non-match, so on the order of 190 of the 18,833 verdicts here were
never actually judged and are recorded as out-of-catalog. `code/verify_verdicts.py`
measures the effect: re-judging 500 out-of-catalog records against 500 in-catalog controls
puts the excess flip rate at +2.0 pp (95% CI -1.0 to +5.0, p = 0.19). The current code
records a `judged` flag so the distinction is preserved going forward.

Judge error is not uniform across catalogs: 0.000 on MovieLens, 0.125 on Yelp, 0.231 on
Amazon. That gradient runs in the same direction as the paper's headline finding, so the
per-catalog rates should be read as the intervals the paper reports rather than as points.

`human_labels_blind_60.csv` is a partial control, not a clean one. It removed the judge's
verdict from the labeling form, but it also drew its candidates at a different depth, so
candidates 5 through 8 differ from the round-2 form on 58 of 59 rows. Hint removal and
candidate substitution are confounded there. It supports only the weak claim that no large
anchoring effect is visible at this sample size.

## License

Our contributions here — the human labels, the judge verdicts, the annotation guideline,
the prompt and the code — are released under CC BY 4.0 for the data and MIT for the code.
See `LICENSE`.

That covers our work only. The MovieLens, Yelp and Amazon catalogs remain under their own
terms, which are linked in `DATA_SOURCES.md` and which you accept directly with each
provider. Anything you restore with `rehydrate.py` is theirs, not ours to license.

## Citation

```bibtex
@inproceedings{ravikumar2026catalog,
  author    = {Ravikumar, Srijith},
  title     = {Do {LLM} Recommenders Know When They're Hallucinating?
               Auditing Confidence Calibration in Catalog Faithfulness},
  booktitle = {Proceedings of the 35th ACM International Conference on Information
               and Knowledge Management (CIKM '26)},
  year      = {2026}
}
```
