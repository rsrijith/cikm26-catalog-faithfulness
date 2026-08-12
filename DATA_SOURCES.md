# Getting the three source catalogs

This repository publishes our own artifacts in full: the generated titles, the judge
verdicts, the human labels, every instrument's prediction, the annotation guideline, the
prompt and the code. It does **not** republish the catalogs those artifacts refer to.

Two of the three licenses forbid that. MovieLens: *"The user may not redistribute the data
without separate permission."* Yelp Terms of Use §4.A and §4.D: *"you may not publicly
display any of the Data to any third party"* and *"share or make available the Data to any
third party."* Rather than ignore those terms or drop the two catalogs from the release,
we refer to their entries by `item_id` and let you restore the titles from your own
licensed copy.

Amazon Reviews 2023 states no such restriction, so its titles are included as they are.

## Where each catalog comes from

| catalog | source | access | license |
|---|---|---|---|
| MovieLens-25M | GroupLens, University of Minnesota | direct download, https://grouplens.org/datasets/movielens/25m/ | https://files.grouplens.org/datasets/movielens/ml-25m-README.html — no redistribution without permission, non-commercial, cite the dataset |
| Yelp Open Dataset | Yelp Inc. | registration and agreement required, https://business.yelp.com/data/resources/open-dataset/ | https://s3-media0.fl.yelpcdn.com/assets/srv0/engineering_pages/f64cb2d3efcc/assets/vendor/Dataset_User_Agreement.pdf — academic use, no public display or redistribution, findings submitted to Yelp before publication (§3) |
| Amazon Reviews 2023 (Toys & Games) | McAuley Lab, UC San Diego | direct download, https://amazon-reviews-2023.github.io/ | no redistribution restriction stated; cite the dataset paper |

Read each license yourself before using the data. The summaries above are our reading, not
legal advice, and the terms change.

## Building the catalogs

`code/prep_movielens.py`, `code/prep_yelp.py` and `code/prep_amazon.py` are the scripts we
used. MovieLens and Amazon download automatically. Yelp does not, because its download is
gated behind the agreement: accept it, then put `yelp_academic_dataset_business.json` and
`yelp_academic_dataset_review.json` in `data/raw/yelp/` before running the script.

```bash
python code/prep_movielens.py
python code/prep_amazon.py
python code/prep_yelp.py          # after the manual download
```

Each writes `data/processed/<dataset>/catalog.parquet` with columns `item_id`, `title`,
`normalized_title`, `interaction_count`, `popularity_quartile`. Only `item_id`, `title` and
`popularity_quartile` matter for restoring this release.

You do not have to use our scripts. Any table mapping the dataset's own item identifier to
its title will do: MovieLens `movieId`, Yelp `business_id`, Amazon `parent_asin`.

## Restoring what is withheld

```bash
python rehydrate.py --catalogs /path/to/data/processed
```

This writes a `rehydrated/` copy of every affected file with the MovieLens and Yelp titles
and quartiles joined back in. We verified the round trip: all 18,833 verdict records and
all 2,088 candidate cells come back identical to the originals, so a licensed reader
reproduces every number in the paper from the restored files.

`rehydrated/` is git-ignored here. If you fork this repository, do not commit it. That
content is MovieLens and Yelp data under their licenses, not ours to pass on.

## What is withheld, exactly

| file | field | MovieLens | Yelp | Amazon |
|---|---|---|---|---|
| `verdicts/*.jsonl` | `match_title`, `match_quartile` | `null` | `null` | present |
| `verdicts/*.jsonl` | `title`, `in_catalog`, `match_id` | present | present | present |
| `labels/human_labels_201.csv` | `cand_1`…`cand_8`, `llm_pick` | empty | empty | present |
| `labels/human_labels_201.csv` | `cand_1_id`…`cand_8_id`, and every label and verdict column | present | present | present |
| `labels/human_labels_blind_60.csv` | `cand_1`…`cand_8` | empty | empty | present |
| `labels/sampling_frame_1200.csv` | `llm_match_title`, `llm_match_quartile` | empty | empty | present |

In the verdict files, `"match_title": null` means the title is withheld and `""` means the
judge found no match. The two are never confused: `in_catalog` and `match_id` carry the
verdict either way.

The `title` field in every verdict record is the recommender's own output, not the
catalog's, so it is never withheld. That field is what the audit is about. It sometimes
coincides with a real catalog entry, which is precisely the thing being measured.

`match_quartile` is a popularity bucket we derived from interaction counts, so on the two
restricted catalogs it is treated as their data too and withheld. Nothing in the paper
conditions on it: the popularity strata in the results are keyed to the user's held-out
target item, not to the matched item.
