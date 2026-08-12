# Round-2 labeling — `label_round2.csv` (205 rows, ~25-30 min)

## Why again

Round 1's candidates came from the matcher being validated, so the labels inherited its
blind spots. Verified case: 4 of 7 `Amelie` rows were labeled "not in catalog" only
because a diacritic bug stopped the true entry from ever being retrieved, so the correct
answer was never on screen. Those labels cannot validate anything.

Round 2 fixes both halves. Candidates now come from an independent retriever that shares
no logic with any matcher under test, and **8** are shown instead of 3. The sample is
stratified on the LLM matcher's verdict, so these labels measure both its false positives
and its false negatives.

## The question (unchanged)

> **Does the generated title refer to the SAME real-world item as one of the candidates?**

Put a number in `same_item`:

| answer | meaning |
|---|---|
| `1`-`8` | same item as that candidate |
| `0` | none of them is the same item |
| `?` | genuinely cannot tell (a handful is fine) |

## Same item, or not

**Same** (answer 1-8):
- alternate or original-language title: `Se7en` = `Seven (a.k.a. Se7en) (1995)`
- moved article: `The Rock` = `Rock, The (1996)`
- punctuation, accents, spacing: `Sabrina's Cafe` = `Sabrina's Cafe`
- catalog entry carries extra descriptive or marketing text for the same product
- business with or without a location suffix: `Walk-On's` = `Walk-On's - New Orleans`

**Not the same** (answer 0):
- sequel or another entry in a series: `Dumb and Dumber To` is not `Dumb and Dumber`
- different edition, size, count, colour, or numbered variant: a 100-block set is not a
  200-block set; `Barbie Fashionistas #171` is not `#166`
- an accessory, case, or expansion for a product is not that product
- a different business or work that merely shares words

The hardest rows are numbered and sized variants. Read the numbers, not just the words.

## Ignore the two hint columns while deciding

`llm_says` and `llm_pick` show what the LLM matcher concluded. They are recorded so we can
score it against you. **Do not let them anchor you** — the point is to find where it is
wrong, so disagreeing is the useful signal. Decide from the candidate list, then move on.

## Coverage note

MovieLens has only 5 rows in the OUT stratum because the LLM matcher marks ~98.8% of
MovieLens items as in-catalog, so there are few OUT cases to sample. MovieLens recall will
therefore carry a wide interval; that gets reported, not hidden.

## Return

Save with `same_item` filled and hand the file back.
