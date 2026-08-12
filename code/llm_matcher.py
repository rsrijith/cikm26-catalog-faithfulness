"""LLM catalog matcher.

Catalog membership is a semantic judgment, not a string-similarity one. Two rule-based
matchers written for this paper failed in opposite directions on exactly the cases that
matter: "Se7en" vs "Seven (a.k.a. Se7en) (1995)" is the same film and no length-aware
string rule matches it; "Dumb and Dumber To" vs "Dumb and Dumber (1994)" is a different
film and every containment rule matches it. This module asks a model the question
directly, over candidates from an independent recall-oriented retriever
(code/retrieval.py), and is validated against the 252 human labels before use.

Prompt-cached system block, temperature 0, batched. Records model id and date.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, "code")
import pandas as pd
from retrieval import get_index

OUT = Path("data/analysis_corrected")
DEFAULT_MODEL = "claude-sonnet-4-6"   # must match the model the paper names
BATCH = 10

SYSTEM = """You decide whether a recommended item title refers to the same real-world \
item as one of several candidate entries from a product/media/business catalog.

Answer with the candidate number if one of them is the SAME item, or 0 if none is.

Same item means the same specific entity, allowing for surface differences:
- alternate or original-language titles: "Se7en" = "Seven (a.k.a. Se7en) (1995)"
- moved articles: "The Rock" = "Rock, The (1996)"
- punctuation, accents, spacing: "Sabrina's Cafe" = "Sabrina's Cafe"
- a catalog entry carrying extra descriptive or marketing text for the same product
- a business listed with or without a location suffix or descriptor

NOT the same item:
- a sequel, prequel, or different entry in a series: "Dumb and Dumber To" is NOT \
"Dumb and Dumber"
- a different edition, size, count, colour, or variant of a product: a 100-block set is \
NOT a 200-block set
- an accessory, case, or expansion for a product is NOT that product
- a different business that shares words in its name
- a different work that shares words in its title

Output format: one line per item, exactly "<item_number>: <candidate_number>".
No explanation."""


def build_user(items):
    parts = []
    for n, (gen, cands) in enumerate(items, 1):
        lines = [f"ITEM {n}", f"  recommended: {gen}"]
        for ci, c in enumerate(cands, 1):
            lines.append(f"  {ci}. {c}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts) + "\n\nAnswer each item."


def parse_reply(text, n_items, n_cands):
    out = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        a, b = line.split(":", 1)
        try:
            i = int(a.strip().split()[-1])
            v = int(b.strip().split()[0])
        except (ValueError, IndexError):
            continue
        if 1 <= i <= n_items and 0 <= v <= n_cands.get(i, 0):
            out[i] = v
    return out


def judge(rows, model=DEFAULT_MODEL, k=12, verbose=True):
    """rows: list of dicts with 'catalog' and 'generated_title'.
    Returns the same list with 'llm_match_title' and 'llm_in_catalog' added."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    idxs = {}
    for r in rows:
        d = r["catalog"]
        if d not in idxs:
            idxs[d] = get_index(d)
    # attach candidates
    for r in rows:
        idx = idxs[r["catalog"]]
        cids = idx.retrieve(r["generated_title"], k=k)
        r["_cands"] = [idx.titles[i] for i in cids]
        r["_cand_ids"] = [idx.ids[i] for i in cids]
        r["_cand_quart"] = [idx.quart[i] for i in cids]

    done = 0
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        items = [(r["generated_title"], r["_cands"]) for r in chunk]
        ncands = {i + 1: len(c) for i, (_, c) in enumerate(items)}
        for attempt in range(4):
            try:
                resp = client.messages.create(
                    model=model, max_tokens=500, temperature=0.0,
                    system=[{"type": "text", "text": SYSTEM,
                             "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": build_user(items)}],
                )
                ans = parse_reply(resp.content[0].text, len(chunk), ncands)
                break
            except Exception as e:
                if attempt == 3:
                    print(f"  FAIL batch {start}: {e}", flush=True)
                    ans = {}
                else:
                    time.sleep(2 ** attempt)
        for i, r in enumerate(chunk, 1):
            v = ans.get(i, 0)
            r["llm_in_catalog"] = bool(v)
            r["llm_match_title"] = r["_cands"][v - 1] if v else ""
            r["llm_match_id"] = r["_cand_ids"][v - 1] if v else ""
            r["llm_match_quartile"] = r["_cand_quart"][v - 1] if v else ""
        done += len(chunk)
        if verbose and done % 100 < BATCH:
            print(f"  judged {done}/{len(rows)}", flush=True)
    for r in rows:
        r.pop("_cands", None); r.pop("_cand_ids", None); r.pop("_cand_quart", None)
    return rows
