"""Run the LLM matcher over every recommendation in all 12 cells.

Chosen on evidence: against 201 round-2 human labels the LLM matcher scores F1 0.870 /
accuracy 0.841, against surface v2 at 0.792/0.781, token-sort 0.762/0.751 and the
published token-set 0.849/0.786. The decisive gap is Amazon recall, where the surface
matcher reaches only 0.189 and so converts real catalog items into hallucinations.

Resumable per cell (appends JSONL, skips completed titles), concurrent across batches,
system block prompt-cached.
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, "code")
import pandas as pd
from llm_matcher import SYSTEM, build_user, parse_reply, BATCH
from retrieval import get_index

OUT = Path("data/analysis_corrected")
DATASETS = ["movielens", "yelp", "amazon"]
LLMS = ["mistral", "llama", "gpt-oss", "claude"]
MODEL = "claude-sonnet-4-6"
WORKERS = 8
_lock = threading.Lock()


def run_cell(client, dataset, llm, idx):
    p = OUT / dataset / llm / "llm_match.jsonl"
    done = set()
    if p.exists():
        with p.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["title"])
                except Exception:
                    pass
    df = pd.read_parquet(OUT / dataset / llm / "final.parquet")
    todo = [t for t in df["title"].astype(str).tolist() if t not in done]
    todo = list(dict.fromkeys(todo))  # unique titles only; map back later
    if not todo:
        print(f"  {dataset}/{llm}: complete", flush=True)
        return
    print(f"  {dataset}/{llm}: {len(todo)} unique titles to judge", flush=True)
    fh = p.open("a")

    def do_batch(chunk):
        items = []
        for t in chunk:
            cids = idx.retrieve(t, k=12)
            items.append((t, [idx.titles[i] for i in cids][:8], cids[:8]))
        payload = [(t, c) for t, c, _ in items]
        ncands = {i + 1: len(c) for i, (_, c) in enumerate(payload)}
        ans = {}
        for attempt in range(4):
            try:
                r = client.messages.create(
                    model=MODEL, max_tokens=500, temperature=0.0,
                    system=[{"type": "text", "text": SYSTEM,
                             "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": build_user(payload)}],
                )
                ans = parse_reply(r.content[0].text, len(payload), ncands)
                break
            except Exception as e:
                if attempt == 3:
                    print(f"    FAIL: {str(e)[:70]}", flush=True)
                    ans = {}
                else:
                    time.sleep(2 ** attempt)
        if len(ans) != len(payload):
            print(f"    PARTIAL: {len(ans)}/{len(payload)} items answered in this batch; "
                  "the rest are recorded judged:false, not as non-matches", flush=True)
        out = []
        for i, (t, cands, cids) in enumerate(items, 1):
            # `judged` separates "the judge considered this and chose no candidate" from
            # "no answer came back for this item". Both used to land as in_catalog:false,
            # so a failed or partially-parsed batch was recorded as ten hallucinations
            # and was indistinguishable from ten real non-matches. A verdict with
            # judged:false is missing data and must not be counted as a non-match.
            answered = i in ans
            v = ans.get(i, 0)
            out.append({"title": t, "in_catalog": bool(v), "judged": answered,
                        "match_title": cands[v - 1] if v else "",
                        "match_id": str(idx.ids[cids[v - 1]]) if v else "",
                        "match_quartile": str(idx.quart[cids[v - 1]]) if v else ""})
        with _lock:
            for o in out:
                fh.write(json.dumps(o) + "\n")
            fh.flush()
        return len(out)

    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    n = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for got in ex.map(do_batch, batches):
            n += got
            if n % 500 < BATCH:
                print(f"    {dataset}/{llm} {n}/{len(todo)}", flush=True)
    fh.close()


def main():
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    for d in DATASETS:
        idx = get_index(d)
        for l in LLMS:
            run_cell(client, d, l, idx)
    print("LLM MATCH ALL COMPLETE", flush=True)


if __name__ == "__main__":
    main()
