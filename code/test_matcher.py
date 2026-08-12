"""Regression suite for the catalog matcher.

Every case here broke a previous version of this code. The point is that no future
matcher can silently reintroduce a failure that has already cost us a rewrite.

Run: python code/test_matcher.py
"""
import pathlib
import re
import sys
sys.path.insert(0, "code")
import math
from surface_match import score_pair, fold, short_title_conflict

FAIL = []


def check(cond, msg):
    if not cond:
        FAIL.append(msg)


def expect(gen, cat, ds, want_in, why):
    s = score_pair(gen, cat, ds)
    got = (s == s) and s >= 90     # NaN is neither
    if got != want_in:
        FAIL.append(f"[{why}] {gen!r} vs {cat!r} ({ds}): got {s}, wanted "
                    f"{'IN' if want_in else 'OUT'}")


# --- BUG 1: token_set_ratio containment (the original published bug) -------------
expect("Melissa & Doug Suspend Family Game - Balancing Game with 24 Pieces",
       "Suspend game", "amazon", False, "bug1-containment")
expect("SplashEZ 3-in-1 Splash Pad, Sprinkler for Kids", "1", "amazon", False,
       "bug1-degenerate-entry")
expect("Dinosaur Toys for Kids 3-5, 12 Pack", "Dinosaur Toys for", "amazon", False,
       "bug1-truncated-stub")

# --- BUG 2: prefix_match hard-coded 95, i.e. bug 1 reimplemented -----------------
expect("Dumb and Dumber To", "Dumb and Dumber (1994)", "movielens", False,
       "bug2-sequel")
expect("The Original Pancake House", "Pancake House", "yelp", False, "bug2-prefix")
expect("Joe's New York Pizza", "New York Pizza", "yelp", False, "bug2-prefix")
expect("Toy Story 3 Special Edition", "Toy Story 3 (2010)", "movielens", False,
       "bug2-edition")
expect("Melissa & Doug Wooden Building Blocks Set, 100 Pieces",
       "Melissa & Doug Wooden Building Blocks", "amazon", False, "bug2-containment")

# --- BUG 3: diacritics zeroed the score ------------------------------------------
expect("Café Du Monde", "Cafe Du Monde", "yelp", True, "bug3-diacritic")
expect("Laughing Planet Café", "Laughing Planet Cafe", "yelp", True, "bug3-diacritic")
expect("Amélie (2001)", "Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001)",
       "movielens", True, "bug3-diacritic-alt-title")

# --- BUG 4: short-title guard keyed on max, so it flipped off inconsistently ------
expect("Blue Moon Pizza", "Moon Pizza", "yelp", False, "bug4-short-guard")
expect("The Publican", "The Republican", "yelp", False, "bug4-short-guard")

# --- Legitimate matches that must NOT regress to false negatives ------------------
expect("Se7en", "Seven (a.k.a. Se7en) (1995)", "movielens", True, "alt-title")
expect("Men in Black", "Men in Black (a.k.a. MIB) (1997)", "movielens", True, "alt-title")
expect("The Rock", "Rock, The (1996)", "movielens", True, "article-inversion")
expect("Independence Day (1996)", "Independence Day (a.k.a. ID4) (1996)", "movielens",
       True, "alt-title")
expect("Camellia Grill", "The Camellia Grill", "yelp", True, "leading-article")
expect("Walk-On's Sports Bistreaux", "Walk-On's Sports Bistreaux - New Orleans", "yelp",
       True, "location-suffix")
expect("Sabrina's Cafe", "Sabrina's Café", "yelp", True, "diacritic-reverse")

# --- Year discriminator ----------------------------------------------------------
expect("The Empire Strikes Back (1980)", "The Saint Strikes Back (1939)", "movielens",
       False, "year-discriminator")

# --- BUG 5: 0.0 meant BOTH "no match" and "unusable query" -----------------------
check(math.isnan(score_pair("", "anything", "yelp")),
      "bug5: empty query must return NaN, not 0.0 (else retrieval failure reads as "
      "a hallucination)")
check(score_pair("Totally Unrelated Thing", "Something Else Entirely", "yelp") == 0.0,
      "bug5: genuine non-match should be 0.0")

# --- Unicode folding is actually applied ------------------------------------------
check(fold("Café") == fold("Cafe"), "fold() must make accented and plain text equal")

# --- Short-title guard keys on the SHORTER side ------------------------------------
check(short_title_conflict("moon pizza", "blue moon pizza") is True,
      "short guard must apply when the shorter side is short")
check(short_title_conflict("a b c d e", "a b c d f") is False,
      "short guard must not apply to long titles")

# Retrieval depth. Three defects came from one consumer quietly using a different one:
# the published Table 1 scored a 20-candidate judge while deployment used 8, the blind
# relabel drew 4, and the first verification re-judge used 20 and reported a 19% flip
# rate that was measuring depth rather than fabrication.
from llm_matcher import DEPLOY_TOPN
check(DEPLOY_TOPN == 8, "deployed candidate depth must be 8")
for _f in ("llm_match_all.py", "build_round2.py", "verify_verdicts.py"):
    _src = (pathlib.Path("code") / _f).read_text()
    check("DEPLOY_TOPN" in _src, f"{_f} must slice to the shared depth constant")
    check("8" not in re.findall(r"\[:\s*(\d+)\s*\]", _src),
          f"{_f} still slices to a hard-coded 8; use DEPLOY_TOPN")

if FAIL:
    print(f"FAILED {len(FAIL)}:")
    for f in FAIL:
        print("  -", f)
    sys.exit(1)
print(f"matcher regression suite: all checks passed")
