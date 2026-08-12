"""Surface-form catalog matcher (v2, 2026-08-11).

REWRITTEN after a review found that v1 reproduced the very bug it was written to fix.

What v1 got wrong, and why this version is simpler:

  * v1's `prefix_match` awarded a hard-coded 95 whenever one token sequence was a
    contiguous subsequence of the other. That is the same asymmetric-containment
    inference as the original `token_set_ratio` bug, with a new constant that cleared
    the 90 threshold. It scored "Dumb and Dumber To" -> "Dumb and Dumber (1994)" as a
    match, which the project's own labeling instructions give as a worked example of
    NOT the same item. 43.4% of Amazon items counted as grounded came from that
    constant. The rule is removed outright; containment is not evidence of identity.
  * v1 compared accented and unaccented text as different tokens, so a single diacritic
    zeroed a match ("Cafe Du Monde" vs "Cafe Du Monde"). Text is now Unicode-folded.
  * v1's short-title guard keyed on max(len) so it switched OFF as soon as either side
    reached four tokens, giving opposite verdicts to the same structural relationship
    ("Blue Moon Pizza"/"Moon Pizza" blocked, "The Original Pancake House"/"Pancake
    House" matched). It now keys on min(len) and is applied consistently.
  * v1 returned 0.0 both for "no candidate matched" and for "empty query", so a
    retrieval failure was silently recorded as a hallucination. Empty queries now
    return NaN.

What remains is the part that is defensible: catalog entries are expanded into the
surface forms a model could plausibly emit (alternate titles recorded in parentheses,
trailing years, inverted articles, location suffixes), and scoring is plain
length-aware token_sort_ratio over those forms.

This matcher is deliberately CONSERVATIVE. It will miss same-item pairs where the
catalog entry carries extra material the generated title omits (series prefixes such as
"Star Wars: Episode V - ...", long Amazon titles with trailing marketing text). Those
false negatives are real and are measured against human labels rather than engineered
away, because every attempt to close them by rule in v1 introduced a larger and less
visible false-positive class.
"""
import re
import sys
import unicodedata
sys.path.insert(0, "code")
from utils import normalize_title

SHORT_TITLE_TOKENS = 3

_YEAR = re.compile(r"\s*\((\d{4})\)\s*$")
_PAREN = re.compile(r"\(([^)]*)\)")
_AKA = re.compile(r"^\s*a\.?k\.?a\.?\s*", re.I)
_ARTICLE_INV = re.compile(r"^(.*?),\s*(The|A|An)$", re.I)
_LOC_SUFFIX = re.compile(r"\s+-\s+[^-]+$")
_LEAD_ARTICLE = re.compile(r"^(the|a|an)\s+", re.I)


def fold(s: str) -> str:
    """Unicode-fold then normalize. 'Cafe' and 'Cafe' must be one token."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return normalize_title(s)


def surface_forms(title: str, dataset: str) -> list[str]:
    """Plausible surface forms of a catalog title, Unicode-folded."""
    t = str(title).strip()
    forms = {t}

    if dataset == "movielens":
        base = _YEAR.sub("", t)
        forms.add(base)
        for inner in _PAREN.findall(base):
            alt = _AKA.sub("", inner).strip()
            if len(alt) >= 2 and not alt.isdigit():
                forms.add(alt)
                m = _ARTICLE_INV.match(alt)
                if m:
                    forms.add(f"{m.group(2)} {m.group(1)}")
        stripped = _PAREN.sub("", base).strip()
        forms.add(stripped)
        m = _ARTICLE_INV.match(stripped)
        if m:
            forms.add(f"{m.group(2)} {m.group(1)}")

    elif dataset == "yelp":
        forms.add(_LEAD_ARTICLE.sub("", t))
        noloc = _LOC_SUFFIX.sub("", t)
        forms.add(noloc)
        forms.add(_LEAD_ARTICLE.sub("", noloc))

    else:  # amazon
        forms.add(_PAREN.sub("", t))
        forms.add(re.sub(r"\[[^\]]*\]", "", t))

    out = set()
    for f in forms:
        n = fold(f)
        if n:
            out.add(n)
    return sorted(out)


def sortkey(s: str) -> str:
    return " ".join(sorted(s.split()))


def get_year(t: str):
    m = _YEAR.search(str(t))
    return m.group(1) if m else None


def short_title_conflict(q_norm: str, form: str) -> bool:
    """For short titles, character similarity is unreliable ('The Publican' vs 'The
    Republican'), so require exact token agreement. Keyed on the SHORTER side so the
    rule applies consistently to the same structural relationship."""
    tq, tf = q_norm.split(), form.split()
    if not tq or not tf:
        return True
    if min(len(tq), len(tf)) > SHORT_TITLE_TOKENS:
        return False
    return set(tq) != set(tf)


def score_pair(query: str, cat_title: str, dataset: str) -> float:
    """Best match score of a generated title against one catalog entry.

    Returns NaN for an unusable query (so a retrieval/parse failure is never silently
    recorded as a hallucination) and 0.0 for a genuine non-match.
    """
    from rapidfuzz import fuzz

    qn = fold(query)
    if not qn:
        return float("nan")
    # MovieLens years are decisive when both sides carry one.
    if dataset == "movielens":
        qy, cy = get_year(query), get_year(cat_title)
        if qy and cy and qy != cy:
            return 0.0
    best = 0.0
    qs = sortkey(qn)
    for f in surface_forms(cat_title, dataset):
        if short_title_conflict(qn, f):
            continue
        s = float(fuzz.ratio(qs, sortkey(f)))
        if s > best:
            best = s
    return best
