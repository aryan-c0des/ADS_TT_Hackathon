"""
Brand-section isolation.

A given (Filename, Brand) row only cares about the slice of the policy text
that governs *that* brand for plaque psoriasis. Sending the whole text to the
LLM either dilutes the prompt or burns quota — so we localise first.

Three observed layouts:

  (a) Single-drug policy        — full text is the slice.
  (b) Multi-drug class policy   — slice by brand heading until next brand.
  (c) Medicaid mega-formulary   — anchor on a brand-name + dosage-form regex.

For corner cases (no anchor, slice too short or too long) we delegate to the
LLM to return a (start_anchor, end_anchor) pair so Python can locate the slice.
The slice always ends with the universal "all-indications" block tagged
[UNIVERSAL ...] when one exists — the business rules require us to UNION it
with the indication-specific criteria.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from . import config


# All PsO-related brand tokens we might find in a policy; used for heading detection
PSO_BRAND_TOKENS = sorted(
    set(config.BRAND_CANONICAL.values()) |
    {b.upper() for b in config.BRAND_WHITELIST_BIOLOGIC if len(b) > 4} |
    {"TREMFYA", "STELARA", "SKYRIZI", "COSENTYX", "ENBREL", "OTEZLA",
     "HUMIRA", "TALTZ", "ILUMYA", "CIMZIA", "BIMZELX", "SOTYKTU",
     "REMICADE", "INFLECTRA", "RENFLEXIS", "AMJEVITA", "OTULFI",
     "YESINTEK", "SILIQ"}
)


@dataclass
class BrandSegment:
    """Output of segmentation for one (Filename, Brand) row."""
    filename: str
    brand: str
    layout: str                  # "single_drug" | "multi_drug" | "mega_formulary"
    text: str                    # the isolated slice + universal block
    universal_block: str         # the universal criteria appended (may be "")
    char_span: Tuple[int, int]   # original-text indices the slice spans
    used_llm_fallback: bool

    def cache_path(self):
        from pathlib import Path
        stem = Path(self.filename).stem
        return config.SEGMENT_CACHE / f"{stem}__{self.brand}.txt"


def detect_layout(full_text: str, brand: str) -> str:
    """Decide which of the three layout strategies applies.

    The signal we trust most is: how often is the *target* brand mentioned
    relative to other PsO brands? When it dominates we keep the full text
    (single_drug); when many distinct brands all get heading-level treatment
    we slice (multi_drug); when the doc is massive we anchor-slice
    (mega_formulary).
    """
    n = len(full_text)
    upper = full_text.upper()
    target = brand.upper()

    target_count = len(re.findall(rf"(?i)\b{re.escape(target)}\b", full_text))
    other_brand_headings = sum(
        1 for b in PSO_BRAND_TOKENS
        if b != target and re.search(rf"(?m)^\s*{re.escape(b)}\b", upper)
    )

    if n >= config.LARGE_PDF_TEXT_THRESHOLD and target_count < 20:
        return "mega_formulary"
    if target_count >= 5 and target_count > other_brand_headings:
        return "single_drug"
    if other_brand_headings >= 4:
        return "multi_drug"
    if target_count >= 1:
        return "single_drug"
    return "multi_drug"


def _find_universal_block(text: str) -> Tuple[str, int, int]:
    """Locate a universal/all-indications block, if present, returning the
    block text and its (start, end) span in `text`. ('', -1, -1) if absent.
    """
    patterns = [
        r"(?im)^[\s]*Documentation\s+for\s+all\s+indications.*?(?=\n[A-Z][^\n]{0,80}\n|\Z)",
        r"(?im)^[\s]*UNIVERSAL\s+CRITERIA.*?(?=\n[A-Z][^\n]{0,80}\n|\Z)",
        r"(?im)^[\s]*Criteria\s+for\s+all\s+indications.*?(?=\n[A-Z][^\n]{0,80}\n|\Z)",
        r"(?im)^[\s]*ALL\s+INDICATIONS.*?(?=\n[A-Z][^\n]{0,80}\n|\Z)",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.DOTALL)
        if m:
            return m.group(0).strip(), m.start(), m.end()
    return "", -1, -1


def _slice_around(text: str, start: int, end: int, radius: int) -> Tuple[str, int, int]:
    s = max(0, start - radius)
    e = min(len(text), end + radius)
    return text[s:e], s, e


def _segment_single_drug(text: str) -> Tuple[str, str, Tuple[int, int]]:
    # Use full text; still pull out universal block as its own block so the
    # extractor can tell what was universal vs indication-specific.
    u, _, _ = _find_universal_block(text)
    return text, u, (0, len(text))


def _segment_multi_drug(text: str, brand: str) -> Tuple[str, str, Tuple[int, int]]:
    """Slice around the brand occurrence with the most psoriasis context nearby.

    Cheap heuristic: for each occurrence of the target brand in the text,
    score it by counting psoriasis-related terms within ±2000 chars. Pick
    the highest-scoring occurrence and widen ±3000 chars. This avoids the
    failure mode where the brand appears in a tabular preferred-drug
    listing at the top of the doc but the real criteria sit downstream.
    """
    target = brand.upper()
    upper = text.upper()
    brand_positions = [m.start() for m in re.finditer(rf"\b{re.escape(target)}\b", upper)]
    if not brand_positions:
        return text[: config.SEGMENT_MAX_CHARS], "", (0, min(config.SEGMENT_MAX_CHARS, len(text)))

    pso_pattern = re.compile(r"(?i)psoriasis|plaque|pso\b|moderate to severe")
    pso_positions = [m.start() for m in pso_pattern.finditer(text)]

    def score(pos: int, radius: int = 2000) -> int:
        lo, hi = pos - radius, pos + radius
        return sum(1 for p in pso_positions if lo <= p <= hi)

    best = max(brand_positions, key=score)
    radius = 3000
    slice_text, s, e = _slice_around(text, best, best + len(target), radius)
    u, _, _ = _find_universal_block(text)
    return slice_text, u, (s, e)


def _segment_mega_formulary(text: str, brand: str) -> Tuple[str, str, Tuple[int, int]]:
    """Find a brand+dosage-form anchor and widen ±SEGMENT_DEFAULT_RADIUS.

    Falls back to a broader brand-only search if the strict anchor misses.
    """
    target = brand
    strict = re.compile(
        rf"(?i)\b{re.escape(target)}\b[\s\W]{{0,40}}(?:tablet|injection|syringe|capsule|vial|cream|gel|kit|pen)",
    )
    m = strict.search(text)
    if m is None:
        # Try near "psoriasis"
        loose = re.compile(rf"(?i)\b{re.escape(target)}\b")
        anchors = [hit.start() for hit in loose.finditer(text)]
        psor = [hit.start() for hit in re.finditer(r"(?i)psoriasis|plaque", text)]
        if not anchors:
            return text[:config.SEGMENT_MAX_CHARS], "", (0, min(config.SEGMENT_MAX_CHARS, len(text)))
        if psor:
            start = min(anchors, key=lambda p: min(abs(p - x) for x in psor))
        else:
            start = anchors[0]
        end = start + 1
    else:
        start, end = m.start(), m.end()
    slice_text, s, e = _slice_around(text, start, end, config.SEGMENT_DEFAULT_RADIUS)
    u, _, _ = _find_universal_block(text)
    return slice_text, u, (s, e)


_INDICATION_HEADING = re.compile(
    r"(?im)^[\s\d•\-\.\)]*"
    r"(plaque\s+psoriasis|psoriasis\s*\(?ps[oO]\)?|psoriatic\s+arthritis|"
    r"ulcerative\s+colitis|crohn'?s?\s+disease|ankylosing\s+spondylitis|"
    r"rheumatoid\s+arthritis|hidradenitis\s+suppurativa|nr-axspa|"
    r"juvenile\s+idiopathic\s+arthritis|atopic\s+dermatitis)"
)


def _focus_pso_indication(slice_text: str) -> str:
    """Locate the PsO indication section inside the brand slice and keep it
    (with reauth/duration following). If only a single indication is
    described or no headings parse, return the slice unchanged so we keep
    full context for downstream extraction."""
    matches = list(_INDICATION_HEADING.finditer(slice_text))
    if len(matches) < 2:
        return slice_text
    pso_match = None
    for m in matches:
        head = m.group(1).lower()
        if "psoriasis" in head and "psoriatic" not in head:
            pso_match = m
            break
    if pso_match is None:
        return slice_text
    next_starts = [m.start() for m in matches if m.start() > pso_match.end()]
    end = next_starts[0] if next_starts else len(slice_text)
    # Preserve the brand-level header (first 1200 chars) so brand identity,
    # quantity-limit tables and authorization-duration callouts up top survive.
    head = slice_text[: min(1200, pso_match.start())]
    body = slice_text[pso_match.start():end]
    tail_start = end
    tail = slice_text[tail_start: min(len(slice_text), tail_start + 800)]
    return head + "\n\n" + body + "\n\n" + tail


def segment(filename: str, brand: str, full_text: str) -> BrandSegment:
    """Run heuristic segmentation, with optional LLM fallback when slice
    quality is poor. The actual LLM call is delegated to llm_client.locate
    if needed; here we just produce the heuristic result.
    """
    layout = detect_layout(full_text, brand)
    if layout == "single_drug":
        slice_text, universal, span = _segment_single_drug(full_text)
    elif layout == "multi_drug":
        slice_text, universal, span = _segment_multi_drug(full_text, brand)
    else:
        slice_text, universal, span = _segment_mega_formulary(full_text, brand)

    slice_text = _focus_pso_indication(slice_text)
    universal_block = universal or ""
    combined_text = slice_text
    if universal_block and universal_block not in combined_text:
        combined_text = (
            slice_text.rstrip()
            + "\n\n[UNIVERSAL CRITERIA (all indications)]\n"
            + universal_block
        )

    return BrandSegment(
        filename=filename,
        brand=brand,
        layout=layout,
        text=combined_text,
        universal_block=universal_block,
        char_span=span,
        used_llm_fallback=False,
    )


def needs_llm_fallback(seg: BrandSegment) -> bool:
    """Heuristic slice is suspect — flag for an LLM-assisted re-locate.

    Single-drug policies can legitimately run to 80K chars; only flag those
    when the slice is implausibly small. Multi-drug and mega-formulary
    slices are bounded both below and above.
    """
    n = len(seg.text)
    if seg.layout == "single_drug":
        return n < config.SEGMENT_MIN_CHARS
    return n < config.SEGMENT_MIN_CHARS or n > config.SEGMENT_MAX_CHARS


def save_segment(seg: BrandSegment) -> None:
    seg.cache_path().write_text(seg.text, encoding="utf-8")


def load_cached_segment(filename: str, brand: str) -> str | None:
    from pathlib import Path
    stem = Path(filename).stem
    path = config.SEGMENT_CACHE / f"{stem}__{brand}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None
