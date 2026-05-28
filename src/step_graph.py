"""
Deterministic step counter that runs over the LLM-produced step_graph.

The LLM builds the structural decomposition (AND / OR / LEAF + class hint).
Python does the counting — that way the integer in result.csv is reproducible,
auditable, and falsifiable against a hand-traced reasoning of the policy.

The business rules (PA_Business_Rules.xlsx → Business Rules sheet) require:

  1. UNION universal-criteria branch with the indication-specific branch via AND.
  2. From the combined required steps, identify the LEAST-RESTRICTIVE OR path.
  3. Count only branded/biologic steps for "Number of Steps through Brands"
     (count only non-biologic + topical steps for "Number of Steps through Generic").
  4. Exclude phototherapy from both counts.
  5. Output "NA" when no steps of that target class are required at all.

Phototherapy field is "Yes" only if a PHOTOTHERAPY leaf is required AND
not under any OR ancestor.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from . import config


# ---------------------------------------------------------------------------
# Class normalization & whitelist re-classification
# ---------------------------------------------------------------------------
# Word-boundary regex match instead of naive substring containment. Two
# reasons the bare `in` was wrong:
#   1. "tar" (a coal-tar TOPICAL keyword) is a substring of "targeted",
#      "started", etc. — false positives that flipped biologic leaves to
#      TOPICAL.
#   2. CLAUDE.md rule #4 says the brand whitelist is the source of truth.
#      So we check biologic/generic FIRST, then fall back to TOPICAL /
#      PHOTOTHERAPY keyword heuristics.
def _make_token_re(tokens):
    parts = sorted({t.lower() for t in tokens}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in parts) + r")\b")


_BIOLOGIC_RE = _make_token_re(config.BRAND_WHITELIST_BIOLOGIC)
_GENERIC_RE = _make_token_re(config.BRAND_WHITELIST_GENERIC)
_TOPICAL_RE = _make_token_re(config.TOPICAL_KEYWORDS)
_PHOTOTHERAPY_RE = _make_token_re(config.PHOTOTHERAPY_KEYWORDS)


CLASSES = ("BRANDED_BIOLOGIC", "GENERIC_SYSTEMIC", "TOPICAL", "PHOTOTHERAPY", "OTHER")


def classify_drug_name(drug: str) -> str:
    """Map a drug or category name to one of the five classes via whitelist.

    Returns 'OTHER' when no whitelist hit; in that case reconcile_class
    falls back to the LLM's class field. Per CLAUDE.md rule #4, the brand
    whitelist is the source of truth — checked BEFORE the TOPICAL /
    PHOTOTHERAPY keyword heuristics, so a leaf naming a known biologic
    can't accidentally classify as TOPICAL via a substring collision.

    PHOTOTHERAPY-only refinement: when the LLM lumps a multi-option OR
    criterion into a single LEAF (real-world example from row 325611:
    "topical agent + systemic agent OR topical agent + phototherapy OR
    systemic agent + phototherapy OR 2 systemic agents OR ..."), the
    description contains "phototherapy" alongside "topical"/"systemic"
    keywords. Classifying that as PHOTOTHERAPY triggers a spurious -6
    score penalty. So we only return PHOTOTHERAPY when the description
    is a SINGLE-PURPOSE phototherapy step — i.e., no other class keywords
    are also present. Mixed leaves fall through to TOPICAL (or OTHER),
    which the counter treats as a generic step. Safer in the lumped case.
    """
    if not drug:
        return "OTHER"
    name = drug.lower()
    if _BIOLOGIC_RE.search(name):
        return "BRANDED_BIOLOGIC"
    if _GENERIC_RE.search(name):
        return "GENERIC_SYSTEMIC"
    has_photo = bool(_PHOTOTHERAPY_RE.search(name))
    has_topical = bool(_TOPICAL_RE.search(name))
    # Pure phototherapy leaf only — no topical alternatives in the same description.
    if has_photo and not has_topical:
        return "PHOTOTHERAPY"
    if has_topical:
        return "TOPICAL"
    return "OTHER"


def reconcile_class(node: Dict[str, Any]) -> str:
    """Use the whitelist to override the LLM's class field when it disagrees.

    The whitelist is the source of truth for any drug it covers; when it's
    silent (OTHER) we trust the LLM's hint.
    """
    drug = (node.get("drug_or_category") or "").strip()
    whitelist_class = classify_drug_name(drug)
    llm_class = (node.get("class") or "OTHER").upper()
    if whitelist_class != "OTHER":
        return whitelist_class
    if llm_class in CLASSES:
        return llm_class
    return "OTHER"


# ---------------------------------------------------------------------------
# Path-counts: single-walk that picks a coherent OR-path
# ---------------------------------------------------------------------------
# Earlier versions of this file walked the graph once per target class
# (BRAND, GENERIC, TOPICAL) and took the min independently inside each OR.
# That produced incoherent results: an OR with [A=(1 brand, 0 gen),
# B=(0 brand, 1 gen)] would resolve to "0 brand AND 0 gen" — i.e., the
# counter pretended the patient could satisfy both branches simultaneously.
#
# The fix: walk the graph ONCE, return a _Path tuple per branch, and at
# each OR pick the single child path that minimises a tie-break key.
# Tie-break order (from Reference-sheet worked example):
#   1. Fewest total steps (sum of brand+generic+topical+photo)
#   2. Fewest phototherapy steps (prefer paths the patient can satisfy
#      without UV/PUVA visits — Reference resolves photo-vs-generic to generic)
#   3. Fewest branded steps (generic is cheaper for the patient than brand)
@dataclass
class _Path:
    brand: int = 0
    generic: int = 0
    topical: int = 0
    photo: int = 0
    trace: List[str] = field(default_factory=list)

    @property
    def sort_key(self):
        total = self.brand + self.generic + self.topical + self.photo
        return (total, self.photo, self.brand)

    def __add__(self, other: "_Path") -> "_Path":
        return _Path(
            brand=self.brand + other.brand,
            generic=self.generic + other.generic,
            topical=self.topical + other.topical,
            photo=self.photo + other.photo,
            trace=self.trace + other.trace,
        )


def _walk(nodes: List[Dict[str, Any]]) -> _Path:
    """Recursive walk that returns the chosen path through any OR nodes.

    The tuple returned represents the count contribution of these nodes
    after collapsing every OR to its least-restrictive child path."""
    path = _Path()
    for n in nodes:
        logic = (n.get("logic") or "LEAF").upper()
        drug = n.get("drug_or_category") or "?"
        if logic == "LEAF":
            cls = reconcile_class(n)
            leaf = _Path()
            if cls == "BRANDED_BIOLOGIC":
                leaf.brand = 1
            elif cls == "GENERIC_SYSTEMIC":
                leaf.generic = 1
            elif cls == "TOPICAL":
                leaf.topical = 1
            elif cls == "PHOTOTHERAPY":
                leaf.photo = 1
            tag = "+1" if (leaf.brand or leaf.generic or leaf.topical or leaf.photo) else ""
            leaf.trace.append(f"  LEAF {drug!r} → {cls} {tag}".rstrip())
            path = path + leaf
        elif logic == "AND":
            sub = _walk(n.get("children", []))
            path.trace.append("  AND-block (sum children):")
            path = path + sub
        elif logic == "OR":
            children = n.get("children", []) or []
            if not children:
                continue
            sub_paths = [_walk([ch]) for ch in children]
            best_idx = min(range(len(sub_paths)),
                           key=lambda i: sub_paths[i].sort_key)
            path.trace.append("  OR-block (least restrictive — single path chosen):")
            for i, sp in enumerate(sub_paths):
                mark = "✓" if i == best_idx else " "
                path.trace.append(
                    f"    {mark} branch {i+1}: "
                    f"brand={sp.brand} gen={sp.generic} top={sp.topical} photo={sp.photo}"
                )
            path = path + sub_paths[best_idx]
        else:
            # Unknown logic — treat as LEAF
            cls = reconcile_class(n)
            leaf = _Path()
            if cls == "BRANDED_BIOLOGIC":
                leaf.brand = 1
            elif cls == "GENERIC_SYSTEMIC":
                leaf.generic = 1
            elif cls == "TOPICAL":
                leaf.topical = 1
            path = path + leaf
    return path


def _has_mandatory_phototherapy(nodes: List[Dict[str, Any]],
                                 under_or: bool = False) -> bool:
    """Walk the tree; return True iff a PHOTOTHERAPY leaf is required and
    not under any OR ancestor."""
    for n in nodes:
        logic = (n.get("logic") or "LEAF").upper()
        if logic == "LEAF":
            if reconcile_class(n) == "PHOTOTHERAPY" and not under_or and bool(
                n.get("is_mandatory", True)
            ):
                return True
        elif logic == "AND":
            if _has_mandatory_phototherapy(n.get("children", []), under_or):
                return True
        elif logic == "OR":
            if _has_mandatory_phototherapy(n.get("children", []), True):
                return True
    return False


def _has_any_phototherapy(nodes: List[Dict[str, Any]]) -> bool:
    for n in nodes:
        logic = (n.get("logic") or "LEAF").upper()
        if logic == "LEAF":
            if reconcile_class(n) == "PHOTOTHERAPY":
                return True
        elif _has_any_phototherapy(n.get("children", []) or []):
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass
class CountResult:
    brands: Any        # int or "NA"
    generics: Any      # int or "NA"
    phototherapy: str  # "Yes" | "No" | "NA"
    brand_trace: List[str]
    generic_trace: List[str]
    photo_present: bool


def count_steps(step_graph: Dict[str, Any]) -> CountResult:
    """Given a step_graph (universal_branch + indication_branch), apply the
    business rules and return Number of Steps through Brands, Generic, and
    the Phototherapy verdict."""
    universal = step_graph.get("universal_branch") or []
    indication = step_graph.get("indication_branch") or []
    # If both branches are empty, no steps at all — counts are "NA" and
    # phototherapy is "NA" too (per Business Rules).
    if not universal and not indication:
        return CountResult("NA", "NA", "NA", [], [], False)

    # Treat the union as AND of the two branches (per Business Rules).
    combined = [{
        "logic": "AND",
        "children": universal + indication,
        "drug_or_category": "",
        "class": "OTHER",
        "is_mandatory": True,
    }]

    chosen = _walk(combined)
    brand_count = chosen.brand
    generic_count = chosen.generic + chosen.topical  # topicals count as generic
    # Same single trace describes both — we walked once and chose one path.
    brand_trace = chosen.trace
    generic_trace = chosen.trace

    photo_present = _has_any_phototherapy(combined)
    photo_mandatory = _has_mandatory_phototherapy(combined)

    brands_out: Any = "NA" if brand_count == 0 else brand_count
    generics_out: Any = "NA" if generic_count == 0 else generic_count

    if photo_mandatory:
        photo_out = "Yes"
    elif photo_present:
        photo_out = "No"  # present but in OR alternative → not mandatory
    else:
        photo_out = "No" if (universal or indication) else "NA"

    return CountResult(
        brands=brands_out,
        generics=generics_out,
        phototherapy=photo_out,
        brand_trace=brand_trace,
        generic_trace=generic_trace,
        photo_present=photo_present,
    )


