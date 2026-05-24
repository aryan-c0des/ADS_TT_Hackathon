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

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from . import config


# ---------------------------------------------------------------------------
# Class normalization & whitelist re-classification
# ---------------------------------------------------------------------------
_BIOLOGIC_TOKENS = {b.lower() for b in config.BRAND_WHITELIST_BIOLOGIC}
_GENERIC_TOKENS = {b.lower() for b in config.BRAND_WHITELIST_GENERIC}
_TOPICAL_TOKENS = config.TOPICAL_KEYWORDS
_PHOTOTHERAPY_TOKENS = config.PHOTOTHERAPY_KEYWORDS


CLASSES = ("BRANDED_BIOLOGIC", "GENERIC_SYSTEMIC", "TOPICAL", "PHOTOTHERAPY", "OTHER")


def classify_drug_name(drug: str) -> str:
    """Map a drug or category name to one of the five classes via whitelist.

    Returns 'OTHER' when no whitelist hit; in that case we still trust the
    LLM's class field as the source of truth (downgrading only when the LLM
    says BRANDED_BIOLOGIC for something that's clearly generic, or vice versa).
    """
    if not drug:
        return "OTHER"
    name = drug.lower()
    # Specific keyword checks before brand-token fuzzy match
    if any(k in name for k in _PHOTOTHERAPY_TOKENS):
        return "PHOTOTHERAPY"
    if any(k in name for k in _TOPICAL_TOKENS):
        return "TOPICAL"
    if any(tok in name for tok in _BIOLOGIC_TOKENS):
        return "BRANDED_BIOLOGIC"
    if any(tok in name for tok in _GENERIC_TOKENS):
        return "GENERIC_SYSTEMIC"
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
# Recursive counters
# ---------------------------------------------------------------------------
def _walk_nodes(nodes: List[Dict[str, Any]], target: str, under_or: bool,
                trace: List[str]) -> int:
    total = 0
    for n in nodes:
        logic = (n.get("logic") or "LEAF").upper()
        if logic == "LEAF":
            cls = reconcile_class(n)
            hit = (cls == target)
            if hit:
                total += 1
                trace.append(f"  LEAF {n.get('drug_or_category','?')!r} → {cls} (+1)")
            else:
                trace.append(f"  LEAF {n.get('drug_or_category','?')!r} → {cls}")
        elif logic == "AND":
            trace.append("  AND-block (sum children):")
            total += _walk_nodes(n.get("children", []), target, under_or, trace)
        elif logic == "OR":
            children = n.get("children", []) or []
            if not children:
                continue
            sub_traces = [[] for _ in children]
            sub_counts: List[int] = []
            for ch, tr in zip(children, sub_traces):
                sub_counts.append(_walk_nodes([ch], target, True, tr))
            best = min(sub_counts)
            best_idx = sub_counts.index(best)
            trace.append("  OR-block (least restrictive):")
            for i, (cnt, tr) in enumerate(zip(sub_counts, sub_traces)):
                mark = "✓" if i == best_idx else " "
                trace.append(f"    {mark} branch {i+1} = {cnt}")
                trace.extend("      " + s for s in tr)
            total += best
        else:
            # Unknown logic — treat as LEAF
            cls = reconcile_class(n)
            if cls == target:
                total += 1
    return total


def _has_mandatory_phototherapy(nodes: List[Dict[str, Any]], under_or: bool) -> bool:
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

    brand_trace: List[str] = []
    brand_count = _walk_nodes(combined, "BRANDED_BIOLOGIC", False, brand_trace)
    generic_trace: List[str] = []
    generic_count = _walk_nodes(combined, "GENERIC_SYSTEMIC", False, generic_trace)
    # Topicals count as generic per business rules
    topical_trace: List[str] = []
    topical_count = _walk_nodes(combined, "TOPICAL", False, topical_trace)
    generic_count += topical_count
    generic_trace.extend(topical_trace)

    photo_present = _has_any_phototherapy(combined)
    photo_mandatory = _has_mandatory_phototherapy(combined, False)

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


def trace_to_str(traces: List[str]) -> str:
    return "\n".join(traces) if traces else "(empty)"
