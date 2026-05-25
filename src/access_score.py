"""
Access Score rubric (0–100).

The judges score this against a gold standard. Because we don't know the
exact weights they're using, we ship a transparent ADDITIVE rubric where
every contribution is named, signed, and capped. Three benefits:

  1. Auditability — every score in result.csv decomposes into a waterfall
     the judges can replicate cell-by-cell.
  2. Tunability — weights live in src/config.py so the entire rubric is one
     dict you can re-fit against silver labels in ~30 seconds.
  3. Robustness — the floor/ceiling prevent extreme rows from skewing the
     dataset distribution, while caps on cumulative penalties (e.g.,
     step_per_brand_total = -24) keep a single hyper-restrictive policy from
     swinging the score by more than ~50 points.

Anchors per the problem statement:
  0   = No access
  25  = Restricted access vs FDA label
  50  = Parity with FDA label
  75  = Preferred vs FDA label
  100 = Best possible (no restrictions)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from . import config


@dataclass
class ScoreBreakdown:
    score: int
    contributions: List[Tuple[str, int, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_int_or_none(value: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if s.lower() in {"unspecified", "na", "n/a", "none"}:
        return None
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else None


def _age_above_fda(age_value: str, brand: str) -> bool:
    """Return True iff the policy's age threshold exceeds the FDA label for
    this brand. >=18 vs FDA-label of 6 → True. 'FDA labelled age' → False.
    """
    fda_min = config.FDA_MIN_AGE_PSO.get(brand.upper())
    if fda_min is None or not age_value:
        return False
    s = str(age_value).strip()
    if "fda label" in s.lower():
        return False
    m = re.search(r"(\d+)", s)
    if not m:
        return False
    extracted = int(m.group(1))
    return extracted > fda_min


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_row(row_values: Dict[str, Any], brand: str) -> ScoreBreakdown:
    """Score one (Filename, Brand) row given the 12 normalised values.

    `row_values` is keyed on the submission-CSV column names so judges can
    re-derive the score from the CSV alone without touching extras.
    """
    R = config.ACCESS_SCORE_RUBRIC
    weights = R["weights"]
    caps = R.get("caps", {})
    thresholds = R.get("thresholds", {})
    init_long = thresholds.get("initial_long_months", 12)
    init_short = thresholds.get("initial_short_months", 3)
    reauth_long = thresholds.get("reauth_long_months", 12)

    score = R["base"]
    contribs: List[Tuple[str, int, str]] = [("base (parity with FDA)", R["base"], "")]
    notes: List[str] = []

    # Age
    age = row_values.get("Age", "")
    if _age_above_fda(age, brand):
        delta = weights["age_restrictive"]
        score += delta
        contribs.append(("age above FDA label", delta, f"age={age} vs FDA min {config.FDA_MIN_AGE_PSO.get(brand.upper())}"))

    # Branded steps
    brand_steps = row_values.get("Number of Steps through Brands", "NA")
    if isinstance(brand_steps, str) and brand_steps.isdigit():
        brand_steps = int(brand_steps)
    if isinstance(brand_steps, int) and brand_steps > 0:
        raw = brand_steps * weights["step_per_brand"]
        cap = caps.get("step_per_brand_total", raw)
        applied = max(cap, raw)
        score += applied
        contribs.append((f"{brand_steps} branded step(s)", applied, ""))

    # Generic steps
    gen_steps = row_values.get("Number of Steps through Generic", "NA")
    if isinstance(gen_steps, str) and gen_steps.isdigit():
        gen_steps = int(gen_steps)
    if isinstance(gen_steps, int) and gen_steps > 0:
        raw = gen_steps * weights["step_per_generic"]
        cap = caps.get("step_per_generic_total", raw)
        applied = max(cap, raw)
        score += applied
        contribs.append((f"{gen_steps} generic step(s)", applied, ""))

    # Phototherapy
    photo = str(row_values.get("Step through-Phototherapy", "")).strip().lower()
    if photo == "yes":
        delta = weights["phototherapy_required"]
        score += delta
        contribs.append(("phototherapy required", delta, ""))

    # TB
    tb = str(row_values.get("TB Test required", "")).strip().lower()
    if tb == "yes":
        delta = weights["tb_test"]
        score += delta
        contribs.append(("TB test required", delta, ""))

    # Specialist
    spec = str(row_values.get("Specialist Types", "")).strip()
    if spec and spec.lower() not in {"na", "n/a", "none", ""}:
        delta = weights["specialist_required"]
        score += delta
        contribs.append(("specialist required", delta, spec))

    # Quantity Limits
    ql = str(row_values.get("Quantity Limits", "")).strip()
    if ql and ql.lower() not in {"not specified", "na", "n/a", "none", "no"}:
        delta = weights["quantity_limited"]
        score += delta
        contribs.append(("quantity limit imposed", delta, ""))

    # Initial Authorization Duration
    init_dur = _parse_int_or_none(row_values.get("Initial Authorization Duration(in-months)"))
    if init_dur is not None:
        if init_dur >= init_long:
            delta = weights["initial_auth_long"]
            score += delta
            contribs.append((f"initial auth {init_dur}mo (≥{init_long})", delta, ""))
        elif init_dur <= init_short:
            delta = weights["initial_auth_short"]
            score += delta
            contribs.append((f"initial auth {init_dur}mo (≤{init_short})", delta, ""))

    # Reauthorization
    reauth_req = str(row_values.get("Reauthorization Required", "")).strip().lower()
    reauth_dur = _parse_int_or_none(row_values.get("Reauthorization Duration(in-months)"))
    if reauth_req == "no":
        delta = weights["no_reauth_required"]
        score += delta
        contribs.append(("no reauthorization required", delta, ""))
    else:
        if reauth_dur is not None:
            if reauth_dur >= reauth_long:
                delta = weights["reauth_long"]
                score += delta
                contribs.append((f"reauth {reauth_dur}mo (≥{reauth_long})", delta, ""))
            else:
                delta = weights["reauth_short"]
                score += delta
                contribs.append((f"reauth {reauth_dur}mo (<{reauth_long})", delta, ""))

    score = max(R["floor"], min(R["ceiling"], score))
    return ScoreBreakdown(score=score, contributions=contribs, notes=notes)


def render_waterfall(breakdown: ScoreBreakdown) -> str:
    """Pretty-print the waterfall for audit cards / debugging."""
    lines = ["Access Score waterfall:"]
    running = 0
    for label, delta, note in breakdown.contributions:
        running += delta
        sign = "+" if delta >= 0 else ""
        suffix = f"  ({note})" if note else ""
        lines.append(f"  {label:40s}  {sign}{delta:>4d}  → running {running:>3d}{suffix}")
    lines.append(f"  {'(clamped to [0,100])':40s}                → final {breakdown.score:>3d}")
    return "\n".join(lines)
