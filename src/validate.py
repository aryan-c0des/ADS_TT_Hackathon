"""
Business-rule cross-checks that run AFTER the LLM has returned values but
BEFORE we write to result.csv.

Most violations are auto-fixable (e.g., reauth_required must be 'Yes' when a
duration is present). A few warrant a targeted re-prompt — those return a
'NEEDS_REPROMPT' flag with the field-set to re-ask.

The validator is deliberately conservative: it never overwrites the LLM's
extracted value unless it has a clear deterministic rule, and it always
records the reason in `violations` so the audit card can show it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ValidationOutcome:
    fixed: Dict[str, Any] = field(default_factory=dict)   # field → new value
    flags: List[str] = field(default_factory=list)        # warnings for audit
    needs_reprompt: List[str] = field(default_factory=list)  # which prompt groups


# ---------------------------------------------------------------------------
# Normalisers
# ---------------------------------------------------------------------------
def _normalise_age(value: str) -> str:
    if not value:
        return "No"
    v = value.strip()
    if v.lower() in {"no", "na", "n/a", "none"}:
        return "No"
    if re.search(r"(?i)fda\s+label", v):
        return "FDA labelled age"
    m = re.search(r"(\d+)", v)
    if m:
        return f">={m.group(1)}"
    return v


def _normalise_yes_no(value: str, default: str = "No") -> str:
    if not value:
        return default
    v = value.strip().lower()
    if v in {"y", "yes", "true", "1"}:
        return "Yes"
    if v in {"n", "no", "false", "0"}:
        return "No"
    return default


def _normalise_months(value: str) -> str:
    """Coerce free-form duration strings to integer months.

    Day-strings are ALWAYS converted to months (rounded). The old version
    only converted when n >= 60, which silently left '30 days' as '30'
    (interpreted as 30 months downstream). Year-strings are also converted.
    """
    if not value:
        return "Unspecified"
    v = str(value).strip()
    if v.lower() in {"na", "n/a", "none", ""}:
        return "NA"
    if v.lower() == "unspecified":
        return "Unspecified"
    m = re.search(r"(\d+)", v)
    if not m:
        return "Unspecified"
    n = int(m.group(1))
    vl = v.lower()
    if "day" in vl:
        n = max(1, round(n / 30))
    elif "year" in vl:
        n = n * 12
    elif "week" in vl:
        n = max(1, round(n / 4.345))
    return str(n)


# ---------------------------------------------------------------------------
# Cross-checks
# ---------------------------------------------------------------------------
def validate(extracted, count_result) -> ValidationOutcome:
    """Run all cross-checks on a single row's extracted data.

    Returns the fixed values (key → corrected value) and a list of human-
    readable violation flags for the audit card.
    """
    o = ValidationOutcome()
    s = extracted.scalars
    tf = extracted.text_fields
    st = extracted.step_data

    # ---- Age normalisation ----
    age_raw = (s.get("age") or {}).get("value", "")
    o.fixed["Age"] = _normalise_age(age_raw)

    # ---- TB ----
    tb = (s.get("tb_test_required") or {}).get("value", "")
    o.fixed["TB Test required"] = _normalise_yes_no(tb)

    # ---- Auth durations ----
    init_dur = (s.get("initial_authorization_duration_months") or {}).get("value", "")
    reauth_dur = (s.get("reauthorization_duration_months") or {}).get("value", "")
    o.fixed["Initial Authorization Duration(in-months)"] = _normalise_months(init_dur)
    o.fixed["Reauthorization Duration(in-months)"] = _normalise_months(reauth_dur)

    # ---- Reauth Required cross-check ----
    reauth_req = (s.get("reauthorization_required") or {}).get("value", "")
    reauth_req_norm = _normalise_yes_no(reauth_req)
    reauth_reqs_text = ((tf.get("reauthorization_requirements") or {}).get("value", "") or "").strip()
    reauth_dur_norm = o.fixed["Reauthorization Duration(in-months)"]
    if reauth_dur_norm not in {"NA", ""} or (reauth_reqs_text and reauth_reqs_text.upper() not in {"NA", "N/A"}):
        if reauth_req_norm != "Yes":
            o.flags.append(
                f"Reauthorization Required auto-flipped to Yes "
                f"(duration='{reauth_dur_norm}', requirements_text_len={len(reauth_reqs_text)})"
            )
            reauth_req_norm = "Yes"
    else:
        # No duration, no requirements text → enforce No
        if reauth_req_norm == "Yes":
            o.flags.append("Reauthorization Required asserted Yes but no duration/criteria present; left as Yes for audit.")
    o.fixed["Reauthorization Required"] = reauth_req_norm

    # ---- Step therapy text + counts ----
    step_text = (st.get("step_therapy_text") or "").strip()
    o.fixed["Step Therapy Requirements Documented in Policy"] = step_text or "NA"
    o.fixed["Number of Steps through Brands"] = count_result.brands
    o.fixed["Number of Steps through Generic"] = count_result.generics
    o.fixed["Step through-Phototherapy"] = count_result.phototherapy

    if (isinstance(count_result.brands, int) and count_result.brands > 0
            and not step_text):
        o.flags.append("step count > 0 but step_therapy_text is empty — possible mis-decomposition.")
        o.needs_reprompt.append("step_therapy")

    if count_result.phototherapy == "Yes" and "phototherap" not in step_text.lower():
        o.flags.append("phototherapy=Yes but step text doesn't mention phototherapy — verify.")
        o.needs_reprompt.append("step_therapy")

    # ---- Reauth requirements text ----
    if reauth_reqs_text:
        o.fixed["Reauthorization Requirements Documented in Policy"] = reauth_reqs_text
    else:
        o.fixed["Reauthorization Requirements Documented in Policy"] = "NA"

    # ---- Specialist ----
    spec_raw = ((tf.get("specialist_types") or {}).get("value", "") or "").strip()
    o.fixed["Specialist Types"] = spec_raw or "NA"

    # ---- Quantity Limits ----
    ql_raw = ((tf.get("quantity_limits") or {}).get("value", "") or "").strip()
    ql_evidence = ((tf.get("quantity_limits") or {}).get("evidence", "") or "").lower()
    if ql_raw and ql_raw.lower() not in {"not specified", "na", "n/a", "none"}:
        # Trust only when evidence contains 'quantity limit' / 'QL' / 'limited to'
        ok = any(tok in ql_evidence for tok in ("quantity limit", "ql:", "limited to", "quantity level"))
        if not ok and "limit" not in ql_raw.lower():
            o.flags.append("quantity_limits value present but evidence doesn't support 'quantity limit' label; coerced to 'Not specified'.")
            ql_raw = "Not specified"
    else:
        ql_raw = "Not specified"
    o.fixed["Quantity Limits"] = ql_raw

    return o
