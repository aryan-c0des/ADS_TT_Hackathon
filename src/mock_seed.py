"""
Seed the LLM cache with plausible synthetic responses for ALL 79 rows so
the pipeline can be exercised end-to-end without a Groq API key.

This is used for:
  - the offline smoke test
  - judge re-runs when their Groq quota is depleted
  - regression testing of step_graph, validate, access_score

The seeded responses are deliberately conservative and uniform — they
exercise every code path but are NOT a substitute for real LLM extraction.
The README states this explicitly. Once the user runs with a real API key,
the cache is overwritten with the LLM's actual responses.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict

from . import (
    config,
    extract_params,
    extract_text,
    ingest,
    llm_client,
    segment_brand,
)


def _seed_for_row(filename: str, brand: str) -> None:
    text = extract_text.load_text(filename)
    seg = segment_brand.segment(filename, brand, text)
    seg_text = seg.text

    # Heuristic-only "extraction" from the segment text. This is intentionally
    # weak — the real LLM will do far better. But it's enough to keep the
    # pipeline running end-to-end and exercise every downstream branch.
    seg_lower = seg_text.lower()

    age_match = re.search(r"(\d{1,2})\s*(?:years?\s*(?:of\s*age)?|y/o)\s*(?:and\s*)?(?:older|or\s*older|\+)?", seg_text, re.I)
    if age_match:
        age_val = f">={age_match.group(1)}"
    elif "adult" in seg_lower or "fda" in seg_lower:
        age_val = "FDA labelled age"
    else:
        age_val = "No"

    tb_val = "Yes" if re.search(r"(?i)tuberculosis|\\btb\\b", seg_text) else "No"

    init_match = re.search(r"(?i)(?:initial|first|approval)[^.]{0,80}?(\d{1,2})\s*(?:months?|days?)", seg_text)
    init_val = init_match.group(1) if init_match else "Unspecified"

    reauth_match = re.search(r"(?i)(?:reauth\w*|continuation|renewal)[^.]{0,80}?(\d{1,2})\s*(?:months?|days?)", seg_text)
    reauth_val = reauth_match.group(1) if reauth_match else "Unspecified"

    reauth_req = "Yes" if reauth_match or re.search(r"(?i)reauthor|continuation criteria|renewal", seg_text) else "No"

    scalars = {
        "age": {"value": age_val, "evidence": (age_match.group(0) if age_match else "")},
        "tb_test_required": {"value": tb_val, "evidence": ""},
        "initial_authorization_duration_months": {"value": init_val, "evidence": (init_match.group(0) if init_match else "")},
        "reauthorization_duration_months": {"value": reauth_val, "evidence": (reauth_match.group(0) if reauth_match else "")},
        "reauthorization_required": {"value": reauth_req, "evidence": ""},
    }

    # Very rough step heuristic: count distinct branded biologic mentions in
    # the slice as a proxy for branded-step pressure, and count topical /
    # generic-systemic keyword hits as the generic proxy.
    biologic_brands = [b for b in config.BRAND_WHITELIST_BIOLOGIC if b in seg_lower and b != brand.lower()]
    generic_hits = sum(1 for k in config.BRAND_WHITELIST_GENERIC if k in seg_lower)
    topical_hits = sum(1 for k in config.TOPICAL_KEYWORDS if k in seg_lower)
    photo_hits = sum(1 for k in config.PHOTOTHERAPY_KEYWORDS if k in seg_lower)

    leaves = []
    if biologic_brands[:1]:
        leaves.append({
            "logic": "LEAF",
            "drug_or_category": biologic_brands[0],
            "class": "BRANDED_BIOLOGIC",
            "is_mandatory": True,
        })
    if generic_hits or topical_hits:
        leaves.append({
            "logic": "LEAF",
            "drug_or_category": "methotrexate or topical agent",
            "class": "GENERIC_SYSTEMIC",
            "is_mandatory": True,
        })

    step_text = ""
    if leaves:
        step_text = "(synthetic seed) Patient must have failed one preferred biologic; topical/conventional trial considered."

    step_data = {
        "step_therapy_text": step_text,
        "moderate_to_severe_only": True,
        "phototherapy_mandatory": False,
        "step_graph": {
            "universal_branch": [],
            "indication_branch": leaves,
        },
        "evidence_snippets": [step_text[:120]] if step_text else [],
    }

    spec_match = re.search(r"(?i)dermatolog|rheumatolog|gastroenterolog", seg_text)
    spec_val = spec_match.group(0).title() if spec_match else "NA"

    ql_match = re.search(r"(?i)quantity\s+(?:limit|level\s+limit)[^\n]{0,200}", seg_text)
    ql_val = ql_match.group(0).strip() if ql_match else "Not specified"

    reauth_text = ""
    if reauth_req == "Yes":
        m = re.search(r"(?i)(?:continuation|reauthor\w*)[^\n]{0,400}", seg_text)
        reauth_text = m.group(0).strip()[:600] if m else "Continuation per policy"

    # Build the combined 8B payload (scalars + text fields + step text + flag)
    combined = {
        "age": scalars["age"],
        "tb_test_required": scalars["tb_test_required"],
        "initial_authorization_duration_months": scalars["initial_authorization_duration_months"],
        "reauthorization_duration_months": scalars["reauthorization_duration_months"],
        "reauthorization_required": scalars["reauthorization_required"],
        "reauthorization_requirements": {"value": reauth_text or "NA", "evidence": reauth_text[:200]},
        "specialist_types": {"value": spec_val, "evidence": spec_match.group(0) if spec_match else ""},
        "quantity_limits": {"value": ql_val, "evidence": ql_match.group(0)[:200] if ql_match else ""},
        "step_therapy_text": step_data["step_therapy_text"],
        "has_step_therapy": bool(leaves),
    }

    # Cache the combined 8B call
    _store(extract_params._prompt_combined(brand, seg_text),
           extract_params.SCHEMA_COMBINED,
           extract_params.SYSTEM_COMBINED, combined,
           model=config.LLM_MODEL_FAST)

    # If step therapy is present, cache the 70B step-graph call too
    # (keyed on the verbatim step text, not the segment).
    if leaves and step_data["step_therapy_text"]:
        step_graph_payload = {
            "moderate_to_severe_only": True,
            "phototherapy_mandatory": False,
            "step_graph": step_data["step_graph"],
            "evidence_snippets": step_data.get("evidence_snippets", []),
        }
        _store(extract_params._prompt_step_graph(brand, step_data["step_therapy_text"]),
               extract_params.SCHEMA_STEP_GRAPH,
               extract_params.SYSTEM_STEP_GRAPH, step_graph_payload,
               model=config.LLM_MODEL)


def _store(prompt: str, schema: dict, system: str, payload: dict,
           model: str | None = None,
           temperature: float | None = None) -> None:
    if temperature is None:
        temperature = config.LLM_TEMPERATURE_DEFAULT
    if model is None:
        model = config.LLM_MODEL
    schema_str = json.dumps(schema, sort_keys=True)
    key = llm_client._hash(  # type: ignore[attr-defined]
        model, temperature, system, prompt, schema_str,
    )
    path = config.LLM_CACHE / f"{key}.json"
    # `source: synthetic` is the in-file marker pipeline.run_all uses to detect
    # whether a real run actually called the LLM or silently fell through to
    # the dev-only mock seeds.
    path.write_text(json.dumps({
        "source": "synthetic",
        "raw_text": json.dumps(payload),
        "payload": payload,
    }, indent=2), encoding="utf-8")


def seed_all() -> int:
    rows = ingest.load_submissions()
    for r in rows:
        _seed_for_row(r.filename, r.brand)
    return len(rows)
