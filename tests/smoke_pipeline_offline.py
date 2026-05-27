"""
Offline smoke test: prime the LLM cache with synthetic responses for one
(Filename, Brand) row and verify the pipeline produces a valid CSV row,
evidence sidecar, and audit card without needing a Groq API key.

Updated for the Option H hybrid architecture (one 8B combined call +
one conditional 70B step-graph call).

Run with:  python tests/smoke_pipeline_offline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Windows consoles default to cp1252 and choke on glyphs like ✓ and ≥ that
# appear in score waterfall notes. Force UTF-8 so the smoke test renders
# correctly regardless of host locale.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import (  # noqa: E402
    access_score,
    config,
    evidence_report,
    extract_params,
    extract_text,
    ingest,
    llm_client,
    pipeline,
    segment_brand,
)


def prime_cache_for_row(filename: str, brand: str) -> None:
    """Pre-populate the LLM cache for a single row with plausible synthetic
    responses so we can run the rest of the pipeline end-to-end."""
    text = extract_text.load_text(filename)
    seg = segment_brand.segment(filename, brand, text)
    segment_text = seg.text

    # 8B combined response — all scalars + text fields + step text + flag
    combined_step_text = "The patient must have an inadequate response to a preferred TNF inhibitor or methotrexate."
    combined_resp = {
        "age": {"value": ">=18", "evidence": "18 years of age or older"},
        "tb_test_required": {"value": "Yes", "evidence": "negative TB screening"},
        "initial_authorization_duration_months": {"value": "6", "evidence": "Initial: 6 months"},
        "reauthorization_duration_months": {"value": "12", "evidence": "Reauth: 12 months"},
        "reauthorization_required": {"value": "Yes", "evidence": "Reauth criteria below"},
        "reauthorization_requirements": {
            "value": "Documented positive clinical response (reduction in BSA or symptom improvement).",
            "evidence": "positive clinical response",
        },
        "specialist_types": {"value": "Dermatologist", "evidence": "prescribed by a dermatologist"},
        "quantity_limits": {"value": "Not specified", "evidence": ""},
        "step_therapy_text": combined_step_text,
        "has_step_therapy": True,
    }

    # 70B step-graph response — produces the structured graph from the
    # verbatim step text above.
    step_graph_resp = {
        "moderate_to_severe_only": True,
        "phototherapy_mandatory": False,
        "step_graph": {
            "universal_branch": [],
            "indication_branch": [
                {"logic": "LEAF", "drug_or_category": "a preferred TNF inhibitor",
                 "class": "BRANDED_BIOLOGIC", "is_mandatory": True},
                {"logic": "LEAF", "drug_or_category": "methotrexate",
                 "class": "GENERIC_SYSTEMIC", "is_mandatory": True},
            ],
        },
        "evidence_snippets": ["inadequate response to a preferred TNF inhibitor"],
    }

    def _store(prompt: str, schema: dict, system: str, payload: dict,
               model: str, temperature: float = config.LLM_TEMPERATURE_DEFAULT):
        schema_str = json.dumps(schema, sort_keys=True)
        key = llm_client._hash(model, temperature, system, prompt, schema_str)  # type: ignore[attr-defined]
        path = config.LLM_CACHE / f"{key}.json"
        path.write_text(json.dumps({"raw_text": json.dumps(payload), "payload": payload},
                                   indent=2), encoding="utf-8")

    # 8B combined call
    _store(
        extract_params._prompt_combined(brand, segment_text),
        extract_params.SCHEMA_COMBINED,
        extract_params.SYSTEM_COMBINED,
        combined_resp,
        model=config.LLM_MODEL_FAST,
    )
    # 70B step-graph call — keyed on the verbatim step text, not the segment
    _store(
        extract_params._prompt_step_graph(brand, combined_step_text),
        extract_params.SCHEMA_STEP_GRAPH,
        extract_params.SYSTEM_STEP_GRAPH,
        step_graph_resp,
        model=config.LLM_MODEL,
    )


def main():
    filename, brand = "330109-4880941.pdf", "TREMFYA"
    rows = ingest.load_submissions()
    row = next(r for r in rows if r.filename == filename and r.brand == brand)
    prime_cache_for_row(filename, brand)
    diag = pipeline.process_row(row, verbose=True)
    assert "csv_row" in diag, f"pipeline failed: {diag}"
    print("\nFinal row:")
    for k, v in diag["csv_row"].items():
        v_str = str(v)[:120]
        print(f"  {k}: {v_str}")
    print(f"\nLayout: {diag['layout']}, segment_chars: {diag['segment_chars']}")
    print(f"Violations: {diag['violations']}")
    print(f"Diagnostics: llm_has_steps={diag['extraction_diagnostics'].get('llm_has_steps')}, "
          f"py_has_steps={diag['extraction_diagnostics'].get('py_has_steps')}, "
          f"step_graph_invoked={diag['extraction_diagnostics'].get('step_graph_invoked')}")
    print("Score waterfall:")
    for w in diag["score_waterfall"]:
        print(f"  {w['label']:40s}  {w['delta']:+d}  {w.get('note','')}")
    # Render the audit card
    ev_path = config.EVIDENCE_DIR / f"{Path(filename).stem}__{brand}.json"
    out = evidence_report.render_card(ev_path)
    print(f"\nAudit card: {out}")


if __name__ == "__main__":
    main()
