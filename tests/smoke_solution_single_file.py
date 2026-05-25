"""
Verify the generated solution.py works end-to-end the same way the modular
pipeline does. Primes the LLM cache for one row, then drives the row through
solution.process_row() and renders the audit card via the embedded Jinja
DictLoader.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import solution  # noqa: E402


def prime_cache_for_row(filename: str, brand: str) -> None:
    extract_text = solution.extract_text
    segment_brand = solution.segment_brand
    extract_params = solution.extract_params
    llm_client = solution.llm_client
    config = solution.config

    text = extract_text.load_text(filename)
    seg = segment_brand.segment(filename, brand, text)
    segment_text = seg.text

    scalars_resp = {
        "age": {"value": ">=18", "evidence": "18 years of age or older"},
        "tb_test_required": {"value": "Yes", "evidence": "negative TB screening"},
        "initial_authorization_duration_months": {"value": "6", "evidence": "Initial: 6 months"},
        "reauthorization_duration_months": {"value": "12", "evidence": "Reauth: 12 months"},
        "reauthorization_required": {"value": "Yes", "evidence": "Reauth criteria below"},
    }
    step_resp = {
        "step_therapy_text": "Inadequate response to a preferred TNF inhibitor required.",
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
    text_resp = {
        "reauthorization_requirements": {"value": "Documented positive clinical response.", "evidence": "positive clinical response"},
        "specialist_types": {"value": "Dermatologist", "evidence": "prescribed by a dermatologist"},
        "quantity_limits": {"value": "Not specified", "evidence": ""},
    }

    def _store(prompt, schema, system, payload, temperature=config.LLM_TEMPERATURE_DEFAULT):
        schema_str = json.dumps(schema, sort_keys=True)
        key = llm_client._hash(config.LLM_MODEL, temperature, system, prompt, schema_str)
        path = config.LLM_CACHE / f"{key}.json"
        path.write_text(json.dumps({"raw_text": json.dumps(payload), "payload": payload}, indent=2), encoding="utf-8")

    _store(extract_params._prompt_scalars(brand, segment_text),
           extract_params.SCHEMA_SCALARS, extract_params.SYSTEM_SCALARS, scalars_resp)
    _store(extract_params._prompt_step_therapy(brand, segment_text),
           extract_params.SCHEMA_STEP_THERAPY, extract_params.SYSTEM_STEP_THERAPY, step_resp)
    _store(extract_params._prompt_text_fields(brand, segment_text),
           extract_params.SCHEMA_TEXT_FIELDS, extract_params.SYSTEM_TEXT_FIELDS, text_resp)


def main():
    filename, brand = "330109-4880941.pdf", "TREMFYA"
    rows = solution.ingest.load_submissions()
    row = next(r for r in rows if r.filename == filename and r.brand == brand)
    prime_cache_for_row(filename, brand)

    diag = solution.process_row(row, verbose=True)
    assert "csv_row" in diag, f"pipeline failed: {diag}"
    print("\nFinal row:")
    for k, v in diag["csv_row"].items():
        print(f"  {k}: {str(v)[:120]}")
    print(f"\nLayout: {diag['layout']}, segment_chars: {diag['segment_chars']}")
    print(f"Violations: {diag['violations']}")
    print(f"Access Score: {diag['csv_row']['Access Score']}")

    ev_path = solution.config.EVIDENCE_DIR / f"{Path(filename).stem}__{brand}.json"
    card_path = solution.evidence_report.render_card(ev_path)
    print(f"\nAudit card rendered: {card_path}")
    assert card_path.exists() and card_path.stat().st_size > 1000, "audit card too small or missing"
    print(f"  size: {card_path.stat().st_size:,} bytes")
    print("\nSOLUTION.PY END-TO-END: OK")


if __name__ == "__main__":
    main()
