"""
End-to-end orchestrator. Reads the Submissions sheet, runs every row through
segmentation → LLM extraction → step counting → validation → access scoring,
and emits result.csv plus the per-row evidence JSON.

Designed so judges can call `pipeline.run_all()` from a single notebook cell.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from . import (
    access_score,
    config,
    extract_params,
    extract_text,
    ingest,
    llm_client,
    segment_brand,
    step_graph,
    validate,
)


# Characters that Excel/Sheets interpret as the start of a formula. If an LLM
# returns a value starting with one of these (e.g., "-Tremfya is preferred"),
# opening result.csv in a spreadsheet would execute the cell. Defang by
# prefixing with a single quote — Excel treats the prefixed cell as text.
_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _defang_csv_value(value: Any) -> Any:
    if isinstance(value, str) and value and value[0] in _CSV_FORMULA_PREFIXES:
        return "'" + value
    return value


def _row_to_csv_dict(filename: str, brand: str,
                     fixed: Dict[str, Any], score: int) -> Dict[str, Any]:
    out = {
        "Filename": filename,
        "Brand": brand,
    }
    for col in config.SUBMISSION_COLUMNS[2:-1]:  # skip Filename, Brand, Access Score
        out[col] = _defang_csv_value(fixed.get(col, ""))
    out["Access Score"] = score
    return out


def process_row(row: ingest.Row, *,
                run_self_consistency: bool = False,
                verbose: bool = False) -> Dict[str, Any]:
    """Run one (Filename, Brand) row end-to-end."""
    diag: Dict[str, Any] = {"filename": row.filename, "brand": row.brand}
    try:
        full_text = extract_text.load_text(row.filename)
        seg = segment_brand.segment(row.filename, row.brand, full_text)
        segment_brand.save_segment(seg)
        diag["layout"] = seg.layout
        diag["segment_chars"] = len(seg.text)

        extracted = extract_params.extract_row(
            row.filename, row.brand, seg.text,
            run_self_consistency=run_self_consistency,
        )
        diag["extraction_diagnostics"] = extracted.diagnostics

        graph_payload = extracted.step_data.get("step_graph") or {}
        count_res = step_graph.count_steps(graph_payload)
        diag["step_brand_trace"] = count_res.brand_trace[:50]
        diag["step_generic_trace"] = count_res.generic_trace[:50]

        validation = validate.validate(extracted, count_res)
        diag["violations"] = validation.flags
        diag["needs_reprompt"] = validation.needs_reprompt

        brand_canon = config.canonical_brand(row.brand)
        breakdown = access_score.score_row(validation.fixed, brand_canon)
        diag["score_waterfall"] = [
            {"label": l, "delta": d, "note": n}
            for l, d, n in breakdown.contributions
        ]

        csv_row = _row_to_csv_dict(row.filename, brand_canon,
                                   validation.fixed, breakdown.score)
        diag["csv_row"] = csv_row

        # Persist evidence sidecar
        evidence = {
            "filename": row.filename,
            "brand": brand_canon,
            "layout": seg.layout,
            "segment_text": seg.text,
            "scalars_payload": extracted.scalars,
            "step_payload": extracted.step_data,
            "text_fields_payload": extracted.text_fields,
            "step_brand_trace": count_res.brand_trace,
            "step_generic_trace": count_res.generic_trace,
            "violations": validation.flags,
            "score_waterfall": diag["score_waterfall"],
            "final_row": csv_row,
        }
        evidence_path = config.EVIDENCE_DIR / f"{Path(row.filename).stem}__{brand_canon}.json"
        evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

        if verbose:
            print(f"  ✓ {row.filename} | {brand_canon} | score={breakdown.score}")
        return diag
    except Exception as exc:  # noqa: BLE001
        diag["error"] = f"{type(exc).__name__}: {exc}"
        diag["traceback"] = traceback.format_exc()
        if verbose:
            print(f"  ✗ {row.filename} | {row.brand} | ERROR: {exc}")
        return diag


def run_all(*, run_self_consistency: bool = False, limit: int | None = None,
            verbose: bool = True) -> pd.DataFrame:
    """Process every row in the Submissions sheet. Writes output/result.csv
    and returns the resulting DataFrame."""
    rows = ingest.load_submissions()
    if limit is not None:
        rows = rows[:limit]
    if verbose:
        print(f"Processing {len(rows)} (Filename, Brand) rows...")
    diagnostics: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    start = time.time()
    for i, r in enumerate(rows, 1):
        diag = process_row(r, run_self_consistency=run_self_consistency, verbose=verbose)
        diagnostics.append(diag)
        if "csv_row" in diag:
            csv_rows.append(diag["csv_row"])
        else:
            stub = {col: "" for col in config.SUBMISSION_COLUMNS}
            stub["Filename"] = r.filename
            stub["Brand"] = config.canonical_brand(r.brand)
            stub["Access Score"] = 0
            csv_rows.append(stub)
        if verbose and i % 10 == 0:
            elapsed = time.time() - start
            print(f"  [{i}/{len(rows)}] elapsed {elapsed:.1f}s")

    df = pd.DataFrame(csv_rows)[config.SUBMISSION_COLUMNS]
    df.to_csv(config.RESULT_CSV, index=False)
    (config.OUTPUT_DIR / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, default=str), encoding="utf-8"
    )
    if verbose:
        print(f"\nWrote {config.RESULT_CSV} ({len(df)} rows)")
    _warn_if_synthetic_cache_hit()
    return df


def _warn_if_synthetic_cache_hit() -> None:
    """If any cache reads came from mock_seed-generated entries, warn loudly.
    Judges who skip the README's `rm -f data/llm_cache/*` step would
    otherwise grade the synthetic seeds as a real Gemini run."""
    state = llm_client.counter_state()
    synth = state.get("synthetic_hits", 0)
    real = state.get("real_hits", 0)
    if synth == 0:
        return
    print(
        "\n" + "!" * 72 + "\n"
        f"WARNING: {synth} cache reads came from synthetic seeds "
        f"(real Gemini reads: {real}).\n"
        "result.csv may contain mocked values, not real LLM extraction.\n"
        "To re-run against the live LLM: `rm -f data/llm_cache/*` and ensure "
        "GEMINI_API_KEY is set.\n"
        + "!" * 72
    )
