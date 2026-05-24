"""
Render per-row HTML audit cards from the evidence JSON sidecars.

The interpretability layer: judges open output/audit/<filename>__<brand>.html
in a browser and see each extracted value beside its supporting snippet,
the validation flags, the step-graph reasoning trace, and the access-score
waterfall. Pure HTML + a Jinja template — no JS, no install.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import config


_env = Environment(
    loader=FileSystemLoader(str(config.TEMPLATES_DIR)),
    autoescape=select_autoescape(["html"]),
)


FIELD_LABELS = {
    "Age": "Age",
    "Step Therapy Requirements Documented in Policy": "Step Therapy (verbatim)",
    "Number of Steps through Brands": "# Branded Steps",
    "Number of Steps through Generic": "# Generic Steps",
    "Step through-Phototherapy": "Phototherapy Step",
    "TB Test required": "TB Test Required",
    "Quantity Limits": "Quantity Limits",
    "Specialist Types": "Specialist Types",
    "Initial Authorization Duration(in-months)": "Initial Auth Duration (mo)",
    "Reauthorization Duration(in-months)": "Reauth Duration (mo)",
    "Reauthorization Required": "Reauth Required",
    "Reauthorization Requirements Documented in Policy": "Reauth Requirements",
}


def _evidence_for(scalars: Dict[str, Any], text_fields: Dict[str, Any],
                  step_payload: Dict[str, Any], col: str) -> str:
    map_scalars = {
        "Age": "age",
        "TB Test required": "tb_test_required",
        "Initial Authorization Duration(in-months)": "initial_authorization_duration_months",
        "Reauthorization Duration(in-months)": "reauthorization_duration_months",
        "Reauthorization Required": "reauthorization_required",
    }
    map_text = {
        "Reauthorization Requirements Documented in Policy": "reauthorization_requirements",
        "Specialist Types": "specialist_types",
        "Quantity Limits": "quantity_limits",
    }
    if col in map_scalars:
        return (scalars.get(map_scalars[col]) or {}).get("evidence", "")
    if col in map_text:
        return (text_fields.get(map_text[col]) or {}).get("evidence", "")
    if col == "Step Therapy Requirements Documented in Policy":
        snips = step_payload.get("evidence_snippets") or []
        return " | ".join(snips[:3])
    return ""


def render_card(evidence_path: Path) -> Path:
    data = json.loads(evidence_path.read_text(encoding="utf-8"))
    final_row = data["final_row"]
    fields = []
    for col in config.SUBMISSION_COLUMNS[2:-1]:
        fields.append({
            "label": FIELD_LABELS.get(col, col),
            "value": final_row.get(col, ""),
            "evidence": _evidence_for(
                data.get("scalars_payload", {}),
                data.get("text_fields_payload", {}),
                data.get("step_payload", {}),
                col,
            ),
        })

    waterfall = [(w["label"], w["delta"], w.get("note", ""))
                 for w in data.get("score_waterfall", [])]
    html = _env.get_template("audit_card.html").render(
        filename=data["filename"],
        brand=data["brand"],
        layout=data.get("layout", ""),
        segment_chars=len(data.get("segment_text", "")),
        segment_text=data.get("segment_text", "")[:8000],
        violations=data.get("violations", []),
        fields=fields,
        waterfall=waterfall,
        final_score=final_row.get("Access Score", 0),
        brand_steps=final_row.get("Number of Steps through Brands", "NA"),
        generic_steps=final_row.get("Number of Steps through Generic", "NA"),
        brand_trace="\n".join(data.get("step_brand_trace", []) or ["(empty)"]),
        generic_trace="\n".join(data.get("step_generic_trace", []) or ["(empty)"]),
    )
    out_path = config.AUDIT_DIR / (evidence_path.stem + ".html")
    out_path.write_text(html, encoding="utf-8")
    return out_path


def render_all() -> List[Path]:
    paths: List[Path] = []
    for ev in sorted(config.EVIDENCE_DIR.glob("*.json")):
        try:
            paths.append(render_card(ev))
        except Exception as exc:  # noqa: BLE001
            print(f"[evidence_report] failed on {ev.name}: {exc}")
    return paths


def render_index() -> Path:
    """Tiny index.html listing every audit card for one-click browsing."""
    items = []
    for p in sorted(config.AUDIT_DIR.glob("*.html")):
        items.append(f'<li><a href="{p.name}">{p.stem}</a></li>')
    body = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Audit Index</title>"
        "<style>body{font-family:system-ui;padding:24px;}li{padding:4px 0}</style></head>"
        "<body><h1>Audit cards (" + str(len(items)) + ")</h1><ul>"
        + "\n".join(items)
        + "</ul></body></html>"
    )
    path = config.AUDIT_DIR / "index.html"
    path.write_text(body, encoding="utf-8")
    return path
