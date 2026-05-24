"""
Hold-out accuracy harness.

Day 4 of the plan: hand-label 8 rows in holdout/holdout_labels.csv that span
all three layout types and a mix of brands. This module compares the
pipeline's predictions to those labels and emits a per-parameter precision
table. Run it after every full pipeline pass to see if accuracy is regressing.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from . import config


HOLDOUT_TEMPLATE_ROWS = [
    # Diverse mix recommended by the plan.
    {"Filename": "330109-4880941.pdf", "Brand": "TREMFYA",  "layout_hint": "single_drug"},
    {"Filename": "176207-4867884.pdf", "Brand": "TREMFYA",  "layout_hint": "single_drug"},
    {"Filename": "287728-4459856.pdf", "Brand": "STELARA",  "layout_hint": "multi_drug"},
    {"Filename": "298309-4972610.pdf", "Brand": "STELARA",  "layout_hint": "multi_drug"},
    {"Filename": "56403-5061730.pdf",  "Brand": "STELARA",  "layout_hint": "mega_formulary"},
    {"Filename": "313179-3560271.pdf", "Brand": "TREMFYA",  "layout_hint": "multi_drug"},
    {"Filename": "8889-4641730.pdf",   "Brand": "AMJEVITA", "layout_hint": "minor_brand"},
    {"Filename": "8898-4735285.pdf",   "Brand": "COSENTYX", "layout_hint": "minor_brand"},
]


def write_template(force: bool = False) -> Path:
    """Create holdout/holdout_labels.csv with the canonical columns and
    pre-populated (Filename, Brand) rows. User fills in the values.
    """
    if config.HOLDOUT_CSV.exists() and not force:
        return config.HOLDOUT_CSV
    cols = config.SUBMISSION_COLUMNS + ["layout_hint"]
    rows: List[Dict[str, Any]] = []
    for r in HOLDOUT_TEMPLATE_ROWS:
        row = {c: "" for c in cols}
        row["Filename"] = r["Filename"]
        row["Brand"] = r["Brand"]
        row["layout_hint"] = r["layout_hint"]
        rows.append(row)
    pd.DataFrame(rows)[cols].to_csv(config.HOLDOUT_CSV, index=False)
    return config.HOLDOUT_CSV


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def _norm(value: Any) -> str:
    s = "" if value is None else str(value).strip()
    s = re.sub(r"\s+", " ", s).lower()
    return s


def _equal_loose(gold: Any, pred: Any) -> bool:
    g, p = _norm(gold), _norm(pred)
    if g == p:
        return True
    # Treat empties as equivalent
    if g in {"", "na", "n/a", "none"} and p in {"", "na", "n/a", "none"}:
        return True
    if g in {"not specified", "unspecified"} and p in {"not specified", "unspecified"}:
        return True
    # Yes/No
    if g in {"yes", "no"} and p in {"yes", "no"}:
        return g == p
    return False


def evaluate(result_csv: Path | None = None,
             holdout_csv: Path | None = None) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Return (per-column precision, joined dataframe). Per-column precision
    excludes rows whose holdout value is blank."""
    result_csv = result_csv or config.RESULT_CSV
    holdout_csv = holdout_csv or config.HOLDOUT_CSV
    if not result_csv.exists():
        raise FileNotFoundError(result_csv)
    if not holdout_csv.exists():
        raise FileNotFoundError(holdout_csv)

    pred = pd.read_csv(result_csv)
    gold = pd.read_csv(holdout_csv)
    joined = gold.merge(pred, on=["Filename", "Brand"], suffixes=("_gold", "_pred"))

    precision: Dict[str, float] = {}
    for col in config.SUBMISSION_COLUMNS[2:]:
        g_col = f"{col}_gold"
        p_col = f"{col}_pred"
        if g_col not in joined.columns:
            continue
        mask = joined[g_col].astype(str).str.strip() != ""
        sub = joined.loc[mask]
        if len(sub) == 0:
            continue
        hits = sum(_equal_loose(g, p) for g, p in zip(sub[g_col], sub[p_col]))
        precision[col] = hits / len(sub)
    return precision, joined


def print_report(precision: Dict[str, float]) -> None:
    print("Per-parameter precision (holdout):")
    print("-" * 60)
    for col, p in precision.items():
        marker = "✓" if p >= 0.8 else ("·" if p >= 0.5 else "✗")
        print(f"  {marker} {col:55s} {p*100:5.1f}%")
    if precision:
        mean = sum(precision.values()) / len(precision)
        print(f"\n  Mean per-parameter precision: {mean*100:.1f}%")
