"""
Restrictiveness heatmap + access-score distribution chart.

The heatmap is the one-glance narrative for judges. Rows = (Filename, Brand)
sorted by Access Score; columns = the parameters that drive restrictiveness;
cell shading = numeric restrictiveness (greener = less restrictive).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config


_RESTRICTIVENESS_COLUMNS = [
    "Number of Steps through Brands",
    "Number of Steps through Generic",
    "Step through-Phototherapy",
    "TB Test required",
    "Quantity Limits",
    "Specialist Types",
    "Initial Authorization Duration(in-months)",
    "Reauthorization Duration(in-months)",
]


def _to_restrictiveness(col: str, val) -> float:
    """Map raw column value to a 0..1 restrictiveness score (1 = most restrictive)."""
    s = str(val).strip().lower()
    if col == "Number of Steps through Brands":
        try:
            n = int(s)
            return min(1.0, n / 3.0)
        except ValueError:
            return 0.0
    if col == "Number of Steps through Generic":
        try:
            n = int(s)
            return min(1.0, n / 2.0)
        except ValueError:
            return 0.0
    if col in {"Step through-Phototherapy", "TB Test required"}:
        return 1.0 if s == "yes" else 0.0
    if col == "Specialist Types":
        return 0.0 if s in {"na", "n/a", "none", ""} else 0.5
    if col == "Quantity Limits":
        return 0.0 if s in {"not specified", "na", "n/a", "none", ""} else 0.5
    if col in {"Initial Authorization Duration(in-months)",
               "Reauthorization Duration(in-months)"}:
        if s in {"unspecified", "na", "n/a", "none", ""}:
            return 0.3
        try:
            n = int(s)
            if n >= 12:
                return 0.0
            if n >= 6:
                return 0.3
            return 0.7
        except ValueError:
            return 0.3
    return 0.0


def render_heatmap(result_csv: Optional[Path] = None,
                   out_path: Optional[Path] = None) -> Path:
    result_csv = result_csv or config.RESULT_CSV
    out_path = out_path or config.HEATMAP_PNG
    if not result_csv.exists():
        raise FileNotFoundError(result_csv)
    df = pd.read_csv(result_csv)
    df = df.sort_values("Access Score").reset_index(drop=True)
    M = np.zeros((len(df), len(_RESTRICTIVENESS_COLUMNS)))
    for i, row in df.iterrows():
        for j, col in enumerate(_RESTRICTIVENESS_COLUMNS):
            M[i, j] = _to_restrictiveness(col, row[col])

    fig, ax = plt.subplots(figsize=(11, max(7, 0.16 * len(df))))
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(_RESTRICTIVENESS_COLUMNS)))
    ax.set_xticklabels(
        [c.replace("(in-months)", "").replace("Step through-", "").strip()
         for c in _RESTRICTIVENESS_COLUMNS],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(
        [f"{r['Brand']} · {r['Filename'][:18]} · {int(r['Access Score'])}"
         for _, r in df.iterrows()],
        fontsize=6,
    )
    fig.colorbar(im, ax=ax, label="restrictiveness (red = more restrictive)")
    ax.set_title("Payer Restrictiveness Heatmap (rows sorted by Access Score)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_score_distribution(result_csv: Optional[Path] = None,
                              out_path: Optional[Path] = None) -> Path:
    result_csv = result_csv or config.RESULT_CSV
    out_path = out_path or (config.OUTPUT_DIR / "score_distribution.png")
    df = pd.read_csv(result_csv)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for brand, grp in df.groupby("Brand"):
        if len(grp) < 2:
            continue
        ax.hist(grp["Access Score"], bins=range(0, 105, 5),
                alpha=0.5, label=f"{brand} (n={len(grp)})")
    ax.axvline(25, color="grey", linestyle="--", linewidth=0.7)
    ax.axvline(50, color="grey", linestyle="--", linewidth=0.7)
    ax.axvline(75, color="grey", linestyle="--", linewidth=0.7)
    ax.set_xlabel("Access Score (0=no access, 50=parity, 100=best)")
    ax.set_ylabel("Number of policies")
    ax.set_title("Access Score Distribution by Brand")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
