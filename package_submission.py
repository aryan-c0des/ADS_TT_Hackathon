"""
Bundle the submission into a single ZIP for upload.

Includes:
  - result.csv (the deliverable)
  - notebook.ipynb (judges' entrypoint)
  - src/ (all pipeline modules)
  - tests/ (unit tests + offline smoke test)
  - templates/ (Jinja audit-card template)
  - data/llm_cache/ (so judges can re-run without a live API key)
  - output/ (final result.csv, heatmap, audit cards)
  - holdout/ (your hand-labelled rows)
  - README.md, requirements.txt

Excludes: data/text/, data/segments/, data/evidence/ (re-generated on run).
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from src import config


def build_zip(zip_path: Path | None = None) -> Path:
    zip_path = zip_path or (config.PROJECT_ROOT / "arein_submission.zip")
    root = config.PROJECT_ROOT
    include_dirs = ["src", "tests", "templates", "data/llm_cache",
                    "output", "holdout"]
    include_files = ["notebook.ipynb", "README.md", "requirements.txt",
                     "package_submission.py"]

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for d in include_dirs:
            for p in (root / d).rglob("*"):
                if p.is_file() and "__pycache__" not in p.parts:
                    z.write(p, p.relative_to(root))
        for f in include_files:
            fp = root / f
            if fp.exists():
                z.write(fp, fp.relative_to(root))
    return zip_path


if __name__ == "__main__":
    out = build_zip()
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")
