"""
Bundle the submission into a single self-contained ZIP for upload.

Per organizers' clarification (29 May 2026), the ZIP must be fully
reproducible: evaluators unzip, update .env credentials, run the driver,
and read result.csv. No path edits, no file drops, no code modifications.
If the unzipped ZIP doesn't run as-is, the submission is nullified.

So the ZIP contains EVERYTHING needed:
  - Sample_PsO_ADS_Track/*.pdf  ← input PDFs (70 files, ~700KB)
  - PA_Business_Rules.xlsx       ← reference rules + Submissions sheet
  - solution.py                  ← the canonical driver (run with `python solution.py`)
  - .env.example                 ← template; evaluator copies to .env and fills in
  - README.md                    ← evaluator-facing run instructions
  - requirements.txt             ← Python deps
  - src/                         ← modular dev tree (readable reference)
  - tests/                       ← unit tests + smoke tests
  - templates/                   ← Jinja audit-card template (also embedded in solution.py)
  - data/llm_cache/              ← (optional) pre-computed cache for free re-runs
  - output/                      ← (optional) our reference result.csv + audit cards + heatmap
  - holdout/                     ← (optional) hand-labelled rows + accuracy report
  - notebook.ipynb               ← alternate notebook-style entrypoint
  - build_single_file.py         ← script that regenerates solution.py from src/

Excludes: data/text/, data/segments/, data/evidence/ (auto-regenerated),
the actual .env (never ship credentials), __pycache__, .git.

Path layout: PDFs and the xlsx live at PROJECT_ROOT level INSIDE the ZIP.
config.py's _resolve_data_path() tries PROJECT_ROOT/<name> first, falling
back to PROJECT_ROOT.parent/<name> for the developer's source layout —
both work without code changes.
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

from src import config


def _rebuild_solution_py() -> Path:
    """Re-run build_single_file.py so the bundled solution.py reflects the
    current src/ state. Fail loudly if the build script errors — we don't
    want to ship a stale or broken single-file."""
    root = config.PROJECT_ROOT
    build_script = root / "build_single_file.py"
    if not build_script.exists():
        raise FileNotFoundError(
            f"{build_script} missing — solution.py cannot be regenerated."
        )
    result = subprocess.run(
        [sys.executable, str(build_script)],
        cwd=str(root), capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"build_single_file.py failed (exit {result.returncode}):\n"
            f"{result.stderr or result.stdout}"
        )
    solution = root / "solution.py"
    if not solution.exists():
        raise RuntimeError("build_single_file.py succeeded but solution.py is missing")
    return solution


def build_zip(zip_path: Path | None = None) -> Path:
    zip_path = zip_path or (config.PROJECT_ROOT / "arein_submission.zip")
    root = config.PROJECT_ROOT

    # Always regenerate solution.py from the current src/ state. Doing this
    # here (rather than relying on the user to remember) means the bundled
    # single-file artifact and the modular tree can never drift.
    _rebuild_solution_py()

    # Code + supporting files that live INSIDE the project tree.
    include_dirs = ["src", "tests", "templates", "data/llm_cache",
                    "output", "holdout"]
    include_files = ["solution.py", "build_single_file.py",
                     "notebook.ipynb", "README.md", "requirements.txt",
                     "package_submission.py", ".env.example"]

    # Input data — lives at REPO_ROOT (one above project) in the dev
    # layout, but MUST be inside the ZIP for reproducibility. We resolve
    # the source location via the same helper config.py uses, then write
    # them under PROJECT_ROOT-relative paths so they unzip into the
    # submission layout.
    data_payloads = [
        ("Sample_PsO_ADS_Track", config.PDF_DIR),
        ("PA_Business_Rules.xlsx", config.RULES_XLSX),
    ]

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        # 1. Project tree code + assets.
        for d in include_dirs:
            for p in (root / d).rglob("*"):
                if p.is_file() and "__pycache__" not in p.parts:
                    z.write(p, p.relative_to(root))
        for f in include_files:
            fp = root / f
            if fp.exists():
                z.write(fp, fp.relative_to(root))

        # 2. Source data — write at submission-layout paths (PROJECT_ROOT/Name).
        for rel_name, src_path in data_payloads:
            if not src_path.exists():
                raise FileNotFoundError(
                    f"Source data missing: {src_path}. Cannot build a "
                    "reproducible submission ZIP without it."
                )
            if src_path.is_dir():
                for f in src_path.rglob("*"):
                    if f.is_file():
                        z.write(f, Path(rel_name) / f.relative_to(src_path))
            else:
                z.write(src_path, rel_name)

        # 3. Refuse to ship secrets. Defense-in-depth: if a developer's
        # real .env got accidentally added to include_files, abort.
        for name in z.namelist():
            if name.endswith(".env") and not name.endswith(".env.example"):
                raise RuntimeError(
                    f"Refusing to ship credentials file: {name}. "
                    "Remove .env from include_files."
                )
    return zip_path


if __name__ == "__main__":
    out = build_zip()
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")
