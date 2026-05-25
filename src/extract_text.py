"""
PDF → text via poppler's pdftotext.

All 70 sample PDFs in this corpus are text-based (no OCR needed); pdftotext
with -layout preserves the indented bullet structures the policies rely on.
Outputs are cached per-PDF in data/text/<filename>.txt so the rest of the
pipeline never re-shells out.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List

from . import config

PDFTOTEXT = shutil.which("pdftotext") or "pdftotext"


def extract_one(pdf_path: Path, force: bool = False) -> str:
    cache_path = config.TEXT_CACHE / (pdf_path.stem + ".txt")
    if cache_path.exists() and not force:
        return cache_path.read_text(encoding="utf-8", errors="replace")
    result = subprocess.run(
        [PDFTOTEXT, "-layout", str(pdf_path), "-"],
        capture_output=True, text=True, timeout=120, errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pdftotext failed on {pdf_path.name}: {result.stderr[:200]}"
        )
    text = result.stdout
    cache_path.write_text(text, encoding="utf-8")
    return text


def extract_all(pdfs: Iterable[Path] | None = None, force: bool = False) -> List[Path]:
    """Extract text for every PDF in the corpus; returns the cached paths.

    Idempotent: if cache files exist they are skipped unless force=True.
    """
    if pdfs is None:
        pdfs = sorted(p for p in config.PDF_DIR.glob("*.pdf"))
    extracted: List[Path] = []
    for p in pdfs:
        try:
            extract_one(p, force=force)
        except Exception as exc:
            print(f"[extract_text] FAILED {p.name}: {exc}")
            continue
        extracted.append(config.TEXT_CACHE / (p.stem + ".txt"))
    return extracted


def load_text(filename: str) -> str:
    """Load cached text for a given PDF filename (with .pdf suffix)."""
    stem = Path(filename).stem
    cache_path = config.TEXT_CACHE / f"{stem}.txt"
    if not cache_path.exists():
        pdf_path = config.PDF_DIR / filename
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)
        return extract_one(pdf_path)
    return cache_path.read_text(encoding="utf-8", errors="replace")


