"""
Central configuration for the Payer Policy Intelligence pipeline.

Paths, constants, brand whitelist, and the Access Score rubric all live here so
judges can audit the formula in one place.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent

PDF_DIR = REPO_ROOT / "Sample_PsO_ADS_Track"
RULES_XLSX = REPO_ROOT / "PA_Business_Rules.xlsx"

DATA_DIR = PROJECT_ROOT / "data"
TEXT_CACHE = DATA_DIR / "text"
SEGMENT_CACHE = DATA_DIR / "segments"
LLM_CACHE = DATA_DIR / "llm_cache"
EVIDENCE_DIR = DATA_DIR / "evidence"

OUTPUT_DIR = PROJECT_ROOT / "output"
AUDIT_DIR = OUTPUT_DIR / "audit"
RESULT_CSV = OUTPUT_DIR / "result.csv"
RESULT_JSON = OUTPUT_DIR / "result_with_evidence.json"
HEATMAP_PNG = OUTPUT_DIR / "heatmap.png"

HOLDOUT_DIR = PROJECT_ROOT / "holdout"
HOLDOUT_CSV = HOLDOUT_DIR / "holdout_labels.csv"

TEMPLATES_DIR = PROJECT_ROOT / "templates"

for d in (TEXT_CACHE, SEGMENT_CACHE, LLM_CACHE, EVIDENCE_DIR, AUDIT_DIR, HOLDOUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MODEL_HEAVY = "gemini-2.5-pro"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_TEMPERATURE_DEFAULT = 0.0
GEMINI_TEMPERATURE_SECONDARY = 0.2
GEMINI_MAX_OUTPUT_TOKENS = 4096
GEMINI_MAX_RETRIES = 3
DAILY_CALL_BUDGET = 1500

# ---------------------------------------------------------------------------
# Output schema — exact column order required by Submissions sheet
# ---------------------------------------------------------------------------
SUBMISSION_COLUMNS = [
    "Filename",
    "Brand",
    "Age",
    "Step Therapy Requirements Documented in Policy",
    "Number of Steps through Brands",
    "Number of Steps through Generic",
    "Step through-Phototherapy",
    "TB Test required",
    "Quantity Limits",
    "Specialist Types",
    "Initial Authorization Duration(in-months)",
    "Reauthorization Duration(in-months)",
    "Reauthorization Required",
    "Reauthorization Requirements Documented in Policy",
    "Access Score",
]

# ---------------------------------------------------------------------------
# PsO market basket (from the "PsO Brands- For Ground Truth" sheet)
# Classified into branded biologics vs conventional generics — the
# step_graph.classify() function uses this as the source of truth and treats
# the LLM's class label as a hint that must agree with this whitelist.
# ---------------------------------------------------------------------------
BRAND_WHITELIST_BIOLOGIC = {
    "amjevita", "avsola", "bimzelx", "cimzia", "cosentyx",
    "cyltezo", "enbrel", "humira", "hyrimoz", "idacio",
    "ilumya", "inflectra", "hulio", "otezla", "remicade",
    "renflexis", "siliq", "skyrizi", "sotyktu", "stelara",
    "taltz", "tremfya", "yuflyma", "yusimry", "wezlana",
    "selarsdi", "yesintek", "pyzchiva", "quallent", "steqeyma",
    "otulfi", "ustekinumab", "adalimumab", "infliximab",
    "guselkumab", "risankizumab", "ixekizumab", "secukinumab",
    "etanercept", "certolizumab", "tildrakizumab", "brodalumab",
    "bimekizumab", "deucravacitinib", "apremilast",
}

BRAND_WHITELIST_GENERIC = {
    "acitretin", "cyclosporine", "methotrexate",
    "vtama", "zoryve",  # non-biologic topicals
    "tapinarof", "roflumilast",
}

# Topical / phototherapy keywords
TOPICAL_KEYWORDS = {
    "topical", "corticosteroid", "vitamin d", "calcipotriene", "calcipotriol",
    "retinoid", "tazarotene", "tar", "anthralin", "emollient", "moisturizer",
    "calcineurin inhibitor", "tacrolimus", "pimecrolimus",
}

PHOTOTHERAPY_KEYWORDS = {
    "phototherapy", "uvb", "puva", "ultraviolet",
    "narrowband", "narrow-band", "psoralen", "uv light",
}

# Brand canonical names for the submission
BRAND_CANONICAL = {
    "tremfya": "TREMFYA",
    "stelara": "STELARA",
    "amjevita": "AMJEVITA",
    "cosentyx": "COSENTYX",
    "enbrel": "ENBREL",
    "remicade": "REMICADE",
    "siliq": "SILIQ",
    "cimzia": "CIMZIA",
    "bimzelx": "BIMZELX",
    "skyrizi": "SKYRIZI",
    "otezla": "OTEZLA",
    "yesintek": "YESINTEK",
    "otulfi": "OTULFI",
    "ilumya": "ILUMYA",
    "acitretin": "ACITRETIN",
}

# FDA-labelled minimum age per brand for PsO (used by Access Score)
FDA_MIN_AGE_PSO = {
    "TREMFYA": 6,
    "STELARA": 6,
    "COSENTYX": 6,
    "SKYRIZI": 18,
    "ENBREL": 4,
    "TALTZ": 6,
    "ILUMYA": 18,
    "OTEZLA": 6,
    "SOTYKTU": 18,
    "SILIQ": 18,
    "BIMZELX": 18,
    "HUMIRA": 18,
    "CIMZIA": 18,
    "ACITRETIN": 18,
    "REMICADE": 18,
    "OTULFI": 6,
    "AMJEVITA": 18,
    "YESINTEK": 6,
}

# ---------------------------------------------------------------------------
# Access Score Rubric (transparent, additive)
# Documented in /Users/yuvraj/.claude/plans/i-have-a-hackathon-whimsical-lovelace.md §6
# ---------------------------------------------------------------------------
ACCESS_SCORE_RUBRIC = {
    "base": 50,
    "weights": {
        "age_restrictive":      -10,   # extracted min age > FDA-label age for brand
        "step_per_brand":        -8,   # capped at -24 cumulative
        "step_per_generic":      -5,   # capped at -15 cumulative
        "phototherapy_required": -6,
        "tb_test":               -2,
        "specialist_required":   -4,
        "quantity_limited":      -4,
        "initial_auth_long":     +5,   # >= 12 months
        "initial_auth_short":    -3,   # <= 3 months
        "reauth_long":           +3,   # >= 12 months
        "reauth_short":          -5,   # < 12 months but present
        "no_reauth_required":    +8,
    },
    "caps": {
        "step_per_brand_total":   -24,
        "step_per_generic_total": -15,
    },
    "floor": 0,
    "ceiling": 100,
}

# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------
SEGMENT_MIN_CHARS = 800
SEGMENT_MAX_CHARS = 15000
SEGMENT_DEFAULT_RADIUS = 2000   # widen anchor by this many chars on either side
SEGMENT_MED_SEVERE_RADIUS = 1500
LARGE_PDF_TEXT_THRESHOLD = 300_000  # >= this many chars → Medicaid mega-formulary path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_api_key() -> str | None:
    """Return the Gemini API key from env, or None if not set.

    Pipeline tolerates a missing key for the local cache-replay path but
    will raise when an actual call is needed."""
    return os.environ.get(GEMINI_API_KEY_ENV)


def canonical_brand(brand: str) -> str:
    return BRAND_CANONICAL.get(brand.lower().strip(), brand.upper().strip())
