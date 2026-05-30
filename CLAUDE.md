# CLAUDE.md

Instructions for Claude when working in this repo. The README is for judges and humans; this file is for the agent.

## What this is

A GenAI pipeline for the H1'26 Payer Policy Intelligence hackathon. It reads 70 US Prior-Authorization PDFs and writes `output/result.csv` with 79 rows (one per `Filename + Brand`), each populated with 12 extracted parameters and a 0–100 Access Score. Submission deadline: **9 AM IST, 1 June 2026**.

The plan that drove this build lives at `/Users/yuvraj/.claude/plans/i-have-a-hackathon-whimsical-lovelace.md`. Read it before touching the architecture.

## Layout

```
src/
├── config.py          # paths, brand whitelist, FDA ages, ACCESS_SCORE_RUBRIC
├── ingest.py          # load PA_Business_Rules.xlsx + list 70 PDFs
├── extract_text.py    # pdftotext wrapper → data/text/
├── segment_brand.py   # 3-layout brand-section isolation
├── llm_client.py      # Groq/Llama wrapper, SHA-keyed disk cache
├── extract_params.py  # 3 grouped prompts (Scalars / StepTherapy / TextFields)
├── step_graph.py      # deterministic step counter — HIGHEST LEVERAGE FILE
├── validate.py        # business-rule cross-checks + auto-fixes
├── access_score.py    # additive rubric → score + waterfall
├── pipeline.py        # orchestrator (run_all, process_row)
├── evidence_report.py # Jinja HTML audit cards
├── visualize.py       # heatmap + score distribution PNGs
├── holdout.py         # 8-row hand-labelled accuracy harness
└── mock_seed.py       # synthetic cache populator (dev-only)
templates/audit_card.html   # Jinja template
tests/                      # test_step_counting.py + smoke_pipeline_offline.py
notebook.ipynb              # judges' Run-All entrypoint
output/                     # result.csv, heatmap.png, audit/*.html
data/{text,segments,llm_cache,evidence}/   # all cached, ignore in commits
holdout/holdout_labels.csv  # user fills this in (Day-4 hand-labels)
```

Source data is one level up: `../Sample_PsO_ADS_Track/` (70 PDFs) and `../PA_Business_Rules.xlsx`.

## How to run

```bash
# Tests
python3 tests/test_step_counting.py        # 6 unit tests on the counter
python3 tests/smoke_pipeline_offline.py    # 1-row end-to-end with cache

# Full pipeline (~11 s from cold cache)
python3 -c "from src import pipeline; pipeline.run_all()"

# After full run: audit cards + visuals
python3 -c "from src import evidence_report, visualize; evidence_report.render_all(); evidence_report.render_index(); visualize.render_heatmap(); visualize.render_score_distribution()"

# Submission ZIP
python3 package_submission.py
```

To run with the real LLM (not the synthetic cache):
```bash
rm -f data/llm_cache/*
export GROQ_API_KEY="..."
python3 -c "from src import pipeline; pipeline.run_all()"
```

## Hard rules — do NOT change without re-running tests

### Step counting (`src/step_graph.py`)
1. `universal_branch` AND `indication_branch` are unioned via AND (sum).
2. Inside an OR node, take the LEAST RESTRICTIVE child (min count) — that's the business rule.
3. Phototherapy leaves are NEVER counted in branded or generic totals.
4. `reconcile_class()` uses `BRAND_WHITELIST_BIOLOGIC` / `BRAND_WHITELIST_GENERIC` to override the LLM's `class` field. The whitelist is the source of truth — the LLM is only consulted when the whitelist returns `OTHER`.
5. Returns `"NA"` (string) when the count is 0 — not `0`. Per the Business Rules sheet.
6. `_has_mandatory_phototherapy` returns True only when a PHOTOTHERAPY leaf is mandatory AND not under any OR ancestor.

If you touch any of these, `tests/test_step_counting.py` must still pass — those tests encode the Reference-sheet worked example.

### LLM client (`src/llm_client.py`)
- Cache key: `SHA256(model + temperature + system + prompt + schema)`. Don't change the hash inputs without invalidating the existing cache and re-running everything.
- Response is always JSON (Groq `response_format={"type": "json_object"}` + schema injected into the system prompt — Llama has no native schema enforcement, unlike the old Gemini `responseSchema`). The `payload` field in the cache file is the parsed dict.
- Default temperature is 0.0 (deterministic). Prompt B can run a second pass at 0.2 for self-consistency (`run_self_consistency=True` in pipeline).

### Submission CSV (`config.SUBMISSION_COLUMNS`)
- Column order matters — judges' eval harness reads by name but human eyeballs read by position. Don't reorder.
- The 13 fields are: Filename, Brand, Age, Step Therapy Requirements Documented in Policy, Number of Steps through Brands, Number of Steps through Generic, Step through-Phototherapy, TB Test required, Quantity Limits, Specialist Types, Initial Authorization Duration(in-months), Reauthorization Duration(in-months), Reauthorization Required, Reauthorization Requirements Documented in Policy, Access Score.
- Empty / not-applicable cells use these literal strings: `"NA"`, `"Unspecified"`, `"Not specified"`. Pick the one the Business Rules sheet uses for that field.

### Access Score (`src/access_score.py`)
- The rubric is in `config.ACCESS_SCORE_RUBRIC`. Anchors: 0=no access, 25=restricted, 50=parity (base), 75=preferred, 100=best.
- Score is additive over named contributions; floor 0, ceiling 100.
- Brand-step total is capped at -24, generic-step total at -15 — uncapped, a hyper-restrictive policy would dominate the distribution.
- Every contribution is logged in `ScoreBreakdown.contributions` and rendered in the HTML audit card. If you add a new contribution, also add it to the rubric dict.

## Things to watch

- **Synthetic cache shadows real LLM calls.** `data/llm_cache/` ships pre-populated with `mock_seed.py` outputs. Clearing it (`rm -f data/llm_cache/*`) is the first step before a real Groq/Llama run.
- **All paths in `config.py` are absolute via `pathlib.Path(__file__).resolve()`.** Don't hard-code working directory assumptions.
- **`pdftotext` (poppler) must be on PATH.** Colab: `!apt-get install -y poppler-utils`. macOS: `brew install poppler`.
- **Brand canonicalization** happens in `config.canonical_brand()`. Use it — don't compare brand strings raw.
- **Multi-brand PDFs** (9 of the 70) appear in the Submissions sheet as separate rows. Pipeline iterates Submissions, not PDFs.
- **The 13 minor brands** (AMJEVITA, COSENTYX, etc., 18 rows total) are likely the weakest spot for accuracy. Spot-check those audit cards before submitting.
- **Specialist Types must be PsO-scoped.** Multi-indication policies (STELARA, OTEZLA, etc.) list a prescriber per indication; only Dermatologist — and Rheumatologist when the PsO/PsA criteria name it — are valid. Specialties from co-indications (Gastroenterologist, Colorectal Surgeon, Pulmonologist, Immunologist, Hematologist, Oncologist) must be stripped. Prompt scoping in `extract_params.py` is unreliable on the 8B model; the deterministic backstop belongs in `validate.py`.

## When the user asks "what are my next steps"

Read `NEXTSTEPS.md` at the project root and present its contents. That file owns the user-facing remaining-tasks playbook; do not rewrite it from memory.

## When the user asks for changes

- Bug in step counting: open `src/step_graph.py`, write a failing test in `tests/test_step_counting.py` first, then fix.
- Adjusting Access Score weights: edit `config.ACCESS_SCORE_RUBRIC` only. Don't hard-code numbers in `access_score.py`.
- Adding a new parameter: this is invasive — it touches `config.SUBMISSION_COLUMNS`, one of the three prompt schemas in `extract_params.py`, `validate.py`, and the audit card template. Ask the user before doing it.
- "Stop using the synthetic cache" → `rm -f data/llm_cache/*` + ensure `GROQ_API_KEY` is set.
- "Run on a single row to debug" → `from src import pipeline, ingest; pipeline.process_row(ingest.load_submissions()[N], verbose=True)`.

## Don't

- Don't add a Streamlit dashboard. The user explicitly chose HTML audit cards (zero-install for judges).
- Don't switch the LLM. The hackathon mandates Llama-3.3-70B-versatile (Groq free tier). Llama-3.1-8B-instant is the only allowed fallback.
- Don't OCR the PDFs — all 70 are text-extractable; OCR is dead weight.
- Don't extend `mock_seed.py` to look smart. It's a development scaffold; the README is honest about what it is. Improving it risks the user confusing synthetic output with real extraction.
- Don't remove the per-cell evidence sidecar or the audit cards — they ARE the standout interpretability layer.

## Current architecture & progress

The evolving project state — current architecture ("Option H" hybrid), what's
done, what's left, and the bug-fix history — lives in **`PROGRESS.md`** at the
project root. Read it at the start of a session to load context. Update it
whenever a meaningful chunk of work lands; keep CLAUDE.md for durable
instructions and hard rules only.

Quick orientation (see PROGRESS.md for detail):

- **Pipeline shape:** per `(Filename, Brand)`, one `llama-3.1-8b-instant`
  combined call (scalars + verbatim `step_therapy_text` + `has_step_therapy`),
  then a conditional `llama-3.3-70b-versatile` call that structures step therapy
  into an AND/OR/LEAF graph. Deterministic Python does counting, validation, and
  scoring.
- **Age rule** (`validate._normalise_age(value, brand)`): silent → `NA`;
  "No restriction" → `No`; "adult" → `>=18`; "FDA labelled age" →
  `config.FDA_MIN_AGE_PSO[brand]` as `>=N`; numeric → `>=N`.
- **Reproducibility (MANDATORY, failure = nullified):** the ZIP must be
  self-contained (PDFs, xlsx, code, `.env.example` at relative paths). Evaluator
  unzips → `cp .env.example .env` → `python solution.py` → reads
  `output/result.csv`, with NO code edits. Driver `solution.py` is built from
  `src/` via `python build_single_file.py` — never edit it by hand. Never ship a
  real `.env`.
