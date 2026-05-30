# PROGRESS.md

Running log of what's been built and what's left. CLAUDE.md holds durable
agent instructions and hard rules; this file holds the evolving project state.
Update this whenever a meaningful chunk of work lands.

**Last updated:** 2026-05-30 (after full real-cache re-run)
**Repo:** https://github.com/aryan-c0des/ADS_TT_Hackathon (branch `main`, public)
**Latest commit:** `544bfab`
**Deadline:** 9 AM IST, 1 June 2026

---

## Architecture (current = "Option H" hybrid)

- Per `(Filename, Brand)`: ONE `llama-3.1-8b-instant` **combined** call (8
  scalar/text fields + verbatim `step_therapy_text` + `has_step_therapy` bool),
  then CONDITIONALLY ONE `llama-3.3-70b-versatile` call to structure step
  therapy into an AND/OR/LEAF step graph. The 70B call fires only when the 8B
  `has_step_therapy` flag OR the Python `_has_step_therapy_markers()` heuristic
  fires; if 8B returned empty `step_therapy_text`, the capped segment is sent to
  70B instead.
- Groq free-tier limits: 8B = 500K TPD / 6000 TPM-per-request; 70B = 100K TPD /
  6000 TPM. `config.LLM_MAX_OUTPUT_TOKENS=1024`,
  `extract_params._MAX_SEGMENT_CHARS_FOR_COMBINED=8000`, schema injected compact
  (no indent) — all to stay under the 6000 TPM per-request cap.
- Deterministic Python does the rest: `step_graph.count_steps`,
  `validate.validate`, `access_score.score_row`.

---

## Latest run — 2026-05-30 (full 79-row, real cache)

Resumed in Colab with last night's downloaded cache; the 429 backlog cleared.

- **Health:** 79 rows, 0 hard failures. 14 fresh LLM calls, 0 errors, **0
  synthetic hits**, 110 real cache replays — confirms the cache is now the REAL
  Colab cache (no `mock_seed` warning). 70B step graph invoked=45, succeeded=45,
  failed=0.
- **Age rule verified:** the fresh output resolves FDA-labelled refs to
  drug-specific `>=N` (distribution: `>=18` ×37, `>=6` ×26, `>=12` ×2, `>=2` ×1,
  `>=4` ×1) — the old `"FDA labelled age"` literal is gone.
- **Score distribution:** min 12, max 59, mean 41.2. Buckets: NoAccess (0-25) 4,
  Restricted (26-50) 63, Preferred (51-75) 12, Best (76-100) 0. Heavily
  compressed in Restricted — plausibly genuine (all are restrictive PA
  policies), but worth a sanity check.

### Findings to act on

1. **Specialist leaks — 6/79 flagged, 2 true leaks.** Multi-indication policies
   leak non-PsO prescribers. Genuine leaks: row 74 OTEZLA (`Pulmonologist`, from
   Behçet's) and row 78 STELARA (`Immunologist, Gastroenterologist, Colorectal
   Surgeon`, from Crohn's/UC). The other 4 are `Dermatologist, Rheumatologist` —
   defensible for PsO/PsA. Root cause: prompt-only scoping in
   `extract_params.py`, no deterministic backstop in `validate.py`.
   **Planned fix:** deterministic reject-list filter in `validate.py` (strip
   Gastro/Colorectal/Pulmonologist/Immunologist/Hematologist/Oncologist/etc.,
   keep Dermatologist + Rheumatologist) + a unit test.
2. **Step-count outliers to spot-check:** brand steps = 7 (1 row), generic
   steps = 6 (1 row, echoes the old OR-vs-AND mis-decomposition). Possible
   over-count tanking a score.
3. **NoAccess outliers:** 4 rows in 0-25 (min score 12) — confirm genuinely
   restrictive vs. an artifact of step over-counting.
4. **Possible age mis-extraction:** TREMFYA `>=2` (FDA min for PsO is 6) — verify
   against the policy (`187701-5050284.pdf`).

### Notes / gotchas

- **Minor inefficiency:** step graph `fallback_to_segment=45/45` — the 8B's
  `step_therapy_text` is never used; we always re-send the segment to 70B.
  Harmless (counts populate) but that 8B output is currently dead weight.
- **Stale local `result.csv`:** the `result.csv` beside the source PDFs (one
  level up from the repo) PREDATES the Age fix (shows `"FDA labelled age"` and
  different scores). The authoritative current output is the fresh Colab run —
  do not verify against the local copy.
- **Spot-check setup:** all 70 source PDFs are available locally at
  `../Sample_PsO_ADS_Track/`, so PDF cross-checks need no upload — only the
  fresh `result.csv` to compare against.

---

## What's DONE

- **LLM swap** Gemini 2.5 → Llama (70B-versatile + 8B-instant) on Groq free tier.
- **Single-file build** `build_single_file.py` concatenates `src/*.py` into
  `solution.py` (deps injected via `types.ModuleType` + exec, Jinja
  FileSystemLoader→DictLoader swap, synthetic `__file__` so path math holds).
- **Option H hybrid routing** (8B combined + conditional 70B step graph).
- **TPM/TPD budget fixes** — `LLM_MAX_OUTPUT_TOKENS` 4096→2048→1024, compact
  schema, 8K segment cap for mega-formularies.
- **step_graph correctness** — word-boundary `classify_drug_name`,
  biologic-whitelist-first ordering, phototherapy-only-when-single-purpose,
  OR-vs-AND ("ONE of the following" = OR, least-restrictive) decomposition.
- **Specialist rejection list** in the prompt (stops over-extraction across all
  indications).
- **Organizer Age rule** in `validate._normalise_age(value, brand)`.
- **Self-contained reproducible ZIP** — `package_submission.py` bundles the 70
  PDFs + xlsx + `.env.example` at submission-layout paths; refuses to ship a
  real `.env`. `config._resolve_data_path()` resolves both submission and dev
  layouts.
- **Tests green** — 11 step-counter unit tests + 2 smoke tests
  (`smoke_pipeline_offline.py`, `smoke_solution_single_file.py`).
- **Full real-cache re-run (2026-05-30)** cleared the 429 backlog — 79 rows, 0
  hard failures, 0 synthetic hits; Age rule confirmed working in the output.

---

## What's LEFT

1. ~~Re-run the TPD-failed (429) rows~~ **DONE (2026-05-30)** — backlog cleared
   (14 fresh calls, 0 errors).
2. **Package the REAL cache into the ZIP.** The real Colab `data/llm_cache/` now
   exists (0 synthetic hits this run); pull it into the repo and repackage so
   the shipped ZIP no longer falls back to `mock_seed`.
3. **Fix text-cache reproducibility.** Dev `data/text/` is stale CRLF (~17,636
   chars) vs a fresh `pdftotext` run (~16,308 chars, LF) → cache-key mismatch
   means an evaluator's fresh extraction won't hit the shipped LLM cache. Ship
   `data/text/` in the ZIP, or regenerate it fresh before seeding.
4. **Refine extraction quality** (see "Findings to act on" above):
   - Specialist deterministic filter in `validate.py` (highest-value, clear win).
   - Spot-check brand=7 / generic=6 / NoAccess outliers + TREMFYA `>=2` age.
5. Optional: hand-label 8 holdout rows in `holdout/holdout_labels.csv`.

**Submission-blocking pair:** #2 + #3 — the evaluator must reproduce our numbers
from the shipped cache. Cleanest fix: ship `data/text/` alongside the real
`data/llm_cache/` so keys line up.

---

## Bug-fix history (chronological)

- **"tar" ⊂ "targeted" misclassification** — `(Sotyktu, Otezla)` mislabelled
  TOPICAL because "tar" was a substring of "targeted". Fixed with word-boundary
  regex + biologic-first ordering. (commit `3a7c8aa`)
- **70B TPD exhaustion** — 100K/day burned after ~9 rows. Redesigned to the
  Option H hybrid so the cheap 8B handles most rows. (commit `4ef5f28`)
- **413 "Request too large"** (>6000 TPM) — max_output 4096→2048 + compact
  schema (`ccc05a3`), then →1024 + 8K segment cap for long mega-formulary
  segments (`9214994`).
- **Empty `step_therapy_text` dropped the step graph** — added segment-fallback
  to 70B + strengthened the detection prompt. (commit `f3dc9e7`)
- **Specialist over-extraction** — rejection list added to the prompt. (`b662e71`)
- **Phototherapy lumped-leaf false positive** — classify only single-purpose
  phototherapy leaves. (`75be8ca`)
- **OR-vs-AND misdecomposition** (generic count came out 5/6) — worked example
  added to the 70B prompt. (`0606333`)
- **Age extraction alignment** with organizer clarification. (`8c5b996`)
- **Self-contained ZIP** for the evaluator workflow. (`c854d39`)

---

## How to run (Windows / this repo)

- Tests: `python tests/test_step_counting.py` (11),
  `python tests/smoke_pipeline_offline.py`,
  `python tests/smoke_solution_single_file.py`
- Rebuild single-file: `python build_single_file.py`
- Package: `python package_submission.py` → `arein_submission.zip` (~33 MB,
  includes PDFs + xlsx)
- Real run: set `GROQ_API_KEY` (or `.env`), clear `data/llm_cache/*.json`, then
  `python solution.py`
