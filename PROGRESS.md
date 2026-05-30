# PROGRESS.md

Running log of what's been built and what's left. CLAUDE.md holds durable
agent instructions and hard rules; this file holds the evolving project state.
Update this whenever a meaningful chunk of work lands.

**Last updated:** 2026-05-30
**Repo:** https://github.com/aryan-c0des/ADS_TT_Hackathon (branch `main`, public)
**Latest commit:** `c854d39`
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

---

## What's LEFT

1. **Re-run the ~20 TPD-failed (429) rows** in Colab to fill in missing step
   counts. User has downloaded the output cache and can re-run only the failed
   rows.
2. **Replace the SYNTHETIC `mock_seed` cache** currently in the ZIP with the
   REAL Colab cache, then repackage. A synthetic-seed warning currently prints
   on run.
3. **Fix text-cache reproducibility.** Dev `data/text/` is stale CRLF (~17,636
   chars) vs a fresh `pdftotext` run (~16,308 chars, LF) → cache-key mismatch
   means an evaluator's fresh extraction won't hit the shipped LLM cache. Either
   ship `data/text/` in the ZIP or regenerate it fresh before seeding.
4. **Further refine extraction quality** — user's stated next intent (get
   specifics before acting).
5. Optional: hand-label 8 holdout rows in `holdout/holdout_labels.csv`; verify
   Age outputs (FDA-labelled → drug-specific `>=N`) on the re-run.

**Submission-blocking pair:** #2 + #3. The shipped cache is synthetic and the
dev text cache won't hash-match a fresh `pdftotext` run, so an evaluator's run
won't reproduce our numbers from cache. Cleanest fix: ship `data/text/`
alongside the real `data/llm_cache/` so keys line up.

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
