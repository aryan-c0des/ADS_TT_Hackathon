# Payer Policy Intelligence — Extraction & Access Score

A reproducible GenAI pipeline that reads US Prior-Authorization PDFs and emits a structured `result.csv` with 12 extracted parameters + an Access Quality Score (0–100) per `(Filename, Brand)` row.

Built for the H1'26 hackathon. The design priority is **auditability** over raw model power: every value in `result.csv` traces back to a PDF snippet, every Access Score decomposes into a transparent additive waterfall, and the deterministic step-counting logic is unit-tested against the worked example in the Business Rules sheet.

---

## 🚀 Evaluator workflow (4 steps)

The submission ZIP is fully self-contained. Run the pipeline as follows:

**1. Unzip the submission**

```bash
unzip arein_submission.zip -d arein_submission
cd arein_submission
```

Everything ships inside the ZIP with relative paths — PDFs, code, Excel sheets, and dependencies list. No external files or path edits needed.

**2. Install dependencies and prepare the environment**

```bash
pip install -r requirements.txt
apt-get install -y poppler-utils   # Linux; macOS: brew install poppler; Windows: choco install poppler
```

The pipeline needs `pdftotext` (from poppler) on the PATH for PDF parsing.

**3. Set your Groq API key via `.env`**

Copy the template and fill in your key:

```bash
cp .env.example .env
# Edit .env and replace `your_groq_api_key_here` with your actual key from
# https://console.groq.com/keys (free tier is sufficient)
```

The driver auto-loads `.env` on startup. Alternatively, export `GROQ_API_KEY=gsk_...` in your shell.

**4. Run the driver and read `result.csv`**

```bash
python solution.py
```

That's it. The driver:
- Processes all 70 PDFs across all 79 `(Filename, Brand)` rows from the Submissions sheet.
- Writes the 15-column CSV to **`output/result.csv`** (the submission deliverable).
- Renders per-row audit cards to `output/audit/*.html` (one HTML file per row, plus `index.html`).
- Renders visualization PNGs (`output/heatmap.png`, `output/score_distribution.png`).

Typical wallclock: **8-12 minutes** for the full 79 rows on Groq free tier (cold cache).

CLI flags:
- `python solution.py --limit 5` — process first 5 rows only (smoke test).
- `python solution.py --no-cards` — skip audit-card rendering.
- `python solution.py --help` — show all options.

### Outputs

| Path | Description |
|------|-------------|
| **`output/result.csv`** | **The 79-row submission deliverable (15 columns)** |
| `output/audit/<filename>__<brand>.html` | Per-row audit card with evidence snippets + score waterfall |
| `output/audit/index.html` | Click-through index of all 79 audit cards |
| `output/heatmap.png` | Per-row restrictiveness heatmap |
| `output/score_distribution.png` | Access Score histogram |
| `output/diagnostics.json` | Pipeline diagnostics (per-row LLM call metadata + traces) |

### Project layout inside the ZIP

```
arein_submission/
├── solution.py                  # ← DRIVER (run with `python solution.py`)
├── .env.example                 # ← Copy to `.env`, fill in GROQ_API_KEY
├── README.md                    # ← This file
├── requirements.txt             # ← Python dependencies
├── Sample_PsO_ADS_Track/        # ← 70 input PDFs
│   ├── 330109-4880941.pdf
│   ├── 148593-4960549.pdf
│   └── ... (68 more)
├── PA_Business_Rules.xlsx       # ← Reference rules + Submissions sheet
├── src/                         # ← Modular source (solution.py is built from this)
├── tests/                       # ← Unit + smoke tests
├── templates/                   # ← Jinja audit-card template
├── data/llm_cache/              # ← (optional) pre-computed cache for free re-runs
├── output/                      # ← (optional) our reference output for comparison
└── holdout/                     # ← (optional) hand-labelled accuracy harness
```

### Re-running with cached responses

If the bundled `data/llm_cache/` is present, re-runs hit the cache and skip the Groq calls entirely — useful when the evaluator wants to reproduce our `output/result.csv` without spending tokens. To force fresh LLM calls, clear the cache first:

```bash
rm -f data/llm_cache/*.json
python solution.py
```

---

## Model declaration

**This submission uses two models from the Llama family via Groq free tier**, both spec-compliant per hackathon rules:

| Pipeline step | Model | Why this model |
|---|---|---|
| Combined extraction (8 scalars + step-therapy text + has_step_therapy flag) | `llama-3.1-8b-instant` | Fast + high daily-token budget (500K TPD). Handles structured extraction of well-defined fields reliably. |
| Step-graph decomposition (AND/OR/LEAF structure for step therapy) | `llama-3.3-70b-versatile` | Needs reasoning over nested step requirements. Only invoked when step therapy is detected (~30 of 79 rows), keeping us well under 70B's 100K TPD. |

All non-LLM pipeline steps (segmentation, step counting, validation, access scoring) are deterministic Python — fully reproducible and unit-tested.

---

## Pipeline shape

```
70 PDFs ──► pdftotext ──► brand-section slice ──┬─► Llama-3.1-8b (1 combined call):
                                                │    8 scalars/text fields + verbatim step_therapy_text + has_step_therapy flag
                                                │
                                                ▼
                          (if has_step_therapy OR Python keyword heuristic fires)
                                                │
                                                ▼
                          Llama-3.3-70b: structure step therapy into AND/OR/LEAF graph
                                                │
                                                ▼
                          deterministic Python: step counter + validators + access score
                                                │
                                                ▼
                          result.csv  +  per-row evidence JSON  +  HTML audit cards
```

### Why this shape

- **Text-first, no OCR.** All 70 PDFs in the sample corpus extract cleanly with `pdftotext -layout`. Skips multimodal complexity entirely.
- **Brand-aware segmentation.** `segment_brand.py` isolates the PsO-relevant slice for each `(Filename, Brand)` row using 3 layout heuristics (single-drug / multi-drug / mega-formulary). Cuts LLM input from ~50K to ~2K chars per row — a ~95% token reduction that keeps every call comfortably under Groq's per-request TPM cap.
- **One combined 8B call instead of 3 separate.** The 8 simple-extract fields (age, TB test, durations, specialist, quantity limits, reauth criteria, reauth required) are folded into a single `llama-3.1-8b-instant` call alongside the verbatim step-therapy text and a `has_step_therapy` flag. One JSON schema, one cache key, one round trip per row.
- **70B reserved for the hardest task.** Step-therapy decomposition into AND/OR/LEAF is the only step that genuinely benefits from `llama-3.3-70b-versatile`'s reasoning. It's invoked only when step therapy is detected (~30 of 79 rows), keeping us under 70B's 100K TPD daily budget.
- **Python keyword heuristic backs up the LLM flag.** `_has_step_therapy_markers()` scans the segment for "step therapy", "previously received", "inadequate response", "contraindication to" etc. If either signal fires, 70B gets called. Defensive — false positives are cheap, false negatives would silently drop step counts.
- **LLM builds the step graph, Python counts.** Asking the LLM for an integer step count is brittle. Asking it to decompose into `AND/OR/LEAF` and then counting deterministically in Python is auditable: the trace prints each leaf, each OR-path choice, and the final sum.
- **Whitelist re-classification.** The PsO Brands sheet provides a 35-drug ground-truth split (branded biologic vs generic systemic vs topical). When the LLM mis-labels a drug class, `step_graph.reconcile_class` overrides via whitelist match — see `tests/test_step_counting.py::test_branded_step_counting_via_whitelist`.
- **Graceful degradation.** If a 70B call fails (rate limit, network), the row still lands in CSV with 8B fields intact — step counts default to `NA`, and the failure is logged. No row is ever lost.

---

## The 12 parameters

| # | Parameter | Source module | Hard rule |
|---|---|---|---|
| 1 | Age | `extract_params.py` (Prompt A) | Youngest threshold if two groups; "FDA labelled age" when unspecified numerically. |
| 2 | Step Therapy Requirements (text) | `extract_params.py` (Prompt B) | Verbatim; moderate-to-severe only. |
| 3 | Number of Steps through Brands | `step_graph.py` | UNION universal AND indication, least-restrictive OR path, count only `BRANDED_BIOLOGIC` leaves. |
| 4 | Number of Steps through Generic | `step_graph.py` | Same union/OR logic, count `GENERIC_SYSTEMIC` + `TOPICAL` leaves. |
| 5 | Step through-Phototherapy | `step_graph.py` | `Yes` only if a `PHOTOTHERAPY` leaf is mandatory AND not under any OR ancestor. |
| 6 | TB Test required | Prompt A | Y/N. |
| 7 | Initial Authorization Duration (months) | Prompt A | Integer or "Unspecified". |
| 8 | Reauthorization Duration (months) | Prompt A | Integer or "Unspecified". |
| 9 | Reauthorization Required | `validate.py` | Auto-`Yes` if duration or criteria are present. |
| 10 | Reauthorization Requirements (text) | Prompt C | Verbatim. |
| 11 | Specialist Types | Prompt C | Comma-separated specialty names. |
| 12 | Quantity Limits | `validate.py` | Only when evidence contains "quantity limit"/"QL"/"limited to"; never "dosage". |
| 13 | Access Score | `access_score.py` | Transparent additive rubric (see below). |

---

## Access Score Rubric (0–100)

Anchors per the problem statement: **0 = no access**, **25 = restricted vs FDA**, **50 = parity**, **75 = preferred**, **100 = best possible**.

```python
RUBRIC = {
  "base": 50,
  "weights": {
    "age_restrictive":       -10,
    "step_per_brand":         -8,   # cap at -24 cumulative
    "step_per_generic":       -5,   # cap at -15 cumulative
    "phototherapy_required":  -6,
    "tb_test":                -2,
    "specialist_required":    -4,
    "quantity_limited":       -4,
    "initial_auth_long":      +5,   # >= 12 months
    "initial_auth_short":     -3,   # <= 3 months
    "reauth_long":            +3,
    "reauth_short":           -5,
    "no_reauth_required":     +8,
  },
  "floor": 0, "ceiling": 100,
}
```

Every contribution is logged with its evidence; the audit card renders the full waterfall:

```
base (parity with FDA)                    +50
age above FDA label                       -10  age=>=18 vs FDA min 6
1 branded step(s)                          -8
1 generic step(s)                          -5
TB test required                           -2
specialist required                        -4  Dermatologist
reauth 12mo (≥12)                          +3
                                          ------
Access Score                                24
```

The weights live in `src/config.py:ACCESS_SCORE_RUBRIC` — judges can re-fit them by editing one dict and re-running.

---

## Reproducibility

- **LLM call caching**: every Groq call is keyed on `SHA256(model + temperature + system + prompt + schema)` and written to `data/llm_cache/`. The cache ships in the submission ZIP, so a fresh judge environment with no API key can still reproduce `result.csv` (provided we hit the cache).
- **Determinism after LLM**: step counting, validation, and access scoring are pure functions — no randomness, no time/network dependencies.
- **Smoke test cell** at top of the notebook processes a single row in ~10 seconds (cached) — fails fast if the env is broken.

### Tests

```bash
python tests/test_step_counting.py
python tests/smoke_pipeline_offline.py
```

Both should pass on a fresh checkout.

---

## Project layout

```
arein_hackathon/
├── notebook.ipynb               # judges' one-click entrypoint
├── src/
│   ├── config.py                # paths, brand whitelist, RUBRIC
│   ├── ingest.py                # load xlsx sheets, list PDFs
│   ├── extract_text.py          # pdftotext wrapper, cached
│   ├── segment_brand.py         # 3-layout brand-section isolation
│   ├── llm_client.py            # Groq/Llama wrapper with caching + retries
│   ├── extract_params.py        # 3 grouped prompts + few-shot
│   ├── step_graph.py            # deterministic step counter (the hard part)
│   ├── validate.py              # business-rule cross-checks
│   ├── access_score.py          # transparent additive rubric
│   ├── evidence_report.py       # HTML audit cards
│   ├── visualize.py             # heatmap + distribution
│   ├── pipeline.py              # orchestrator
│   └── holdout.py               # holdout accuracy harness
├── data/
│   ├── text/                    # cached pdftotext outputs
│   ├── segments/                # cached brand-isolated slices
│   ├── llm_cache/               # SHA256(prompt) → response.json
│   └── evidence/                # per-row evidence JSON
├── output/                      # result.csv + heatmap + audit cards
├── templates/audit_card.html    # Jinja template for audit cards
├── tests/                       # step counter + offline smoke test
├── holdout/holdout_labels.csv   # hand-labelled 8 rows for self-grading
├── requirements.txt
└── README.md
```

---

## Limitations + honesty

- **8-row hold-out, not blind**: I hand-labelled 8 diverse rows to self-measure accuracy. That's a *self-graded* metric, not an independent eval.
- **Access Score weights are a hypothesis**: tuned against the 440-row silver-label table in `Additional Extracted Data`, not the actual hackathon gold standard.
- **Multi-modal fallback not exercised** on the sample corpus because all 70 PDFs were OCR-clean. If judges hand us scanned PDFs, the pipeline would need an OCR pre-processing step before text extraction — not implemented.
- **Self-consistency pass** (running Prompt B at two temperatures and tie-breaking on disagreement) is implemented but disabled by default to stay within the daily call budget; enable with `pipeline.run_all(run_self_consistency=True)`.

---

## What stands out

1. **Per-cell evidence sidecar** — every value in `result.csv` is traceable to a specific PDF snippet via `data/evidence/<filename>__<brand>.json`.
2. **HTML audit cards** — zero-install for judges. Open `output/audit/index.html` and click any row.
3. **Step-graph reasoning trace** — prints the OR-path selection, leaf classifications, and running sum so judges can replicate the integer step counts by hand.
4. **Transparent Access Score** — additive rubric where every contribution is named and capped, never a black box.
5. **Whitelist-overridden LLM classifications** — the LLM can mis-label drugs; the PsO Brands sheet is the source of truth.
6. **Reproducibility-first** — cached LLM calls ship in the ZIP so first re-run is free.
