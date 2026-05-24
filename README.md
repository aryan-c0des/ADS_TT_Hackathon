# Payer Policy Intelligence — Extraction & Access Score

A reproducible GenAI pipeline that reads US Prior-Authorization PDFs and emits a structured `result.csv` with 12 extracted parameters + an Access Quality Score (0–100) per `(Filename, Brand)` row.

Built for the H1'26 hackathon. The design priority is **auditability** over raw model power: every value in `result.csv` traces back to a PDF snippet, every Access Score decomposes into a transparent additive waterfall, and the deterministic step-counting logic is unit-tested against the worked example in the Business Rules sheet.

---

## Quick start

```bash
# 1. Install (Colab/Kaggle: prepend !)
pip install -r requirements.txt
apt-get install -y poppler-utils    # needs pdftotext on the PATH

# 2. Set the API key
export GEMINI_API_KEY="..."

# 3. IMPORTANT: clear synthetic cache before the first real run
rm -f data/llm_cache/*

# 4. Run the notebook end-to-end (judges should just hit "Run All")
jupyter notebook notebook.ipynb
```

> **Why the cache clear?** The repo ships with a *synthetic-response* cache (built by `src/mock_seed.py`) so the pipeline can run offline for development and demos. Once you set a real `GEMINI_API_KEY`, you want fresh LLM calls — clearing `data/llm_cache/` forces them. Subsequent runs are cached again and free.

Or run headless from a shell:

```bash
python -c "from src import pipeline; pipeline.run_all()"
```

Outputs land in `output/`:

| Path | Description |
|------|-------------|
| `output/result.csv` | The 79-row submission |
| `output/result_with_evidence.json` | Full per-row evidence bundle |
| `output/heatmap.png` | Payer restrictiveness heatmap |
| `output/score_distribution.png` | Access Score histogram by brand |
| `output/audit/<filename>__<brand>.html` | Per-row audit card with snippet+score waterfall |
| `output/audit/index.html` | Click-through index of all cards |

---

## Pipeline shape

```
70 PDFs ──► pdftotext ──► brand-section slice ──► Gemini (3 grouped prompts)
                                                       │
                                                       ▼
                                          structured JSON (step_graph, scalars)
                                                       │
                                                       ▼
                            deterministic Python: step counter + validators + access score
                                                       │
                                                       ▼
                              result.csv  +  per-row evidence JSON  +  HTML audit cards
```

### Why this shape

- **Text-first, multimodal as fallback.** All 70 PDFs in the sample corpus extract cleanly with `pdftotext -layout` (no OCR needed). Sending the whole PDF to Gemini multimodally would burn quota; the text-first path keeps each prompt under ~3K input tokens.
- **Three grouped prompts (not 12), not one monolithic.** Prompt A returns 5 scalars, Prompt B returns the step-therapy text and a structured `step_graph`, Prompt C returns three long-form text fields. This stays small enough to JSON-validate every response yet large enough that we only burn 3 calls per row (237 total — well under the 1500/day Gemini Flash free quota).
- **LLM builds the step graph, Python counts.** Asking the LLM for an integer step count is brittle. Asking it to decompose the step therapy into an `AND/OR/LEAF` graph and then counting deterministically in Python is auditable: the trace prints each leaf, each OR-path choice, and the final sum.
- **Whitelist re-classification.** The PsO Brands sheet provides a 35-drug ground-truth split (branded biologic vs generic systemic vs topical). When the LLM mis-labels a drug class, `step_graph.reconcile_class` overrides via whitelist match — see `tests/test_step_counting.py::test_branded_step_counting_via_whitelist`.

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

- **LLM call caching**: every Gemini call is keyed on `SHA256(model + temperature + system + prompt + schema)` and written to `data/llm_cache/`. The cache ships in the submission ZIP, so a fresh judge environment with no API key can still reproduce `result.csv` (provided we hit the cache).
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
│   ├── llm_client.py            # Gemini wrapper with caching + retries
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
- **Multi-modal fallback not exercised** on the sample corpus because all 70 PDFs were OCR-clean. If judges hand us scanned PDFs, the pipeline would need to switch to Gemini multimodal — the path is wired in `llm_client.py` but untested at scale.
- **Self-consistency pass** (running Prompt B at two temperatures and tie-breaking on disagreement) is implemented but disabled by default to stay within the daily call budget; enable with `pipeline.run_all(run_self_consistency=True)`.

---

## What stands out

1. **Per-cell evidence sidecar** — every value in `result.csv` is traceable to a specific PDF snippet via `data/evidence/<filename>__<brand>.json`.
2. **HTML audit cards** — zero-install for judges. Open `output/audit/index.html` and click any row.
3. **Step-graph reasoning trace** — prints the OR-path selection, leaf classifications, and running sum so judges can replicate the integer step counts by hand.
4. **Transparent Access Score** — additive rubric where every contribution is named and capped, never a black box.
5. **Whitelist-overridden LLM classifications** — the LLM can mis-label drugs; the PsO Brands sheet is the source of truth.
6. **Reproducibility-first** — cached LLM calls ship in the ZIP so first re-run is free.
