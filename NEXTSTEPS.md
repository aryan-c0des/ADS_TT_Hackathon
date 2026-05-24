# Next Steps

The end-to-end pipeline is built and verified. These are the remaining tasks before submission, in order.

---

### Step 1 — Get a Gemini API key

1. Go to **https://aistudio.google.com/apikey** → "Create API key" → copy it.
2. Free tier on Gemini 2.5 Flash is 1500 calls/day — plenty for the 237-call budget.

### Step 2 — Choose your run environment

The hackathon rules require Kaggle or Colab free tier. Recommended: **Google Colab** because:
- Easier to upload the folder (drag-drop to `/content/`)
- `poppler-utils` available with one `!apt-get install`
- Mounting Google Drive makes the cache persistent across sessions

In Colab, first cell:
```python
!apt-get install -y poppler-utils
!pip install -q google-generativeai pandas openpyxl jinja2 matplotlib
import os; os.environ['GEMINI_API_KEY'] = 'paste-your-key-here'
```

### Step 3 — Run the real pipeline

```bash
# from arein_hackathon/
rm -f data/llm_cache/*          # CRITICAL — clear synthetic seeds first
python3 -c "from src import pipeline; pipeline.run_all(verbose=True)"
```

This will make ~237 real Gemini calls (~5–10 minutes wallclock). Watch the output for `ERROR` lines on any rows. The pipeline now prints a loud warning at the end if any cache reads were synthetic — that confirms you actually called Gemini.

### Step 4 — Eyeball-check 5 audit cards

```bash
python3 -c "from src import evidence_report; evidence_report.render_all(); evidence_report.render_index()"
```

Open `output/audit/index.html` in a browser. Click these specific cards to validate:
- `330109-4880941__TREMFYA.html` — single-drug Aetna policy (easiest)
- `298309-4972610__TREMFYA.html` — multi-drug class policy (medium)
- `298309-4972610__STELARA.html` — same PDF, different brand (tests brand isolation)
- `56403-5061730__STELARA.html` — Medicaid mega-formulary (hardest)
- `8889-4641730__AMJEVITA.html` — minor brand (highest risk)

For each: does the step-graph trace match what the snippet actually says? If not, that's a prompt issue.

### Step 5 — Hand-label 8 holdout rows

Open `holdout/holdout_labels.csv`. The 8 `(Filename, Brand)` pairs are pre-filled. For each:
1. Open the PDF in `../Sample_PsO_ADS_Track/<filename>.pdf`
2. Read the section that governs that brand for PsO
3. Fill in the 13 columns (Age, # brand steps, # generic steps, phototherapy Y/N, TB test Y/N, quantity limits text, specialist, durations, reauth Y/N, reauth criteria text, and your own access-score estimate 0-100)

Don't overthink the access score — use the rubric in `README.md` as a guide.

### Step 6 — Compute holdout accuracy

```bash
python3 -c "from src import holdout; p, _ = holdout.evaluate(); holdout.print_report(p)"
```

You'll see per-parameter precision. If mean is **≥ 80%** you're in good shape. **65–80%** means tweak the worst parameters. **< 65%** means something is structurally wrong.

### Step 7 — Iterate on the worst parameters

For the 2-3 lowest-scoring parameters:
1. Find the rows where it failed → look at `data/evidence/<filename>__<brand>.json`
2. Adjust the system prompt in `src/extract_params.py` (the `SYSTEM_SCALARS` / `SYSTEM_STEP_THERAPY` / `SYSTEM_TEXT_FIELDS` constants)
3. Re-run **just the failing rows** by deleting their cache entries and re-running

Common fixes:
- Step-count wrong → add a harder-rule clause to `SYSTEM_STEP_THERAPY`
- Quantity limits over-capturing → strengthen the "ONLY if labelled 'quantity limit'" rule in `SYSTEM_TEXT_FIELDS`
- Age coming back as `>=18` when policy says nothing → tighten the "FDA labelled age" fallback in `_normalise_age`

### Step 8 — Tune the Access Score rubric (optional)

```python
import pandas as pd
df = pd.read_csv('output/result.csv')
print(df.groupby('Brand')['Access Score'].describe())
```

If TREMFYA / STELARA medians look implausibly low (e.g., < 30) or high (> 70), tweak `ACCESS_SCORE_RUBRIC` in `src/config.py` and re-run *just* the scoring:
```python
from src import pipeline; pipeline.run_all()   # cached LLM, only re-scores
```

### Step 9 — Final fresh-env smoke test

In a brand-new Colab notebook:
1. Upload the ZIP
2. `!unzip arein_submission.zip`
3. Run all cells in `notebook.ipynb`
4. Confirm `output/result.csv` regenerates and matches your last good run

If anything breaks in fresh Colab — fix `requirements.txt` or `notebook.ipynb` until it doesn't.

### Step 10 — Submit

```bash
python3 package_submission.py
```

Upload `arein_submission.zip` to the hackathon portal **before 9 AM IST, 1 June 2026**. Keep a screenshot of the submission confirmation.

---

## What can go wrong (and quick fixes)

| Symptom | Fix |
|---|---|
| `pdftotext: command not found` | `!apt-get install -y poppler-utils` (Colab) or `brew install poppler` (mac) |
| Gemini returns `429 RESOURCE_EXHAUSTED` | You hit the 1500/day cap. Wait until UTC midnight or use a second Google account key. |
| Step counts seem wrong on most rows | Open one bad audit card → look at the step_graph JSON → likely the LLM is producing the wrong structure. Tighten the few-shot in `extract_params.SYSTEM_STEP_THERAPY`. |
| Holdout accuracy below 50% on Age | Probably "FDA labelled age" misuse. Check what your hand-labels look like vs predictions. |
| Multi-brand PDFs both extract the same values | `segment_brand.py` failed to isolate. Spot-check the cached slice in `data/segments/<filename>__<brand>.txt`. |
| Pipeline run finishes with "WARNING: N cache reads came from synthetic seeds" | You forgot to clear `data/llm_cache/` before running. Delete the cache, set `GEMINI_API_KEY`, and re-run. |

---

**The single highest-leverage thing you can do is Step 5 (hand-label).** Without it you're flying blind on whether the pipeline is actually accurate. Don't skip it.
