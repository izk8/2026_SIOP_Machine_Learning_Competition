# Fourth Place: MetaML

**Private Test Set Final MSE: 0.012888**

## Team Members

- Pengda Wang @ Rice University

---

## What the solution does

For each study we estimate an **aggregate Pearson correlation *r*** between two constructs (predictor and outcome) defined in meta-analysis rules, using only the **article PDF** and the **study-level text** in `study_definitions.csv`.

The pipeline:

1. **Multi-agent extraction and calculation** — Several LLM steps read the PDF with the research question and construct definitions: extract eligible bivariate effects, check extraction, convert each effect to *r* in construct direction (including reverse-coding), aggregate as an **unweighted mean** of per-effect *r* values, verify calculation, and emit a single CSV row per study.
2. **Two independent runs** — The same pipeline runs twice (configurable providers) so we get two aggregate values per study for robustness.
3. **Reconciliation** — When the two run summaries disagree, a dedicated reconciliation prompt uses the PDF, definitions, and **both** full run logs to produce one final aggregate *r* (or leaves it empty if the PDF is missing).
4. **Empty / zero follow-up** — Studies whose final aggregate is missing, NaN, or exactly zero get an extra LLM pass that either recomputes *r* from the paper or, when truly not computable, fills a **best-estimate** numeric value per the prompt rules.

Supporting logic (effect-size conversions, API clients, prompts) lives in the modules **embedded inside** `run_submission.py`; they are extracted to the system temp directory at runtime.

## How to Run

From the directory that contains `run_submission.py` (or pass `--submission-dir`):

```bash
python run_submission.py --workers 4 --provider openai --run1-provider openai --run2-provider openai
```

- **`--workers`** — Parallelism within each stage (per-study workers).
- **`--run1-provider` / `--run2-provider`** — APIs for the two full pipeline passes (e.g. OpenAI then Anthropic for diversity).
- **`--provider`** — API used by reconciliation and empty-estimate stages.

### API Keys and Dependencies

Install Python dependencies: `anthropic`, `openai`, `google-genai`, `python-dotenv`, `pandas`.

Provide keys via environment variables or a `.env` file:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`

## Pipeline Stages

| Order | Stage | Role |
|------:|-------|------|
| 1 | `run_study` | Runs the full multi-agent pipeline on every study twice; writes `reasoning1/` and `reasoning2/` with per-study logs. |
| 2 | `reconcile_reasoning` | Compares the two runs; on disagreement calls a reconciliation LLM with PDF + both logs. |
| 3 | `empty_estimate` | For rows that are empty, NaN, or zero, runs a recheck/estimate pass. |
