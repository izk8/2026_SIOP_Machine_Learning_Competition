# Second Place: Putka's Army

**Private Test Set Final MSE: 0.010497**

## Team Members

- Ammar Ansari @ HumRRO
- Daniel Barstow @ HumRRO
- Anoop Javalagi @ HumRRO
- Jiayi Liu @ HumRRO
- Karla Castillo-Guerra @ HumRRO
- John Little @ HumRRO
- Robert Wellman @ HumRRO
- Lilang Chen @ HumRRO

**Contact:** Ammar Ansari -- aansari@humrro.org

---

## Approach

This repo automates the extraction and aggregation of Pearson's *r* effect sizes from research PDFs for the SIOP 2026 ML Competition. The full design rationale, challenges, and lessons learned are documented in the accompanying slide deck. In short:

- **Two-agent compression pattern.** A cheap, large-context Gemini extractor compresses each PDF into a small structured JSON of statistics. All compressed JSONs are then sent in a **single** Gemini Deep Research Agent call for cross-study aggregation. This kept us on the free tier through development (~$10 total spend on test set attempts) and avoided per-PDF Deep Research costs that could have run to ~$5/call.
- **Pydantic structured outputs** enforce a strict statistic schema (correlations, *d*, *g*, betas, *t*, *F*, chi-square, odds ratios, eta-squared, 2x2 tables, reliabilities, reverse-coding flags, etc.) to reduce hallucinations.
- **Code-execution-driven aggregation.** The Deep Research Agent performs Fisher *z* -> mean -> inverse Fisher *z* via its code-execution tool to avoid arithmetic hallucinations.
- **Domain expertise in the prompts.** Construct-specific inclusion rules and reverse-coding instructions are injected into both the per-PDF extraction prompt and the batched aggregation prompt.

See the slide deck for the architecture diagram, iterative experiments, and take-home messages.

---

## Required Input Files

The pipeline expects **two CSVs and a folder of PDFs**, all paths configured in the `__main__` block of `final_python_file.py`.

### 1. Articles CSV (`TEST_ARTICLES_CSV_PATH`)

One row per study. Required and optional columns (column names are case-sensitive):

| Column | Required? | Description |
|---|---|---|
| `studyid` | Yes | Unique identifier (e.g., `study1`, `study2`). Used as PDF filename root and JSON key throughout. |
| `Construct1` | Yes | Predictor construct name. **Must exactly match** an entry in the construct-definitions CSV. |
| `Construct2` | Yes | Criterion construct name. **Must exactly match** an entry in the construct-definitions CSV. |
| `research_question` | Optional | Per-study research question. If absent, the script auto-fills it as *"What is the bivariate association between [Construct1] and [Construct2]?"* -- but only when the master prompt template equals that exact string **or** when `autofill_bivariate_relationship_for_research_question=True`. Otherwise the script raises a `KeyError`. |
| `study_filename` | Optional | PDF filename. Defaults to `{studyid}.pdf` if missing. |
| `articletitle`, `citation`, `Google_Scholar_URL` | Optional | Ignored by the pipeline; useful for human bookkeeping. |

**Example** (from the included sample, `test_articles.csv`):

```csv
articletitle,citation,studyid,Google_Scholar_URL,Construct1,Construct2
Understanding Cycles Of Abuse...,"Simon, L. S., Hurst, C., ...",study1,https://scholar.google.com/...,Abusive supervision,Counterproductive workplace behaviors
Thinking Differently...,"Somers, M. J. (2001)...",study2,https://scholar.google.com/...,Role clarity,Role performance
```

> **Encoding note:** The script reads this file as `latin-1` to tolerate smart quotes and accented characters common in citations. Save your CSV accordingly.

### 2. Construct Definitions CSV (`CONSTRUCT_DEFINTIONS_PATH`)

One row per construct used in the articles CSV. Two columns:

| Column | Description |
|---|---|
| `Construct` | Construct name. Must match `Construct1` / `Construct2` values exactly. |
| `Definition` | Long-form inclusion criteria, eligible scales, allowed subscales, and reverse-coding rules for the construct. |

The included sample (`test_construct_definitions.csv`) covers ~45 I/O constructs (Abusive supervision, Burnout, Career satisfaction, Conscientiousness, Job satisfaction, Role clarity, Role performance, ...) and follows this template per construct:

```text
Inclusion Criteria for {Construct} Scales:
{Conceptual definition.}
For this meta-analysis, include any quantitative measure of {construct}. By {construct}, we mean
a construct that captures {list of accepted operationalizations}. Eligible constructs can be the
inverse of it, but the effect sizes must be reverse scored.
Scales qualify whether they target {scope examples}.
Mixed scales are admissible when a distinct {construct} sub-scale or score can be isolated; ...
```

These definitions are injected verbatim into:

- The **per-study extraction system prompt** (used to flag reverse-coded measures and tag which construct each statistic maps to), and
- The **batched Deep Research aggregation prompt** (used to validate that each extracted statistic genuinely maps to the target construct pair).

The pipeline builds a `construct_lookup = {Construct: Definition}` dict and looks up `Construct1` and `Construct2` for each study at runtime -- so any construct name appearing in the articles CSV **must** be a key here, or the system prompt build will fail.

### 3. PDF Folder (`STUDY_DIRECTORY_PATH`)

Plain folder of PDFs. Filenames must match either `{studyid}.pdf` (the default) or the value in the optional `study_filename` column.

---

## Pipeline (in `final_python_file.py`)

`run_DR_agent_loop(...)` is the single entry point. It performs:

1. **Per-study extraction loop** (`produce_json_and_single_prediction`)
   - Builds a construct-pair-specific system prompt from the construct definitions.
   - Calls Gemini with the PDF + `AllPapersResponse` Pydantic schema, medium reasoning, and 5x exponential-backoff retry on 503s.
   - Saves `{studyid}.json` per study.
   - Also runs an immediate per-study aggregate-*r* call (with code execution enabled) as a QC/fallback -> `non_deep_research_results.csv`.

2. **Batched cleanup**
   - Concatenates all per-study JSONs, strips nulls, drops `paper_title` and `page_or_table` to shrink the prompt -> `all_papers_raw_test_agent_2_no_nulls_cleaned.json`.

3. **Single Deep Research Agent call** (`build_single_call_prompt`)
   - Mega-prompt = quick-reference table (`studyid -> construct pair`) + all relevant construct definitions + cleaned JSON + aggregation instructions (Fisher *z*, reverse-coding handling, prefer *r* over derived stats, exclude implausible values, prefer non-overlapping sub-samples over pooled stats, etc.).
   - Dispatched as `client.interactions.create(agent="deep-research-pro-preview-12-2025", background=True, tools=[google_search, code_execution])`, polled every 30s until completion -> `deep_research_filesearch_output.txt`.

4. **Parse -> CSV**
   - A final Gemini call (high reasoning, plain-text output) converts the Deep Research markdown report into raw `studyid,aggregateeffectsize` CSV.

5. **Sample-size-weighted imputation**
   - Sample sizes per study are pulled from the extracted JSON (max `sample_size_total` per sub-study, summed).
   - For any `null` aggregate, impute with the **sample-size-weighted mean *r* across all other studies sharing the same construct pair**, rounded to 2 decimals.
   - Final output -> **`final_deep_research_submission_test_agent_2_weighted_imputed.csv`**.

A second variant with `null -> 0` fallback (`...na_to_0.csv`) is also written.

---

## Requirements

```bash
pip install google-genai pandas numpy pydantic
```

You will also need a Gemini API key (free or paid).

---

## How to Run

1. Set your Gemini API key:

   ```bash
   export GEMINI_API_KEY="your_key_here"
   ```

2. Edit the constants in the `__main__` block of `final_python_file.py`:

   ```python
   MODEL_ID                  = "gemini-3.1-flash-lite-preview"
   STUDY_DIRECTORY_PATH      = "../input_data/.../test_articles"
   OUTPUT_DIRECTORY_PATH     = "../output_data/..."
   TEST_ARTICLES_CSV_PATH    = "../input_data/.../test_articles.csv"
   CONSTRUCT_DEFINTIONS_PATH = "../input_data/.../test_construct_definitions.csv"
   DEEP_RESEARCH_RESEARCH_QUESTION_FOR_BATCH = "What is the bivariate association between [Construct1] and [Construct2]?"
   ```

3. Run the pipeline:

   ```bash
   python final_python_file.py
   ```

---

## Outputs (written to `OUTPUT_DIRECTORY_PATH`)

| File | Purpose |
|---|---|
| `{studyid}.json` | Per-paper extracted statistics (Pydantic-validated) |
| `all_papers_raw_test_agent_2.json` | Concatenated raw extractions |
| `all_papers_raw_test_agent_2_no_nulls_cleaned.json` | Cleaned version sent to Deep Research |
| `deep_research_context_preview.txt` | Exact mega-prompt sent to Deep Research (for debugging/reproducibility) |
| `deep_research_filesearch_output.txt` | Raw Deep Research markdown report |
| `non_deep_research_results.csv` | Per-study QC predictions |
| `debug_responses.json` | Stringified raw model responses |
| `final_deep_research_submission_test_agent_2_na_to_0.csv` | Submission with null->0 fallback |
| **`final_deep_research_submission_test_agent_2_weighted_imputed.csv`** | **Final submission with weighted imputation** |
