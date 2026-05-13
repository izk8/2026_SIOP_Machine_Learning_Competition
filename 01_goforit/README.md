# First Place: goforit

**Private Test Set Final MSE: 0.009233**

## Team Members

- Nga Do @ University of Minnesota
- Michael Hazboun @ University of Minnesota

## Approach

1. **Variable Screening**: For each article, Claude analyzes the Methods section to identify which measured variables match the predictor and outcome definitions for the given construct pair.

2. **Statistics Extraction**: Claude extracts statistics linking matched variables from the article, following a priority hierarchy:
   - Level 1: Bivariate correlations (preferred)
   - Level 2: Derivable statistics (contingency tables, group means, p-values)
   - Level 3: Adjusted regression coefficients (fallback)

3. **Quality Check** (optional): Claude reviews the extracted statistics against the PDF for accuracy and completeness, scoring extraction quality (completeness, accuracy, reverse-coding) and adding any missed statistic pairs. This step can be skipped with `--skip-quality-check` to reduce API costs.

4. **Effect Size Standardization**: All extracted statistics are converted to Pearson correlation coefficients (r) using standard meta-analytic transformations

5. **Aggregation**: Multiple effect sizes from the same study are averaged to produce a single aggregate r value per study.

6. **Fallback Mechanism**: When a study yields no valid correlations, the pipeline falls back to averaging effect sizes from previously-processed studies that share the same predictor-outcome construct pair. This handles edge cases where PDFs are unreadable or papers don't report extractable statistics.

## Requirements

- **Dependencies:**
  - anthropic - Claude API client
  - scipy - Statistical calculations for effect size conversions
  - python-dotenvb- Environment variable management

**Installation:**
pip install anthropic scipy python-dotenv

**Environment Setup:**
Set your Anthropic API key as an environment variable:
export ANTHROPIC_API_KEY="your-api-key-here"

Or create a `.env` file in the project directory with:

ANTHROPIC_API_KEY=your-api-key-here


## How to Run

**Basic usage** (uses default filenames):

python meta_analysis_test_fallback.py


**Full control** (specify all inputs):

python meta_analysis_test_fallback.py \
    --articles test_articles.csv \
    --constructs test_construct_definitions.csv \
    --pdf-dir test_pdfs/ \
    --outcsv submission_test.csv \
    --outdir output_test_fallback/ \
    --fallback-dir output_test_qc/

**Skip quality check** (saves API costs):

python meta_analysis_test_fallback.py --skip-quality-check

**Process a single study**:

python meta_analysis_test_fallback.py --study-id study1


**Input Files:**
- `test_articles.csv` — Study metadata with columns: `studyid`, construct pair reference
- `test_construct_definitions.csv` — Construct definitions with columns: `research_question`, `predictor_name`, `predictor_description`, `outcome_name`, `outcome_description`, `construct_pair_id`
- `test_pdfs/` — Directory containing PDF files named `studyid.pdf`

**Output Files:**
- `submission_test.csv` — Final predictions with columns: `studyid`, `aggregateeffectsize`
- `output_test_fallback/` — Per-study results including JSON artifacts and verbose logs
