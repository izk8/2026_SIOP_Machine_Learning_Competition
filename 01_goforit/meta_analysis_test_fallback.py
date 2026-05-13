"""
meta_analysis_test_fallback.py
-------------------------------
Same pipeline as meta_analysis_test.py, with one addition:

  When a study yields NO valid r values (pipeline returns None), this script
  looks up all previously-processed studies in --fallback-dir (default:
  output_test/) that share the SAME construct pair (predictor → outcome).
  It averages their aggregate_r values and uses that mean as the fallback
  effect size for the study.

  This is identical to meta_analysis_test.py in every other respect.

HOW TO RUN:
  python meta_analysis_test_fallback.py

  # Specify all inputs explicitly:
  python meta_analysis_test_fallback.py \\
      --articles test_articles.csv \\
      --constructs test_construct_definitions.csv \\
      --pdf-dir test_pdfs/ \\
      --outcsv submission_test.csv \\
      --outdir output_test/ \\
      --fallback-dir output_test/

  # Process a single study:
  python meta_analysis_test_fallback.py --study-id study38

REQUIREMENTS:
  pip install anthropic scipy
  Set ANTHROPIC_API_KEY in environment or in a .env file.
"""

import argparse
import base64
import csv
import json
import logging
import math
import os
import re
import sys
import traceback

from anthropic import Anthropic
from dotenv import load_dotenv
from scipy.stats import norm


# ── Logger setup ─────────────────────────────────────────────────────────────

logger = logging.getLogger("meta_analysis_test")
logger.setLevel(logging.DEBUG)

_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_console)

_file_handler = None


def setup_logger(save_dir):
    global _file_handler
    if _file_handler is not None:
        logger.removeHandler(_file_handler)
        _file_handler.close()
    os.makedirs(save_dir, exist_ok=True)
    _file_handler = logging.FileHandler(
        os.path.join(save_dir, "verbose_log.txt"), mode="w", encoding="utf-8"
    )
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_file_handler)


# ── Prompt templates ──────────────────────────────────────────────────────────

SCREENING_PROMPT = """\
You are an expert research methodologist preparing a meta-analysis.

Task: Read the **Methods** section (or equivalent: Measures, Instruments,
Procedure, Materials, Study Design) of this paper.  Identify every measured
variable and determine which ones match the PREDICTOR and OUTCOME
definitions below.

═══════════════════════════════════════════════════════════════════
RESEARCH QUESTION
═══════════════════════════════════════════════════════════════════
{research_question}

═══════════════════════════════════════════════════════════════════
PREDICTOR definition  [{predictor_name}]
═══════════════════════════════════════════════════════════════════
{predictor}

═══════════════════════════════════════════════════════════════════
OUTCOME definition  [{outcome_name}]
═══════════════════════════════════════════════════════════════════
{outcome}

═══════════════════════════════════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════════════════════════════════

1. List every quantitative variable the paper measures.
2. For each variable, note:
   - The exact name used in the paper
   - The instrument / scale used (e.g., "Generalized Trust Scale, 3 items")
   - A one-sentence description of what HIGH SCORES on the final variable
     represent (not what items ask — what the final score means). If the
     scale uses reversed scoring, state the direction AFTER reversal.
3. Decide whether it qualifies as a PREDICTOR match, OUTCOME match,
   or NEITHER, based strictly on the definitions above.
4. If a variable is on the boundary, explain why it does or does not qualify.
5. To set negative_pole, ask: "What do HIGH scores on the FINAL variable
   mean?" (Use the description from step 2 above — which already accounts
   for any reversed scoring.)

   Do NOT derive negative_pole from item content or instrument description
   alone — only from the final scored variable direction.
6. All pairs are studied CROSS-SECTIONALLY. Only include same-wave associations
   (e.g., Time 1 X → Time 1 Y). Never code cross-lagged effects.
7. Apply predictor/outcome definitions LIBERALLY for boundary cases:
   - A subscale qualifies even if its parent instrument is broader, provided
     the subscale's own items substantially target the construct.
   - For interpersonal conflict specifically: any scale or subscale that
     includes items about bullying, harassment, difficult workplace relationships,
     coworker friction, or supervisor antagonism QUALIFIES as a predictor match,
     even if the scale is labelled "relationship quality" or "workplace relationships."
   - When uncertain whether a variable qualifies, classify it as predictor or
     outcome (not "neither") and explain the boundary reasoning in match_rationale.

Return a JSON object with this structure:
{{
  "methods_summary": "brief summary of study design and measures",
  "variables": [
    {{
      "variable_name": "exact name from the paper",
      "instrument": "scale / measure used",
      "description": "what it captures",
      "role": "predictor" | "outcome" | "neither",
      "match_rationale": "why it matches or does not match our definition",
      "negative_pole": true/false
    }}
  ],
  "matched_predictors": ["list of variable names that qualify as PREDICTOR"],
  "matched_outcomes": ["list of variable names that qualify as OUTCOME"]
}}

Return ONLY valid JSON. No other text.
"""


EXTRACTION_PROMPT = """\
You are an expert statistical researcher specialising in meta-analysis.

Task: extract statistics from this paper that link the PREDICTOR to the
OUTCOME defined below.

═══════════════════════════════════════════════════════════════════
RESEARCH QUESTION
═══════════════════════════════════════════════════════════════════
{research_question}

A prior screening step has already identified which variables in this paper
match our predictor and outcome definitions based on the Methods section.
Use ONLY the matched variables listed below when extracting statistics.

NOT EVERY PAIR WILL HAVE A STATISTIC. The screening step identifies which
variables qualify conceptually — but the paper may not report a direct
association between every predictor and every outcome.  Return only statistics
that are actually reported or directly derivable from the paper. Never reuse
the same cell frequencies under a different variable label to fill a gap.

CROSS-SECTIONAL ONLY: All construct pairs are studied cross-sectionally.
Only extract same-wave associations (Time 1 X → Time 1 Y; Time 2 X → Time 2 Y).
Never extract cross-lagged effects (Time 1 X → Time 2 Y).

═══════════════════════════════════════════════════════════════════
MATCHED VARIABLES (from Methods-section screening)
═══════════════════════════════════════════════════════════════════
{matched_variables}

═══════════════════════════════════════════════════════════════════
PRIORITY RULES — follow in strict order; STOP at the first level
that yields at least one valid statistic.
═══════════════════════════════════════════════════════════════════

  Level 1  BIVARIATE CORRELATIONS
     If the paper has a bivariate correlation table (or reports r / rho
     values in text), extract ONLY from there. Do NOT proceed to
     Levels 2–3.
     NOTE: A table that reports only p-values from Pearson correlations
     (without the actual r values) does NOT count as Level 1.  In that
     case, fall through to Level 2 and use "p_only" extraction.

  Level 2  BIVARIATE-DERIVABLE STATISTICS
     If Level 1 yields nothing, look for statistics from which a
     bivariate (unadjusted) effect size can be derived:

     (a) Contingency tables / cross-tabulations:
         If the paper contains a descriptive or frequency table where
         the predictor and the outcome appear as row and column
         categories of THE SAME TABLE, construct a 2×2 contingency
         table and return it as "contingency_2x2".

         CRITICAL GUARDRAILS:
           - The four cell frequencies MUST come from a single cross-
             tabulation where one margin is the predictor and the
             other is the outcome.
           - Dichotomise the PREDICTOR (e.g. high vs. low).
           - Dichotomise the OUTCOME (e.g. high vs. low). When a middle
             category exists (e.g. "undecided"), exclude it.
           - Label cells: a = predictor-high & outcome-positive,
             b = predictor-high & outcome-negative,
             c = predictor-low  & outcome-positive,
             d = predictor-low  & outcome-negative.

         IMPORTANT: When a descriptive table lists several predictors
         in separate row blocks and the columns show categories of a
         shared outcome, each predictor × outcome block IS a valid
         cross-tabulation.  You CAN build a 2×2 from the relevant rows
         × outcome columns.

     IMPORTANT — ANCOVA / MANCOVA F-tests and partial η² are NOT
         Level 2.  These are covariate-adjusted (partial) effects, NOT
         unadjusted bivariate statistics.  However, if the SAME table
         also reports RAW (unadjusted) descriptive means and SDs for
         each group, you CAN extract those as "mean_sd_groups".

     (b) Group means and SDs (return as "mean_sd_groups").
         EXTREME-GROUPS DESIGN: When the study selects participants
         from the tails of a continuous outcome distribution, use
         "mean_sd_groups_extreme" with "n_total_original" and
         "proportion_per_tail" fields.

     (c) Ordinal grouped data (return as "ordinal_groups"):
         When the predictor is an ordinal variable with 3+ ordered
         categories and the paper reports the sample size, mean, and
         SD of the continuous outcome for EACH category.

     (d) Unadjusted t, F, or chi-square tests reported in text.

     (e) p-value-only tables (return as "p_only"):
         Extract each qualifying predictor-outcome pair as a separate
         "p_only" entry with its p-value and the sample size n.

     Do NOT proceed to Level 3 if Level 2 yields anything.

  Level 3  ADJUSTED REGRESSION COEFFICIENTS (last resort)
     If Levels 1 and 2 both yield nothing, extract from regression
     tables (OR, RRR, beta, b, AME, etc.).  These are partial /
     adjusted effects and less preferred for meta-analysis.

     T-STATISTICS IN REGRESSION TABLES: When a paper reports the
     regression coefficient in the cell and the t-statistic in
     parentheses (check the table note), extract the t-statistic
     as statistic_type "t" with df = n − k if derivable.

     DEDUPLICATION for regression ORs / RRRs:
       - For categorical predictors with k > 2 levels, extract ONLY
         the single contrast that best captures the full high-vs-low split.
       - When the same association appears in multiple models, extract
         ONLY the least-adjusted (most bivariate) model.
       - When data are reported at multiple waves, extract ONE effect
         per unique predictor–outcome pair per wave.

═══════════════════════════════════════════════════════════════════
OUTPUT SCHEMA
═══════════════════════════════════════════════════════════════════

For each statistic, return a JSON object with these keys:
  "predictor_variable" : exact variable name from the article
  "outcome_variable"   : exact variable name from the article
  "statistic_type"   : one of "r", "partial_r", "t", "F", "d", "g",
                       "beta_std", "OR", "log_OR", "chi2", "eta2",
                       "b_unstd", "ame", "mean_sd_groups",
                       "mean_sd_groups_extreme", "ordinal_groups",
                       "contingency_2x2", "p_only", "other"
  "statistic_value"  : numeric value (float); null for contingency_2x2
                       and p_only
  "n"                : sample size (integer or null)
  "df"               : degrees of freedom (or null)
  "p"                : reported p-value (or null)
  "direction"        : "positive", "negative", or "unclear"
  "reverse_coded"    : true if an odd number of predictor/outcome scales
                       are negative-pole
  "standard_error"   : reported SE (for b_unstd), or null
  "ci_lower"         : lower bound of 95% CI (or null)
  "ci_upper"         : upper bound of 95% CI (or null)
  "source_location"  : where in the paper this statistic appears
  "extraction_level" : 1, 2, or 3

For "mean_sd_groups": include mean1, sd1, n1, mean2, sd2, n2.
For "mean_sd_groups_extreme": same as mean_sd_groups PLUS
  n_total_original, proportion_per_tail.
For "ordinal_groups": include "groups" array of
  {{"label": "...", "n": int, "mean": float, "sd": float}} from lowest to
  highest predictor value, plus optional "weights_note".
For "p_only": set "statistic_value" to null. "p" and "n" are required.
For "contingency_2x2": include cell_a, cell_b, cell_c, cell_d,
  row_labels, col_labels.

═══════════════════════════════════════════════════════════════════
RULES
═══════════════════════════════════════════════════════════════════
  - Only include DIRECT predictor-to-outcome associations.
  - Include qualifying associations even if non-significant.
  - ALL returned statistics must be from the SAME priority level.
  - Omit any value you are uncertain about.
  - Return ONLY a valid JSON array.  No other text.
  - Return [] if no qualifying statistics are found.
  - CRITICAL: In your output JSON, the "predictor_variable" and
    "outcome_variable" fields MUST use the EXACT variable name in
    quotes from the MATCHED VARIABLES section above.

Predictor: {predictor}
Outcome: {outcome}

The PDF is attached above. Return only a JSON array.
"""


QUALITY_CHECK_PROMPT = """\
You are an expert meta-analyst auditing the output of an automated \
statistics extraction pipeline.

The pipeline extracted statistics from the attached PDF for the following \
research question:

═══════════════════════════════════════════════════════════════════
RESEARCH QUESTION
═══════════════════════════════════════════════════════════════════
{research_question}

PREDICTOR [{predictor_name}]: {predictor}
OUTCOME   [{outcome_name}]:   {outcome}

═══════════════════════════════════════════════════════════════════
SCREENING RESULT (pipeline Step 1 — Methods section scan)
═══════════════════════════════════════════════════════════════════
Matched predictors : {matched_predictors}
Matched outcomes   : {matched_outcomes}

═══════════════════════════════════════════════════════════════════
EXTRACTED STATISTICS (pipeline Step 2)
═══════════════════════════════════════════════════════════════════
{extracted_stats_json}

═══════════════════════════════════════════════════════════════════
YOUR AUDIT TASK
═══════════════════════════════════════════════════════════════════
Carefully re-read the full PDF and audit the pipeline output above.
Evaluate each of the following five dimensions and assign a score 0–10
(10 = perfect, 0 = completely wrong/missing).

1. COMPLETENESS (0-10)
   Are all relevant predictor→outcome statistics present in the extraction?
   A paper may report the association in multiple samples, tables, or models.
   Identify any qualifying statistics that appear in the paper but are absent
   from the extracted list, including their exact location.
   LONGITUDINAL DESIGNS: Only same-wave (cross-sectional) correlations
   qualify — predictor and outcome at the SAME time point. Cross-lagged
   correlations (predictor at t1 with outcome at t2, etc.) must NOT be
   included. Flag any cross-lagged correlations in the extraction as
   inaccurate_statistics (field: "statistic_type", issue: "cross-lagged,
   should be excluded").

2. ACCURACY (0-10)
   Are the extracted numeric values (r, t, F, n, p, df, ci_lower, ci_upper)
   correct as reported in the paper?  Flag any mismatches with the correct
   value and its location.

3. VARIABLE IDENTIFICATION (0-10)
   Were the right variables classified as predictor/outcome in Step 1?
   Were any qualifying variables missed (false negatives)?
   Were any non-qualifying variables incorrectly included (false positives)?

4. EXTRACTION LEVEL (0-10)
   Priority: Level 1 (bivariate r) > Level 2 (t, F, means/SDs, contingency)
   > Level 3 (adjusted regression).
   Was the correct level chosen?  Could a higher-priority statistic have been
   used instead of what was extracted?

5. REVERSE CODING (0-10)
   Is the reverse_coded flag set correctly for every extracted statistic?
   A statistic should be reverse_coded=true only if an ODD number of the
   predictor/outcome variables are negative-pole scales.

RULES FOR REPORTING INACCURATE STATISTICS:

When flagging a value as wrong:
- If you can determine the correct numeric value with confidence, put
  the number in `correct_value` and set `needs_human_review` to false.
- If you cannot determine the correct value with confidence (e.g. the
  table layout is ambiguous, the cell is hard to read, or there are
  multiple plausible readings), set `correct_value` to null and
  `needs_human_review` to true, and explain why in `review_reason`.


Return a JSON object with EXACTLY this structure:
{{
  "scores": {{
    "completeness": <integer 0-10>,
    "accuracy": <integer 0-10>,
    "variable_identification": <integer 0-10>,
    "extraction_level": <integer 0-10>,
    "reverse_coding": <integer 0-10>
  }},
  "dimension_reasoning": {{
    "completeness": "concise explanation of score",
    "accuracy": "concise explanation of score",
    "variable_identification": "concise explanation of score",
    "extraction_level": "concise explanation of score",
    "reverse_coding": "concise explanation of score"
  }},
  "per_stat_accuracy": [
    {{
      "predictor_variable": "exact name matching extracted statistic",
      "outcome_variable": "exact name matching extracted statistic",
      "accuracy_score": <integer 0-10>,
      "notes": "one-sentence note on correctness"
    }}
  ],
  "missing_statistics": [
    {{
      "predictor_variable": "exact name as in paper",
      "outcome_variable": "exact name as in paper",
      "statistic_type": "r/t/F/d/etc",
      "statistic_value": <number or null>,
      "n": <integer or null>,
      "location": "Table X / page Y / Section Z",
      "issue": "why it is missing or what is wrong"
    }}
  ],

  "inaccurate_statistics": [
    {{
      "predictor_variable": "exact name",
      "outcome_variable": "exact name",
      "field": "which field is wrong",
      "extracted_value": <the number the pipeline returned, or string for non-numeric fields>,
      "correct_value": <a number for numeric fields, a string for non-numeric fields, or null if you cannot determine it with confidence>,
      "needs_human_review": <true if you are uncertain and a human should verify; false if you are confident>,
      "review_reason": "short explanation — required if needs_human_review is true or correct_value is null",
      "location": "where in the paper"
    }}
  ],

  "improvement_suggestions": [
    "specific, actionable suggestion referencing the paper location or field"
  ],
  "verified_correct": [
    "predictor -> outcome: brief note confirming it is correct"
  ]
}}

Return ONLY valid JSON. No other text.
"""


# ── Step 2.5: Quality check ───────────────────────────────────────────────────

def run_quality_check(pdf_b64, client, model, construct_def, stats, screening):
    prompt = QUALITY_CHECK_PROMPT.format(
        research_question=construct_def["research_question"],
        predictor_name=construct_def["predictor_name"],
        predictor=construct_def["predictor_description"],
        outcome_name=construct_def["outcome_name"],
        outcome=construct_def["outcome_description"],
        matched_predictors=json.dumps(screening.get("matched_predictors", [])),
        matched_outcomes=json.dumps(screening.get("matched_outcomes", [])),
        extracted_stats_json=json.dumps(stats, indent=2),
    )

    response = client.messages.create(
        model=model,
        max_tokens=8192,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )

    raw = response.content[0].text if response.content else ""
    logger.debug("── RAW QUALITY CHECK RESPONSE ──")
    logger.debug(raw)
    logger.debug("── END QUALITY CHECK RESPONSE ──")

    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    quality = None
    try:
        quality = json.loads(clean)
    except json.JSONDecodeError as e:
        logger.debug(f"  QC JSONDecodeError (direct): {e}")
        match = re.search(r"\{[\s\S]+\}", clean)
        if match:
            try:
                quality = json.loads(match.group())
            except json.JSONDecodeError as e2:
                logger.warning(f"  Could not parse quality check response: {e2}")

    if quality is None:
        logger.warning("  Quality check parse failed; returning empty report")
        quality = {}

    return quality


def apply_quality_corrections(stats, quality):
    """Patch inaccurate fields and append missing statistics from the quality report."""
    corrected = [s.copy() for s in stats]

    numeric_fields = {
        "statistic_value", "n", "df", "p",
        "standard_error", "ci_lower", "ci_upper",
    }

    for fix in quality.get("inaccurate_statistics", []):
        pred = (fix.get("predictor_variable") or "").strip().lower()
        outc = (fix.get("outcome_variable") or "").strip().lower()
        field = fix.get("field", "")
        correct_val = fix.get("correct_value")
        extracted_raw = fix.get("extracted_value")
        needs_review = bool(fix.get("needs_human_review"))
        review_reason = fix.get("review_reason") or fix.get("issue") or ""

        if not field:
            continue

        coerced_val = None
        coercion_ok = False
        if correct_val is not None:
            if field in numeric_fields:
                try:
                    coerced_val = float(correct_val)
                    coercion_ok = True
                except (TypeError, ValueError):
                    coercion_ok = False
            else:
                coerced_val = correct_val
                coercion_ok = True

        for s in corrected:
            if not (s.get("predictor_variable", "").lower() == pred and
                    s.get("outcome_variable", "").lower() == outc and
                    field in s):
                continue

            if extracted_raw is not None:
                current = s.get(field)
                try:
                    if field in numeric_fields:
                        if abs(float(current) - float(extracted_raw)) > 1e-9:
                            continue
                    else:
                        if str(current).strip().lower() != str(extracted_raw).strip().lower():
                            continue
                except (TypeError, ValueError):
                    pass

            if coercion_ok and not needs_review:
                s[field] = coerced_val
                logger.info(
                    f"    [QC fix] {field} for '{s.get('predictor_variable')}'"
                    f" → '{s.get('outcome_variable')}': "
                    f"{extracted_raw} → {coerced_val}"
                )
            else:
                s.setdefault("qc_warnings", []).append({
                    "field": field,
                    "extracted_value": extracted_raw,
                    "proposed_correction": correct_val,
                    "reason": review_reason or "QC flagged but could not produce numeric correction",
                })
                s["needs_human_review"] = True
                logger.warning(
                    f"    [QC flag] {field} for '{s.get('predictor_variable')}'"
                    f" → '{s.get('outcome_variable')}': kept original {extracted_raw}, "
                    f"flagged for review ({review_reason[:80] or 'no reason given'})"
                )

    required = {"predictor_variable", "outcome_variable", "statistic_type"}
    for miss in quality.get("missing_statistics", []):
        if not required.issubset(miss.keys()):
            continue
        if miss.get("statistic_value") is None:
            continue
        new_stat = {
            "predictor_variable": miss["predictor_variable"],
            "outcome_variable":   miss["outcome_variable"],
            "statistic_type":     miss["statistic_type"],
            "statistic_value":    miss.get("statistic_value"),
            "n":                  miss.get("n"),
            "df":                 None,
            "p":                  None,
            "direction":          "unclear",
            "reverse_coded":      False,
            "standard_error":     None,
            "ci_lower":           None,
            "ci_upper":           None,
            "source_location":    miss.get("location", "quality-check addition"),
            "extraction_level":   1,
            "qc_added":           True,
        }
        corrected.append(new_stat)
        logger.info(
            f"    [QC add] '{miss['predictor_variable']}' → "
            f"'{miss['outcome_variable']}' "
            f"({miss['statistic_type']}={miss.get('statistic_value')})"
        )

    return corrected


# ── Helper: flexible column lookup ───────────────────────────────────────────

def get_col(row, *candidates, default=None):
    for col in candidates:
        if col in row and row[col].strip():
            return row[col].strip()
    return default


def detect_encoding(csv_path):
    for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            with open(csv_path, encoding=encoding) as f:
                f.read()
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"


# ── Load construct definitions ────────────────────────────────────────────────

def load_construct_definitions(csv_path):
    definitions = {}
    with open(csv_path, newline="", encoding=detect_encoding(csv_path)) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = get_col(row, "Construct", "construct", "name")
            defn = get_col(row, "Definition", "definition", "description", default="")
            if name:
                definitions[name] = defn

    logger.info(f"  Loaded {len(definitions)} construct definition(s)")
    return definitions


# ── Load article list ─────────────────────────────────────────────────────────

def load_articles(csv_path, pdf_dir=None):
    articles = []

    with open(csv_path, newline="", encoding=detect_encoding(csv_path)) as f:
        reader = csv.DictReader(f)
        for row in reader:
            study_id = get_col(row, "studyid", "study_id", "id")
            if not study_id:
                continue

            construct1 = get_col(row, "Construct1", "construct1", "x_construct", "predictor")
            construct2 = get_col(row, "Construct2", "construct2", "y_construct", "outcome")

            pdf_path = None
            if pdf_dir:
                candidate = os.path.join(pdf_dir, f"{study_id}.pdf")
                if os.path.exists(candidate):
                    pdf_path = candidate

            articles.append({
                "study_id": study_id,
                "construct1": construct1,
                "construct2": construct2,
                "pdf_path": pdf_path,
            })

    logger.info(f"  Loaded {len(articles)} article(s)")
    return articles


# ── Fallback index ────────────────────────────────────────────────────────────

def load_fallback_index(fallback_dir):
    """Scan fallback_dir for result.json files and build a mapping:
      pair_id (normalised) -> list of (study_id, aggregate_r) tuples
    Only includes studies that produced a non-None aggregate_r.
    """
    index = {}  # pair_id_key -> [(study_id, r), ...]

    if not fallback_dir or not os.path.isdir(fallback_dir):
        return index

    for entry in os.scandir(fallback_dir):
        if not entry.is_dir():
            continue
        result_path = os.path.join(entry.path, "result.json")
        if not os.path.exists(result_path):
            continue
        try:
            with open(result_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        agg_r = data.get("aggregate_r")
        if agg_r is None:
            continue

        pair_id = (data.get("pair_id") or "").strip()
        if not pair_id:
            continue

        key = pair_id.lower()
        index.setdefault(key, []).append((entry.name, float(agg_r), pair_id))

    n_pairs = len(index)
    n_studies = sum(len(v) for v in index.values())
    logger.info(
        f"  Fallback index loaded: {n_studies} studies across {n_pairs} construct pair(s)"
    )
    return index


def compute_fallback_r(pair_id, fallback_index, exclude_study_id=None):
    """Return the mean aggregate_r across all studies in fallback_index that share
    the same pair_id, excluding exclude_study_id. Returns (mean_r, sources) or
    (None, []) if no matching studies exist.
    """
    key = (pair_id or "").strip().lower()
    entries = fallback_index.get(key, [])
    valid = [
        (sid, r, pid) for sid, r, pid in entries
        if exclude_study_id is None or sid != exclude_study_id
    ]
    if not valid:
        return None, []

    mean_r = sum(r for _, r, _ in valid) / len(valid)
    sources = [{"study_id": sid, "aggregate_r": r} for sid, r, _ in valid]
    return mean_r, sources


# ── Step 1: Screen variables ──────────────────────────────────────────────────

def screen_variables(pdf_b64, client, model, construct_def):
    prompt = SCREENING_PROMPT.format(
        research_question=construct_def["research_question"],
        predictor_name=construct_def["predictor_name"],
        predictor=construct_def["predictor_description"],
        outcome_name=construct_def["outcome_name"],
        outcome=construct_def["outcome_description"],
    )

    logger.info("  [Step 1] Screening Methods section for matching variables...")
    response = client.messages.create(
        model=model,
        max_tokens=16000,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )

    raw_text = response.content[0].text if response.content else ""
    logger.debug("── RAW SCREENING RESPONSE ──")
    logger.debug(raw_text)
    logger.debug("── END SCREENING RESPONSE ──")

    clean = re.sub(r"```(?:json)?", "", raw_text).strip().rstrip("`").strip()
    try:
        screening = json.loads(clean)
    except json.JSONDecodeError as e:
        logger.debug(f"  JSONDecodeError (direct): {e}")
        match = re.search(r"\{[\s\S]+\}", clean)
        if match:
            try:
                screening = json.loads(match.group())
            except json.JSONDecodeError as e2:
                logger.warning(f"  Could not parse screening response as JSON: {e2}")
                screening = {}
        else:
            logger.warning(f"  Could not parse screening response as JSON: {e}")
            screening = {}

    methods_summary = screening.get("methods_summary", "N/A")
    logger.info(f"    Methods summary: {methods_summary}")

    variables = screening.get("variables", [])
    matched_preds = screening.get("matched_predictors", [])
    matched_outcs = screening.get("matched_outcomes", [])

    for v in variables:
        role = v.get("role", "?")
        name = v.get("variable_name", "?")
        instrument = v.get("instrument", "?")
        rationale = v.get("match_rationale", "")
        neg = v.get("negative_pole", False)
        marker = "✓" if role in ("predictor", "outcome") else "✗"
        logger.info(f"    {marker} {name} [{role}] — {instrument}")
        logger.info(f"      Rationale: {rationale}")
        if neg:
            logger.info(f"      ⚠ Negative-pole (reverse-coded)")

    logger.info(f"    Matched predictors: {matched_preds}")
    logger.info(f"    Matched outcomes:   {matched_outcs}")

    if not matched_preds or not matched_outcs:
        logger.info("    ⚠ No matching predictor-outcome pair found in Methods section")

    return screening


# ── Step 2: Extract statistics ────────────────────────────────────────────────

def extract_stats_from_pdf(pdf_path, client, model, construct_def):
    with open(pdf_path, "rb") as f:
        pdf_b64 = base64.b64encode(f.read()).decode("ascii")

    screening = screen_variables(pdf_b64, client, model, construct_def)

    matched_preds = screening.get("matched_predictors", [])
    matched_outcs = screening.get("matched_outcomes", [])

    if not matched_preds and not matched_outcs:
        logger.info("  No qualifying variables found — skipping extraction")
        return [], screening

    matched_lines = []
    for v in screening.get("variables", []):
        if v.get("role") in ("predictor", "outcome"):
            neg_tag = " [NEGATIVE-POLE]" if v.get("negative_pole") else ""
            matched_lines.append(
                f"  - \"{v.get('variable_name', '?')}\" [{v.get('role').upper()}]"
                f" ({v.get('instrument', '?')}){neg_tag}"
                f"\n    Rationale: {v.get('match_rationale', 'N/A')}"
            )
    matched_variables_text = "\n".join(matched_lines) if matched_lines else "None identified"

    logger.info("  [Step 2] Extracting statistics for matched variables...")

    prompt = EXTRACTION_PROMPT.format(
        research_question=construct_def["research_question"],
        matched_variables=matched_variables_text,
        predictor=construct_def["predictor_description"],
        outcome=construct_def["outcome_description"],
    )

    response = client.messages.create(
        model=model,
        max_tokens=16000,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )

    raw_text = response.content[0].text if response.content else ""
    logger.debug("── RAW EXTRACTION RESPONSE ──")
    logger.debug(raw_text)
    logger.debug("── END EXTRACTION RESPONSE ──")

    clean = re.sub(r"```(?:json)?", "", raw_text).strip().rstrip("`").strip()
    stats = None

    try:
        stats = json.loads(clean)
    except json.JSONDecodeError:
        pass

    if stats is None or not isinstance(stats, list):
        candidates = []
        depth = 0
        arr_start = None
        for i, ch in enumerate(clean):
            if ch == "[" and depth == 0:
                arr_start = i
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0 and arr_start is not None:
                    try:
                        parsed = json.loads(clean[arr_start:i + 1])
                        if isinstance(parsed, list):
                            candidates.append(parsed)
                    except json.JSONDecodeError:
                        pass
                    arr_start = None

        if candidates:
            stats = candidates[-1]
            if len(candidates) > 1:
                logger.info(
                    f"    ⚠ Found {len(candidates)} JSON arrays in response; "
                    f"using last one ({len(stats)} items)"
                )
        else:
            logger.warning("  Could not parse LLM response as JSON")
            stats = []

    stats = [s for s in stats if isinstance(s, dict)]

    LEVEL_1_TYPES = {"r", "partialr"}
    LEVEL_2_TYPES = {"contingency2x2", "meansdgroups", "meansdgroupsextreme",
                     "ordinalgroups", "ponly", "t", "f", "chi2", "d", "g"}

    for s in stats:
        if "extraction_level" not in s or s["extraction_level"] is None:
            stype = (s.get("statistic_type") or "").lower().replace("-", "").replace("_", "")
            if stype in LEVEL_1_TYPES:
                s["extraction_level"] = 1
            elif stype in LEVEL_2_TYPES:
                s["extraction_level"] = 2
            else:
                s["extraction_level"] = 3

    levels_present = {s["extraction_level"] for s in stats}
    if len(levels_present) > 1:
        best = min(levels_present)
        n_before = len(stats)
        stats = [s for s in stats if s["extraction_level"] == best]
        logger.info(
            f"    ⚠ Mixed extraction levels {levels_present} detected; "
            f"keeping only level {best} ({n_before} → {len(stats)} stats)"
        )

    valid_stats = []
    for s in stats:
        stype = (s.get("statistic_type") or "").lower().replace("-", "").replace("_", "")
        if stype == "contingency2x2":
            cells = [s.get("cell_a"), s.get("cell_b"),
                     s.get("cell_c"), s.get("cell_d")]
            if any(c is None for c in cells):
                logger.info(
                    f"    ⚠ Dropping contingency_2x2 with null cells: "
                    f"{s.get('note', '?')}"
                )
                continue
        valid_stats.append(s)
    stats = valid_stats

    valid_stats = []
    for s in stats:
        stype = (s.get("statistic_type") or "").lower().replace("-", "").replace("_", "")
        if stype == "ordinalgroups":
            groups = s.get("groups")
            if not groups or not isinstance(groups, list) or len(groups) < 3:
                logger.info(f"    ⚠ Dropping ordinal_groups with <3 groups")
                continue
            missing = False
            for g in groups:
                if any(g.get(k) is None for k in ("n", "mean", "sd")):
                    missing = True
                    break
            if missing:
                logger.info(f"    ⚠ Dropping ordinal_groups with incomplete data")
                continue
        valid_stats.append(s)
    stats = valid_stats

    seen_cells = set()
    deduped_stats = []
    for s in stats:
        stype = (s.get("statistic_type") or "").lower().replace("-", "").replace("_", "")
        if stype == "contingency2x2":
            cell_key = (s.get("cell_a"), s.get("cell_b"),
                        s.get("cell_c"), s.get("cell_d"))
            if cell_key in seen_cells:
                logger.info(
                    f"    ⚠ Dropping duplicate contingency_2x2: "
                    f"{s.get('outcome_variable', '?')}"
                )
                continue
            seen_cells.add(cell_key)
        deduped_stats.append(s)
    stats = deduped_stats

    neg_pole_vars = set()
    for v in screening.get("variables", []):
        if v.get("negative_pole"):
            neg_pole_vars.add(v.get("variable_name", "").lower().strip())

    for s in stats:
        pred_name = (s.get("predictor_variable") or "").lower().strip()
        outc_name = (s.get("outcome_variable") or "").lower().strip()
        n_neg = sum(
            1 for name in (pred_name, outc_name)
            if any(nv in name or name in nv for nv in neg_pole_vars)
        )
        old_rev = s.get("reverse_coded", False)
        correct_rev = (n_neg % 2 == 1)
        if old_rev != correct_rev:
            logger.info(
                f"    ⚠ Fixing reverse_coded for {s.get('predictor_variable')} -> "
                f"{s.get('outcome_variable')}: {old_rev} → {correct_rev} "
                f"({n_neg} negative-pole variable(s))"
            )
            s["reverse_coded"] = correct_rev

    for i, s in enumerate(stats):
        pred = s.get("predictor_variable", "?")
        outc = s.get("outcome_variable", "?")
        stype = s.get("statistic_type", "?")
        sval = s.get("statistic_value", "?")
        n = s.get("n", "?")
        rev = s.get("reverse_coded", False)
        dirn = s.get("direction", "?")
        logger.info(
            f"    [{i+1}] {pred} -> {outc}  |  "
            f"{stype}={sval}, n={n}, dir={dirn}"
            f"{', REVERSE-CODED' if rev else ''}"
        )
        logger.debug(f"    [{i+1}] full: {json.dumps(s, ensure_ascii=False)}")

    return stats, screening


# ── Convert any statistic to Pearson r ───────────────────────────────────────

def convert_to_r(stat):
    def to_float(x):
        if x is None:
            return None
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    stype = (stat.get("statistic_type") or "").lower().replace("-", "").replace("_", "")
    val = to_float(stat.get("statistic_value"))
    n = stat.get("n")
    df = stat.get("df")
    direction = stat.get("direction", "unclear")
    reverse = bool(stat.get("reverse_coded", False))

    r = None
    formula = None

    if stype in ("r", "partialr"):
        r = val
        formula = f"r = {val} (direct)"

    elif stype == "t" and val is not None:
        if n:
            r = math.sqrt(val ** 2 / (val ** 2 + float(n) - 2))
            formula = (
                f"r = sqrt(t² / (t² + N - 2)) = sqrt({val}² / ({val}² + {n} - 2))"
                f" = {r:.4f}"
            )

    elif stype == "f" and val is not None:
        d2 = stat.get("df2") or (n - 2 if n else None)
        if d2:
            r = math.sqrt(abs(val) / (abs(val) + float(d2)))
            formula = (
                f"r = sqrt(F / (F + df2)) = sqrt({val} / ({val} + {d2}))"
                f" = {r:.4f}"
            )

    elif stype in ("d", "g") and val is not None:
        r = val / math.sqrt(val ** 2 + 4.0)
        formula = (
            f"r = d / sqrt(d² + 4) = {val} / sqrt({val}² + 4)"
            f" = {r:.4f}"
        )

    elif stype == "betastd" and val is not None:
        r = val
        formula = f"r ≈ β_std = {val} (approximation: standardised beta ≈ r)"

    elif stype in ("bunstd", "ame") and val is not None:
        se = to_float(stat.get("standard_error"))
        se_source = "reported SE"
        if se is None or se <= 0:
            ci_lo = to_float(stat.get("ci_lower"))
            ci_hi = to_float(stat.get("ci_upper"))
            if ci_lo is not None and ci_hi is not None and ci_hi > ci_lo:
                se = (ci_hi - ci_lo) / 3.92
                se_source = f"SE = (CI_upper - CI_lower) / 3.92 = ({ci_hi} - {ci_lo}) / 3.92 = {se:.4f}"
        if se and se > 0 and n:
            z_stat = val / se
            r = z_stat / math.sqrt(float(n))
            formula = (
                f"z = b / SE = {val} / {se:.4f} = {z_stat:.4f} [{se_source}]; "
                f"r = z / sqrt(N) = {z_stat:.4f} / sqrt({n}) = {r:.4f}"
            )

    elif stype in ("or", "rr") and val is not None and val > 0:
        log_or = math.log(val)
        d = log_or * math.sqrt(3) / math.pi
        r = d / math.sqrt(d ** 2 + 4.0)
        formula = (
            f"ln(OR) = ln({val}) = {log_or:.4f}; "
            f"d = ln(OR) * sqrt(3) / π = {log_or:.4f} * sqrt(3) / π = {d:.4f}; "
            f"r = d / sqrt(d² + 4) = {d:.4f} / sqrt({d:.4f}² + 4) = {r:.4f}"
        )

    elif stype == "logor" and val is not None:
        d = val * math.sqrt(3) / math.pi
        r = d / math.sqrt(d ** 2 + 4.0)
        formula = (
            f"d = log_OR * sqrt(3) / π = {val} * sqrt(3) / π = {d:.4f}; "
            f"r = d / sqrt(d² + 4) = {d:.4f} / sqrt({d:.4f}² + 4) = {r:.4f}"
        )

    elif stype == "chi2" and val is not None and n:
        r = math.sqrt(abs(val) / float(n))
        formula = f"r = sqrt(χ² / N) = sqrt({val} / {n}) = {r:.4f}"

    elif stype == "eta2" and val is not None:
        r = math.sqrt(abs(val))
        formula = f"r = sqrt(η²) = sqrt({val}) = {r:.4f}"

    elif stype in ("meansdgroups", "meansdgroupsextreme"):
        m1 = to_float(stat.get("mean1"))
        s1 = to_float(stat.get("sd1"))
        n1 = to_float(stat.get("n1"))
        m2 = to_float(stat.get("mean2"))
        s2 = to_float(stat.get("sd2"))
        n2 = to_float(stat.get("n2"))
        if None not in (m1, s1, n1, m2, s2, n2) and n1 > 1 and n2 > 1:
            pooled = math.sqrt(
                ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
            )
            if pooled > 0:
                d = (m1 - m2) / pooled
                r = d / math.sqrt(d ** 2 + 4.0)
                formula = (
                    f"pooled_SD={pooled:.4f}; d={d:.4f}; r={r:.4f}"
                )
                if stype == "meansdgroupsextreme":
                    p_tail = to_float(stat.get("proportion_per_tail"))
                    if p_tail and 0 < p_tail < 0.5:
                        z_cut = abs(norm.ppf(p_tail))
                        h = (1.0 / math.sqrt(2 * math.pi)) * math.exp(
                            -0.5 * z_cut ** 2
                        )
                        if h > 0:
                            correction = p_tail / h
                            r_uncorrected = r
                            r = r * correction
                            formula += (
                                f"; Feldt correction: z_cut={z_cut:.4f}, "
                                f"h={h:.4f}, correction={correction:.4f}; "
                                f"r_corrected={r:.4f}"
                            )

    elif stype == "ordinalgroups":
        groups = stat.get("groups")
        if groups and isinstance(groups, list) and len(groups) >= 3:
            ns_g, means_g, sds_g = [], [], []
            valid = True
            for g in groups:
                gn = to_float(g.get("n"))
                gm = to_float(g.get("mean"))
                gs = to_float(g.get("sd"))
                if gn is None or gm is None or gs is None or gn < 1:
                    valid = False
                    break
                ns_g.append(gn)
                means_g.append(gm)
                sds_g.append(gs)
            if valid:
                K = len(ns_g)
                codes = list(range(1, K + 1))
                N_total = sum(ns_g)
                x_bar = sum(ns_g[k] * codes[k] for k in range(K)) / N_total
                y_bar = sum(ns_g[k] * means_g[k] for k in range(K)) / N_total
                cov_xy = sum(
                    ns_g[k] * (codes[k] - x_bar) * (means_g[k] - y_bar)
                    for k in range(K)
                ) / N_total
                var_x = sum(
                    ns_g[k] * (codes[k] - x_bar) ** 2 for k in range(K)
                ) / N_total
                var_y_between = sum(
                    ns_g[k] * (means_g[k] - y_bar) ** 2 for k in range(K)
                ) / N_total
                var_y_within = sum(
                    ns_g[k] * sds_g[k] ** 2 for k in range(K)
                ) / N_total
                var_y = var_y_between + var_y_within
                if var_x > 0 and var_y > 0:
                    r = cov_xy / math.sqrt(var_x * var_y)
                    if not n:
                        n = int(N_total)
                    formula = (
                        f"Weighted Pearson r from {K} ordinal groups; "
                        f"N={N_total}, r={r:.4f}"
                    )

    elif stype == "contingency2x2":
        a = to_float(stat.get("cell_a"))
        b = to_float(stat.get("cell_b"))
        c = to_float(stat.get("cell_c"))
        d_cell = to_float(stat.get("cell_d"))
        if None not in (a, b, c, d_cell) and b * c > 0:
            odds_ratio = (a * d_cell) / (b * c)
            if odds_ratio > 0:
                log_or = math.log(odds_ratio)
                d = log_or * math.sqrt(3) / math.pi
                r = d / math.sqrt(d ** 2 + 4.0)
                if not n:
                    n = int(a + b + c + d_cell)
                formula = (
                    f"OR={odds_ratio:.4f}; ln(OR)={log_or:.4f}; "
                    f"d={d:.4f}; r={r:.4f}"
                )

    if r is None and stype in ("ponly", "other"):
        p = to_float(stat.get("p"))
        if p is not None and n is not None and 0 < p < 1 and n > 0:
            z = abs(norm.ppf(p / 2))
            r = z / math.sqrt(float(n))
            formula = (
                f"z = |Φ⁻¹(p/2)| = |Φ⁻¹({p}/2)| = {z:.4f}; "
                f"r = z / sqrt(N) = {z:.4f} / sqrt({n}) = {r:.4f}"
            )

    if r is None:
        return None, None

    sign_adjustments = []
    if direction == "negative" and r > 0:
        r = -r
        sign_adjustments.append("sign flipped (direction=negative)")
    elif direction == "positive" and r < 0:
        r = -r
        sign_adjustments.append("sign flipped (r was negative but direction=positive)")

    if reverse:
        r = -r
        sign_adjustments.append("sign flipped (reverse_coded=true)")

    if sign_adjustments:
        formula = (formula or "") + "; " + "; ".join(sign_adjustments)
        formula += f" → final r = {r:.4f}"

    if not (-1.0 <= r <= 1.0):
        return None, None

    return r, formula


# ── Build construct_def for a study ──────────────────────────────────────────

def build_construct_def(article, definitions):
    c1 = article.get("construct1")
    c2 = article.get("construct2")

    if not c1 or not c2:
        logger.warning(f"  Article {article['study_id']} is missing Construct1 or Construct2")
        return None

    def1 = definitions.get(c1)
    def2 = definitions.get(c2)

    if def1 is None:
        logger.warning(f"  No definition found for Construct1={c1!r}")
        return None
    if def2 is None:
        logger.warning(f"  No definition found for Construct2={c2!r}")
        return None

    research_question = (
        f"What is the bivariate association between {c1} and {c2}?"
    )

    return {
        "pair_id": f"{c1} → {c2}",
        "research_question": research_question,
        "predictor_name": c1,
        "predictor_description": def1,
        "outcome_name": c2,
        "outcome_description": def2,
    }


# ── Process one study ─────────────────────────────────────────────────────────

def process_one_study(article, construct_def, api_key, model, save_dir,
                      skip_quality_check=False):
    study_id = article["study_id"]
    pdf_path = article.get("pdf_path")

    if not pdf_path or not os.path.exists(pdf_path):
        logger.warning(f"  PDF not found for {study_id}: {pdf_path}")
        return None

    pdf_path = os.path.abspath(pdf_path)
    os.makedirs(save_dir, exist_ok=True)
    setup_logger(save_dir)

    logger.info(f"  Construct pair: {construct_def['pair_id']}")
    logger.info(f"  Research question: {construct_def['research_question']}")
    logger.info(f"  Predictor: {construct_def['predictor_name']}")
    logger.info(f"  Outcome:   {construct_def['outcome_name']}")
    logger.info(f"  Sending PDF to Claude: {os.path.basename(pdf_path)}")

    client = Anthropic(api_key=api_key)

    try:
        stats, screening = extract_stats_from_pdf(
            pdf_path, client, model, construct_def
        )
    except Exception as e:
        logger.error(f"  ERROR during extraction: {e}")
        logger.debug(traceback.format_exc())
        return None

    logger.info(f"  Found {len(stats)} statistic(s)")

    with open(os.path.join(save_dir, "variable_screening.json"), "w", encoding="utf-8") as f:
        json.dump(screening, f, indent=2, ensure_ascii=False)

    with open(os.path.join(save_dir, "extracted_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    quality = {}
    if not skip_quality_check:
        logger.info("  [Step 2.5] Running quality check...")
        with open(pdf_path, "rb") as _f:
            pdf_b64_qc = base64.b64encode(_f.read()).decode("ascii")
        try:
            quality = run_quality_check(
                pdf_b64_qc, client, model, construct_def, stats, screening
            )
        except Exception as e:
            logger.warning(f"  Quality check failed: {e}")
            quality = {}

        with open(os.path.join(save_dir, "quality_check.json"), "w", encoding="utf-8") as f:
            json.dump(quality, f, indent=2, ensure_ascii=False)

        scores = quality.get("scores", {})
        logger.info(
            f"    QC scores — completeness: {scores.get('completeness', '?')}/10  "
            f"accuracy: {scores.get('accuracy', '?')}/10  "
            f"reverse_coding: {scores.get('reverse_coding', '?')}/10"
        )
        original_stats = stats
        stats = apply_quality_corrections(stats, quality)
        n_added = len(stats) - len(original_stats)
        logger.info(
            f"  After QC: {len(stats)} statistic(s)"
            + (f" ({n_added:+d} from QC)" if n_added else "")
        )

        per_stat_acc = {
            (e.get("predictor_variable", "").strip().lower(),
             e.get("outcome_variable", "").strip().lower()): e.get("accuracy_score", 10)
            for e in quality.get("per_stat_accuracy", [])
        }
        for s in stats:
            key = (
                (s.get("predictor_variable") or "").strip().lower(),
                (s.get("outcome_variable") or "").strip().lower(),
            )
            s["accuracy_score"] = per_stat_acc.get(key, 10)

    logger.info("  [Step 3] Converting to Pearson r:")
    r_values = []
    r_values_qc_added = []
    r_conversions = []
    for i, stat in enumerate(stats):
        r, formula = convert_to_r(stat)
        pred = stat.get("predictor_variable", "?")
        outc = stat.get("outcome_variable", "?")
        stype = stat.get("statistic_type", "?")
        sval = stat.get("statistic_value", "?")
        qc_tag = " [QC]" if stat.get("qc_added") else ""
        acc_score = stat.get("accuracy_score", 10)
        acc_tag = f" [acc={acc_score}/10]" if not skip_quality_check else ""
        if r is not None:
            r_values.append(r)
            if stat.get("qc_added"):
                r_values_qc_added.append(r)
            r_conversions.append({
                "predictor_variable": pred,
                "outcome_variable": outc,
                "statistic_type": stype,
                "statistic_value": sval,
                "r": r,
                "conversion_formula": formula,
                "qc_added": bool(stat.get("qc_added")),
                "needs_human_review": bool(stat.get("needs_human_review")),
                "qc_warnings": stat.get("qc_warnings", []),
                "accuracy_score": acc_score,
            })
            logger.info(f"    [{i+1}] {stype}={sval} -> r={r:.4f}  ({pred} -> {outc}){qc_tag}{acc_tag}")
        else:
            logger.warning(
                f"    [{i+1}] {stype}={sval} -> SKIPPED (non-numeric after QC) "
                f"({pred} -> {outc}){qc_tag}{acc_tag}"
            )

    with open(os.path.join(save_dir, "r_values.json"), "w") as f:
        json.dump(r_values, f, indent=2)

    if not r_values:
        logger.info("  [Step 4] No valid r values found")
        aggregate_r = None
    else:
        aggregate_r = sum(r_values) / len(r_values)
        logger.info(f"  [Step 4] Aggregate r = {aggregate_r:.4f}  ({len(r_values)} effect(s))")
        logger.info(f"        Individual r values: {[round(v, 4) for v in r_values]}")
        if r_values_qc_added:
            r_original_only = [r for r, s in zip(r_values, stats) if not s.get("qc_added")]
            aggregate_r_original = sum(r_original_only) / len(r_original_only) if r_original_only else None
            logger.info(
                f"        Original-only aggregate r = "
                f"{f'{aggregate_r_original:.4f}' if aggregate_r_original is not None else 'N/A'}  "
                f"QC-added r values: {[round(v, 4) for v in r_values_qc_added]}"
            )

    r_original_only = [c["r"] for c in r_conversions if not c.get("qc_added")]
    aggregate_r_original = (
        sum(r_original_only) / len(r_original_only) if r_original_only else aggregate_r
    )

    result = {
        "study_id": study_id,
        "pair_id": construct_def["pair_id"],
        "pdf": os.path.basename(pdf_path),
        "aggregate_r": aggregate_r,
        "aggregate_r_original": aggregate_r_original,
        "n_effects": len(r_values),
        "r_values": r_values,
        "r_values_qc_added": r_values_qc_added,
        "r_conversions": r_conversions,
        "extracted_stats": stats,
        "quality_check_scores": quality.get("scores", {}),
        "per_stat_accuracy": quality.get("per_stat_accuracy", []),
    }
    with open(os.path.join(save_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return aggregate_r


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "Meta-analysis pipeline with construct-pair fallback. "
            "When a study yields no valid r values, uses the mean aggregate_r "
            "from studies in --fallback-dir that share the same construct pair."
        )
    )
    parser.add_argument(
        "--articles", default="test_articles.csv",
        help="CSV file listing articles and their construct pairs "
             "(default: test_articles.csv).",
    )
    parser.add_argument(
        "--constructs", default="test_construct_definitions.csv",
        help="CSV file with construct pair definitions "
             "(default: test_construct_definitions.csv).",
    )
    parser.add_argument(
        "--pdf-dir", default="test_pdfs",
        help="Folder containing PDF files named <studyid>.pdf "
             "(default: test_pdfs/).",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env variable).",
    )
    parser.add_argument(
        "--model", default="claude-opus-4-6",
        help="Claude model to use (default: claude-opus-4-6).",
    )
    parser.add_argument(
        "--outcsv", default="submission_fallback.csv",
        help="Output submission CSV path (default: submission_fallback.csv).",
    )
    parser.add_argument(
        "--outdir", default="output_test",
        help="Folder for per-study result files (default: output_test/).",
    )
    parser.add_argument(
        "--fallback-dir", default="output_test",
        help="Folder of previously-processed studies to draw fallback r values from "
             "(default: output_test/). Studies with the same construct pair and a "
             "non-null aggregate_r contribute to the fallback mean.",
    )
    parser.add_argument(
        "--study-id", nargs="+", default=None,
        help="Process only specific study IDs (e.g. --study-id study42 study43).",
    )
    parser.add_argument(
        "--skip-quality-check", action="store_true",
        help="Skip Step 2.5 quality check (saves one API call per study).",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: No API key. Set --api-key or ANTHROPIC_API_KEY env variable.")
        sys.exit(1)

    if not os.path.exists(args.constructs):
        print(f"ERROR: Construct definitions file not found: {args.constructs}")
        sys.exit(1)
    print(f"Loading construct definitions from: {args.constructs}")
    definitions = load_construct_definitions(args.constructs)

    if not os.path.exists(args.articles):
        print(f"ERROR: Articles file not found: {args.articles}")
        sys.exit(1)
    print(f"Loading articles from: {args.articles}")
    articles = load_articles(args.articles, pdf_dir=args.pdf_dir)

    # Build fallback index from previously-processed studies
    print(f"Loading fallback index from: {args.fallback_dir}")
    fallback_index = load_fallback_index(args.fallback_dir)

    study_filter = set(args.study_id) if args.study_id else None

    results = {}        # study_id -> final r (may be fallback)
    fallback_used = {}  # study_id -> fallback metadata
    skipped = []

    for article in articles:
        study_id = article["study_id"]

        if study_filter and study_id not in study_filter:
            continue

        c1 = article.get("construct1", "?")
        c2 = article.get("construct2", "?")
        print(f"\n{'─' * 60}")
        print(f"Processing: {study_id}  ({c1} → {c2})")

        construct_def = build_construct_def(article, definitions)
        if construct_def is None:
            print(f"  SKIPPED: could not resolve construct definitions")
            skipped.append(study_id)
            results[study_id] = None
            continue

        save_dir = os.path.join(args.outdir, study_id)
        try:
            r = process_one_study(article, construct_def, api_key, args.model, save_dir,
                                  skip_quality_check=args.skip_quality_check)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            r = None

        if r is None:
            # Try fallback: average aggregate_r from other studies with the same pair
            pair_id = construct_def["pair_id"]
            fb_r, fb_sources = compute_fallback_r(
                pair_id, fallback_index, exclude_study_id=study_id
            )
            if fb_r is not None:
                print(
                    f"  [FALLBACK] No r values found — using mean of "
                    f"{len(fb_sources)} matched study(ies) from {args.fallback_dir}: "
                    f"r = {fb_r:.4f}"
                )
                for src in fb_sources:
                    print(f"    • {src['study_id']}: aggregate_r = {src['aggregate_r']:.4f}")
                r = fb_r
                fallback_used[study_id] = {
                    "pair_id": pair_id,
                    "fallback_r": fb_r,
                    "fallback_sources": fb_sources,
                }

                # Append fallback info to the result.json already written
                result_path = os.path.join(save_dir, "result.json")
                if os.path.exists(result_path):
                    with open(result_path, encoding="utf-8") as _f:
                        result_data = json.load(_f)
                    result_data["fallback_used"] = True
                    result_data["fallback_r"] = fb_r
                    result_data["fallback_sources"] = fb_sources
                    with open(result_path, "w", encoding="utf-8") as _f:
                        json.dump(result_data, _f, indent=2)
            else:
                print(
                    f"  [FALLBACK] No r values found and no matching studies in "
                    f"{args.fallback_dir} for pair '{pair_id}' — leaving blank."
                )

        results[study_id] = r

    # Merge with any existing submission CSV so incremental runs accumulate
    if os.path.exists(args.outcsv):
        with open(args.outcsv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sid = row.get("studyid", "").strip()
                val = row.get("aggregateeffectsize", "").strip()
                if sid and sid not in results:
                    results[sid] = float(val) if val else None

    # Write submission CSV (all studies, sorted)
    with open(args.outcsv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["studyid", "aggregateeffectsize"])
        for study_id in sorted(
            results,
            key=lambda s: int(s.replace("study", "")) if s.replace("study", "").isdigit() else s,
        ):
            r = results[study_id]
            writer.writerow([study_id, "" if r is None else f"{r:.6f}"])

    print(f"\n{'═' * 60}")
    print(f"Results saved to {args.outcsv}")
    n_ok = sum(1 for r in results.values() if r is not None)
    n_fb = len(fallback_used)
    print(f"  Processed: {n_ok}/{len(results)} studies with effect sizes")
    if n_fb:
        print(f"  Fallback used for {n_fb} study(ies): {list(fallback_used)}")
    if skipped:
        print(f"  Skipped (missing construct definition): {skipped}")


if __name__ == "__main__":
    main()
