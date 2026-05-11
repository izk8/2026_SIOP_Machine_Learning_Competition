"""
Final .py file to run it all

Ammar Ansari 
"""

import csv
import io
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from io import StringIO
from pathlib import Path
import pathlib
from typing import List, Literal, Optional
import numpy as np
import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ============================================
# FIX WINDOWS CONSOLE ENCODING
# ============================================
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass  # older Python versions may not support reconfigure

# ============================================
# SCHEMA
# ============================================

class StatisticEntry(BaseModel):
    predictor_variable: str = Field(description="Name of the predictor/independent variable")
    criterion_variable: str = Field(description="Name of the criterion/dependent/outcome variable")
    predictor_construct: Optional[str] = Field(default=None, description="Which construct from system instructions does the predictor map to?")
    criterion_construct: Optional[str] = Field(default=None, description="Which construct from system instructions does the criterion map to?")
    statistic_type: Literal[
        "pearsons_r", "cohens_d", "hedges_g", "glass_delta", "point_biserial",
        "standardized_beta", "unstandardized_beta", "t_statistic", "f_statistic",
        "chi_square", "odds_ratio", "eta_squared", "partial_eta_squared",
        "r_squared", "mean_and_sd", "contingency_table_2x2", "other"
    ] = Field(description="Type of statistic reported")
    statistic_value: Optional[float] = Field(default=None, description="The reported value of the statistic")
    regression_type: Optional[Literal["simple", "multiple"]] = Field(default=None, description="If beta: simple (1 predictor) or multiple regression?")
    number_of_predictors: Optional[int] = Field(default=None, description="If multiple regression: how many predictors?")
    sample_size_total: Optional[int] = Field(default=None, description="Total N for this specific statistic")
    sample_size_group1: Optional[int] = Field(default=None, description="N for group 1")
    sample_size_group2: Optional[int] = Field(default=None, description="N for group 2")
    mean_group1: Optional[float] = Field(default=None, description="Mean of group 1 on criterion")
    mean_group2: Optional[float] = Field(default=None, description="Mean of group 2 on criterion")
    sd_group1: Optional[float] = Field(default=None, description="SD of group 1 on criterion")
    sd_group2: Optional[float] = Field(default=None, description="SD of group 2 on criterion")
    mean_predictor: Optional[float] = Field(default=None, description="Mean of predictor variable")
    sd_predictor: Optional[float] = Field(default=None, description="SD of predictor variable")
    mean_criterion: Optional[float] = Field(default=None, description="Mean of criterion variable")
    sd_criterion: Optional[float] = Field(default=None, description="SD of criterion variable")
    cell_a: Optional[int] = Field(default=None, description="2x2 table cell A (top-left)")
    cell_b: Optional[int] = Field(default=None, description="2x2 table cell B (top-right)")
    cell_c: Optional[int] = Field(default=None, description="2x2 table cell C (bottom-left)")
    cell_d: Optional[int] = Field(default=None, description="2x2 table cell D (bottom-right)")
    pooled_sd: Optional[float] = Field(default=None, description="Pooled SD if reported")
    se: Optional[float] = Field(default=None, description="Standard error of the statistic")
    confidence_interval_lower: Optional[float] = Field(default=None, description="Lower 95% CI")
    confidence_interval_upper: Optional[float] = Field(default=None, description="Upper 95% CI")
    p_value: Optional[float] = Field(default=None, description="P-value")
    degrees_of_freedom: Optional[float] = Field(default=None, description="df (for t-test, chi-square)")
    degrees_of_freedom_numerator: Optional[float] = Field(default=None, description="Numerator df (for F-statistic)")
    degrees_of_freedom_denominator: Optional[float] = Field(default=None, description="Denominator df (for F-statistic)")
    reliability_predictor: Optional[float] = Field(default=None, description="Cronbach's alpha of predictor")
    reliability_criterion: Optional[float] = Field(default=None, description="Cronbach's alpha of criterion")
    is_reverse_coded: Optional[bool] = Field(default=None, description="True if relationship direction needs flipping based on construct definitions")
    reverse_code_reason: Optional[str] = Field(default=None, description="Why this needs reverse coding, if applicable")
    page_or_table: Optional[str] = Field(default=None, description="Page, table, or figure where found")
    notes: Optional[str] = Field(default=None, description="Additional context for conversion")

class SubStudy(BaseModel):
    sub_study_label: Optional[str] = Field(default=None, description="Label if paper has multiple studies (e.g., 'Study 1', 'Sample A'). null if only one study.")
    sample_size: Optional[int] = Field(default=None, description="N for this sub-study/sample")
    statistics: List[StatisticEntry] = Field(description="All relevant predictor-criterion statistics from this sub-study")

class PaperResponse(BaseModel):
    study_id: str = Field(description="Paper identifier matching filename, e.g., 'study1'")
    paper_title: Optional[str] = Field(default=None, description="Title of the paper")
    number_of_sub_studies: int = Field(description="How many separate studies/samples are in this paper")
    sub_studies: List[SubStudy] = Field(description="Each sub-study/sample within the paper")

class AllPapersResponse(BaseModel):
    papers: List[PaperResponse]


###### FUNCTIONS #####

def safe_print(msg):
    """Print that won't crash on Windows with non-ASCII characters."""
    try:
        print(msg)
    except UnicodeEncodeError:
        # Replace problematic characters and retry
        print(msg.encode("ascii", errors="replace").decode("ascii"))


def safe_open(filepath, mode="r", **kwargs):
    """Open a file with UTF-8 encoding by default."""
    kwargs.setdefault("encoding", "utf-8")
    return open(filepath, mode, **kwargs)


def remove_nulls(obj):
    if isinstance(obj, dict):
        return {k: remove_nulls(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [remove_nulls(item) for item in obj]
    return obj


def build_system_instruction(construct_lookup, construct1_name, construct2_name):
    c1_def = construct_lookup[construct1_name]
    c2_def = construct_lookup[construct2_name]

    return f"""You are a meta-analysis data extraction assistant. You extract effect sizes from empirical papers examining the relationship between {construct1_name} (predictor/Construct 1) and {construct2_name} (criterion/Construct 2).

=== CONSTRUCT 1 (PREDICTOR): {construct1_name.upper()} ===
{c1_def}

=== CONSTRUCT 2 (CRITERION): {construct2_name.upper()} ===
{c2_def}

=== REVERSE CODING RULES ===
Set is_reverse_coded=true when:
- The PREDICTOR measures the conceptual opposite of {construct1_name} -- flip sign so higher = more {construct1_name}
- The CRITERION measures the conceptual opposite of {construct2_name} -- flip sign so higher = more {construct2_name}
- Both predictor and criterion are reverse-coded -- the two flips cancel out, so is_reverse_coded=false
Explain your reasoning in reverse_code_reason.

=== STATISTIC TYPES TO EXTRACT ===
Extract ALL of these when found:
- Pearson's r
- Cohen's d or Hedges' g or Glass' delta -- MUST include sample sizes of both groups
- Point biserial -- MUST include sample sizes of both groups
- Standardized beta -- MUST indicate simple vs multiple regression. If available, include SD of predictor and criterion.
- Unstandardized beta -- I NEED SD of predictor and criterion
- t-statistic with df or N
- F-statistic with numerator and denominator df
- Chi-square with total N
- Odds ratio
- 2x2 contingency table cell values
- Eta-squared or partial eta-squared
- R-squared from simple regression
- Group means and SDs -- ONLY when two groups are compared on the CRITERION variable.

=== CRITICAL INSTRUCTIONS ===
1. Extract EVERY {construct1_name}--{construct2_name} relationship, not just one per paper.
2. Note if a construct is REVERSE CODED (set is_reverse_coded=true and explain why).
3. For betas, ALWAYS specify simple vs multiple regression.
4. For unstandardized betas, extract mean and SD of predictor and criterion.
5. For GROUP COMPARISONS (mean_and_sd type): mean_group1/sd_group1 and mean_group2/sd_group2 are the TWO GROUPS measured on the CRITERION variable.
6. Sample size may differ across sub-studies and analyses -- capture each.
7. If a correlation matrix is reported, extract each relevant r as a separate entry.
8. For F-statistics, capture BOTH numerator and denominator df.
9. For chi-square, capture total N.
10. Use null for any field not reported.
11. A single paper may contain MULTIPLE sub-studies with DIFFERENT samples. List each separately.
12. All coded effects must be at the SAME temporal level (Time 1 X -> Time 1 Y, never Time 1 X -> Time 2 Y)."""


def build_prompt_test(c1, c2, research_question):
    return (

        f"RESEARCH QUESTION: {research_question}\n\n"

        f"Extract every {c1} -> {c2} statistic from each paper following the rules "
        f"in your system instructions."
    )

# --- Build prompts per construct pair ---
def build_prompt_for_pair_agent(construct1_name, construct2_name, papers_for_pair, construct_definitions, research_question):
    c1_def = construct_definitions[construct1_name]
    c2_def = construct_definitions[construct2_name]

    papers_json = papers_for_pair #json.dumps(papers_for_pair)

    prompt = f"""
You are an expert psychometrician working on a meta-analytic coding task.

## Research Question

{research_question}

## Construct 1: {construct1_name}

{c1_def}

## Construct 2: {construct2_name}

{c2_def}

## Task

This meta-analytic task focuses on automating the article coding process. Given the research question above, your pipeline must:

1. For the study in the JSON below, identify all statistics that capture the bivariate relationship between Construct 1 ({construct1_name}) and Construct 2 ({construct2_name}).

2. Prefer Pearson's r correlations when available. Only use other statistics (betas, t-statistics, F-statistics, etc.) to derive an approximate Pearson's r if no direct correlation is reported for that specific predictor-criterion pair.

3. Handle reverse coding: if a variable is flagged as `is_reverse_coded: true`, flip the sign of the effect size so it aligns with the direction that higher {construct1_name} -> higher {construct2_name} (or whatever the natural direction is per the construct definitions).

4. For each study, aggregate across all relevant Pearson's r values (or converted-to-r values) by:
   a. Fisher's z-transforming each r: z = arctanh(r)- (use code execution for the math)
   b. Averaging the Fisher's z values - (use code execution for the math)
   c. Inverse Fisher's z-transforming back to get the aggregate r: aggregate_r = tanh(mean_z) - (use code execution for the math)

5. If a study has multiple sub-studies, first aggregate within each sub-study, then aggregate across sub-studies.

6. If a statistic is available as both a Pearson's r AND a regression beta for the same predictor-criterion pair, use ONLY the Pearson's r.

7. If no relevant statistics exist for a study, return null.

Return the aggregate Pearson's R.

## Study

{papers_json}
"""
    return prompt

def produce_json_and_single_prediction(
        this_studyid,
        pdf_filepath,
        construct1,
        construct2,
        research_question,
        output_directory,
        study_constructs,
        construct_lookup,
        client,
        MODEL_ID
):
    all_papers = []
    failed_batches = []

    MAX_RETRIES = 5
    INITIAL_RETRY_DELAY = 30  # seconds

    # Build system instruction for THIS construct pair
    sys_instruction = build_system_instruction(construct_lookup, construct1, construct2)

    contents = []
    contents.append(
        types.Part.from_bytes(
            data=pdf_filepath.read_bytes(),
            mime_type="application/pdf",
        ),
    )
    contents.append(build_prompt_test(construct1, construct2, research_question=research_question))

    success = False
    for attempt in range(MAX_RETRIES):
        try:
            batch_response = client.models.generate_content(
                model=MODEL_ID,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=sys_instruction,
                    thinking_config=types.ThinkingConfig(thinking_level="medium"),
                    response_mime_type="application/json",
                    response_json_schema=AllPapersResponse.model_json_schema(),
                ),
            )

            raw_text = batch_response.text
            clean_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip())
            batch_results = AllPapersResponse.model_validate_json(clean_text)

            # Fix study IDs
            for idx, paper in enumerate(batch_results.papers):
                paper.study_id = f"{this_studyid}"

            all_papers.extend(batch_results.papers)

            # Save individual JSONs
            for paper in batch_results.papers:
                with safe_open(f"{output_directory}/{paper.study_id}.json", "w") as f:
                    json.dump(paper.model_dump(), f, indent=2)

            safe_print(f"  [OK] Parsed {len(batch_results.papers)} papers (total: {len(all_papers)})")
            success = True
            break  # Exit retry loop on success

        except Exception as e:
            error_str = str(e)
            if "503" in error_str and attempt < MAX_RETRIES - 1:
                retry_delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                safe_print(f"  [WARN] 503 error on attempt {attempt+1}/{MAX_RETRIES}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                safe_print(f"  [FAIL] Failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                if attempt == MAX_RETRIES - 1 or "503" not in error_str:
                    break

    if success:
        with safe_open(f"{output_directory}/{this_studyid}.json", "r") as f:
            this_paper = json.load(f)

        dumped_paper = json.dumps(this_paper)

        agent_prompt = build_prompt_for_pair_agent(
            construct1_name=construct1,
            construct2_name=construct2,
            papers_for_pair=dumped_paper,
            construct_definitions=construct_lookup,
            research_question=research_question
        )

        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=agent_prompt,
                config=types.GenerateContentConfig(
                    system_instruction="Return only a single numerical value representing the aggregate Pearson's R, or null if no relevant statistics exist. Use code execution tools for all math operations.",
                    thinking_config=types.ThinkingConfig(thinking_level="high"),
                    tools=[types.Tool(code_execution=types.ToolCodeExecution)],
                    response_mime_type="text/plain",
                ),
            )

            return response

        except Exception as e:
            return e

    else:
        return Exception
    

def remove_fields(data):
    """Remove paper_title and page_or_table fields from the JSON structure."""
    for paper in data.get("papers", []):
        paper.pop("paper_title", None)
        for sub_study in paper.get("sub_studies", []):
            for statistic in sub_study.get("statistics", []):
                statistic.pop("page_or_table", None)
    return data


def run_DR_agent_loop(
        MODEL_ID,
        STUDY_DIRECTORY_PATH,
        OUTPUT_DIRECTORY_PATH,
        GEMINI_API_KEY,
        TEST_ARTICLES_CSV_PATH,
        CONSTRUCT_DEFINTIONS_PATH,
        DEEP_RESEARCH_RESEARCH_QUESTION_FOR_BATCH,
        autofill_bivariate_relationship_for_research_question    
):
    ### Read in provided inputs: 
    articles_df = pd.read_csv(TEST_ARTICLES_CSV_PATH, encoding='latin-1')
    construct_defs = pd.read_csv(CONSTRUCT_DEFINTIONS_PATH)

    if 'research_question' not in articles_df.columns and DEEP_RESEARCH_RESEARCH_QUESTION_FOR_BATCH == "What is the bivariate association between [Construct1] and [Construct2]?":
        articles_df['research_question'] = "What is the bivariate association between " + articles_df['Construct1'] + " and " + articles_df['Construct2'] + "?"
    
    elif autofill_bivariate_relationship_for_research_question:
        articles_df['research_question'] = "What is the bivariate association between " + articles_df['Construct1'] + " and " + articles_df['Construct2'] + "?"

    else:
        raise KeyError('research_question column not in articles csv file. Add this to your csv input file or re-run with "autofill_bivariate_relationship_for_research_question" parameter set to True.')
    
    if 'study_filename' not in articles_df.columns:
        articles_df['study_filename'] = articles_df['studyid'] + ".pdf"


    ### Build a lookup: construct name to definition text
    construct_lookup = dict(zip(construct_defs["Construct"], construct_defs["Definition"]))

    ### For each study, get its construct pair and research question:
    study_constructs = {}
    for _, row in articles_df.iterrows():
        study_constructs[row["studyid"]] = {
            "construct1": row["Construct1"],
            "construct2": row["Construct2"],
            "research_question": row['research_question'],
            "study_filename": row['study_filename']
        }

    ###
    study_id_list = []
    study_val_list = []
    study_debug_list = []

    ### call Gemini Client:
    client = genai.Client(api_key=GEMINI_API_KEY)

    for this_studyid in articles_df["studyid"]:

        this_study_construct1 = study_constructs[f"{this_studyid}"]['construct1']
        this_study_construct2 = study_constructs[f"{this_studyid}"]['construct2']
        this_research_question = study_constructs[f"{this_studyid}"]['research_question']
        this_study_filename = study_constructs[f"{this_studyid}"]['study_filename']

        this_study_path = os.path.join(
            STUDY_DIRECTORY_PATH,
            this_study_filename
        )

        try:
            ret_val = produce_json_and_single_prediction(
                this_studyid = this_studyid,
                pdf_filepath = pathlib.Path(this_study_path),
                construct1 = this_study_construct1,
                construct2 = this_study_construct2,
                research_question = this_research_question,
                output_directory = OUTPUT_DIRECTORY_PATH,
                study_constructs = study_constructs,
                construct_lookup = construct_lookup,
                client = client,
                MODEL_ID = MODEL_ID
            )

            if ret_val is not None and hasattr(ret_val, 'text'):
                ret_val_text = ret_val.text
            else:
                ret_val_text = 'null'

        except Exception as e:
            safe_print(f"Study {this_studyid} failed with exception : {e}")
            ret_val = "null"
            ret_val_text = "null"

        study_id_list.append(f"{this_studyid}")
        study_val_list.append(ret_val_text)
        study_debug_list.append(ret_val)

    results_df = pd.DataFrame({
        "studyid": study_id_list,
        "aggregateeffectsize": study_val_list
    })

    results_df.to_csv(f"{OUTPUT_DIRECTORY_PATH}/non_deep_research_results.csv", index=False)

    ### After the loop, save debug info as JSON using str() conversion
    debug_output = []
    for study_id, response in zip(study_id_list, study_debug_list):
        debug_output.append({
            "study_id": study_id,
            "raw_response": str(response)
        })

    with safe_open(f"{OUTPUT_DIRECTORY_PATH}/debug_responses.json", "w") as f:
        json.dump(debug_output, f, indent=2, ensure_ascii=True)


    all_papers = []
    for this_studyid in articles_df['studyid']:
        path = f"{OUTPUT_DIRECTORY_PATH}/{this_studyid}.json"
        if os.path.exists(path):
            with safe_open(path, "r") as f:
                paper = PaperResponse.model_validate(json.load(f))
                all_papers.append(paper)

    safe_print(f"Loaded {len(all_papers)} papers from individual files")

    with safe_open(f"{OUTPUT_DIRECTORY_PATH}/all_papers_raw_test_agent_2.json", "w") as f:
        json.dump(AllPapersResponse(papers=all_papers).model_dump(), f, indent=2, ensure_ascii=True)
    safe_print(f"Saved all_papers_raw_test_agent_2.json")

    with safe_open(f"{OUTPUT_DIRECTORY_PATH}/all_papers_raw_test_agent_2.json", "r") as f:
        test_agent_2_data = json.load(f)
        test_agent_2_papers = AllPapersResponse.model_validate(test_agent_2_data)
        all_papers = list(test_agent_2_papers.papers)

    # Read, clean, write
    with safe_open(f"{OUTPUT_DIRECTORY_PATH}/all_papers_raw_test_agent_2.json") as f:
        data = json.load(f)

    cleaned = remove_nulls(data)

    with safe_open(f"{OUTPUT_DIRECTORY_PATH}/all_papers_raw_test_agent_2_no_nulls.json", "w") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=True)

    all_papers_raw_string = json.dumps(cleaned)

    # Remove the fields
    cleaned_and_no_paper_title_or_page_number = remove_fields(cleaned)

    with safe_open(f"{OUTPUT_DIRECTORY_PATH}/all_papers_raw_test_agent_2_no_nulls_cleaned.json", "w") as f:
        json.dump(cleaned_and_no_paper_title_or_page_number, f, indent=2, ensure_ascii=True)

    safe_print("Done! Removed 'paper_title' and 'page_or_table' fields.")

    all_papers_raw_string = json.dumps(cleaned_and_no_paper_title_or_page_number)

    # ============================================================
    # 1. LOAD INPUT FILES
    # ============================================================

    with safe_open(f"{OUTPUT_DIRECTORY_PATH}/all_papers_raw_test_agent_2_no_nulls_cleaned.json", "r") as f:
        all_papers_data = json.load(f)

    all_papers_raw_string = json.dumps(all_papers_data)

    # ============================================================
    # 2. BUILD LOOKUPS
    # ============================================================

    construct_lookup = dict(zip(construct_defs["Construct"], construct_defs["Definition"]))

    study_constructs = {}
    for _, row in articles_df.iterrows():
        study_constructs[row["studyid"]] = {
            "construct1": row["Construct1"],
            "construct2": row["Construct2"],
        }

    # Group studies by construct pair
    pair_groups = defaultdict(list)
    for study_id, info in study_constructs.items():
        pair_key = (info["construct1"], info["construct2"])
        pair_groups[pair_key].append(study_id)

    safe_print(f"Found {len(pair_groups)} unique construct pairs:")
    for pair, studies in sorted(pair_groups.items(), key=lambda x: -len(x[1])):
        # Use ASCII arrow to avoid encoding issues
        safe_print(f"  {pair[0]} -> {pair[1]}: {len(studies)} studies")

    # ============================================================
    # 3. BUILD CONSTRUCT DEFINITIONS BLOCK
    # ============================================================

    def build_construct_definitions_block(construct_lookup, pair_groups):
        """Build a formatted text block of all relevant construct definitions."""
        used_constructs = set()
        for (c1, c2) in pair_groups.keys():
            used_constructs.add(c1)
            used_constructs.add(c2)

        lines = []
        for construct_name in sorted(used_constructs):
            if construct_name in construct_lookup:
                lines.append(f"### {construct_name}\n")
                lines.append(f"{construct_lookup[construct_name]}\n")
            else:
                lines.append(f"### {construct_name}\n")
                lines.append(f"(No definition found)\n")

        return "\n".join(lines)

    construct_defs_text = build_construct_definitions_block(construct_lookup, pair_groups)
    safe_print(f"\nConstruct definitions block: {len(construct_defs_text):,} characters")
    safe_print(f"Constructs included: {sum(1 for c in set(c for pair in pair_groups for c in pair) if c in construct_lookup)}")

    # ============================================================
    # 4. BUILD THE MEGA-PROMPT
    # ============================================================

    def build_single_call_prompt(construct_defs_text, all_papers_raw_string, study_constructs_map, research_question_prototype):
        """Build the full prompt for a single deep research call."""

        # Use ASCII arrow to avoid encoding issues in prompt
        study_pair_table = "\n".join([
            f"- {sid}: {info['construct1']} -> {info['construct2']}"
            for sid, info in sorted(study_constructs_map.items(), key=lambda x: int(x[0].replace('study', '')))
        ])

        prompt = f"""
        You are an expert psychometrician automating a meta-analytic coding task across multiple construct pairs.

        ## OVERVIEW

        Below you will find:
        1. QUICK REFERENCE -- which construct pair each study examines
        2. CONSTRUCT DEFINITIONS -- definitions for every construct in this dataset
        3. STUDY-LEVEL DATA -- a JSON with extracted statistics for {len(study_constructs_map)} studies
        4. TASK INSTRUCTIONS -- how to process each study

        ## 1. QUICK REFERENCE: STUDY -> CONSTRUCT PAIRS

        {study_pair_table}

        ## 2. CONSTRUCT DEFINITIONS

        The following definitions specify what scales and measures qualify for each construct. Use these to determine which statistics in each study are relevant to its specific research question.

        {construct_defs_text}

        ## 3. TASK INSTRUCTIONS

        For each study in the JSON:

        a) Look up the study in the QUICK REFERENCE table above to find its target construct pair. The research question for that study is always: {research_question_prototype}

        b) Identify ALL statistics where `predictor_construct` and `criterion_construct` match the target construct pair from the quick reference (in either direction). Use the construct definitions above to confirm that the measured variables genuinely fall under those constructs.

        c) Among matching statistics, USE these priority rules:
        - PREFER `pearsons_r` over any other statistic type for the same predictor-criterion variable pair.
        - Only use regression betas, t-statistics, F-statistics, or other statistics to derive an approximate Pearson's r if NO direct Pearson's r correlation is available for that specific variable pair.
        - Do NOT double-count: if you have r AND beta for the same variable pair, use only r.

        d) Handle reverse coding:
        - If `is_reverse_coded` is true, the reported value has ALREADY been conceptually noted as needing a flip. Apply the flip (multiply by -1) so the effect aligns with: higher Construct1 <-> higher Construct2 in the direction the constructs are written.
        - Read the `reverse_code_reason` field for guidance on directionality.

        e) Aggregate using Fisher's z transformation:
        1. Convert each relevant Pearson's r to Fisher's z: z = arctanh(r)
        2. Average all Fisher's z values for that study
        3. Convert back: aggregate_r = tanh(mean_z)
        4. Round to 2 decimal places.

        f) If a study has multiple sub-studies, collect all relevant r values across ALL sub-studies, then do a single Fisher's z aggregation across all of them.

        g) If a study has zero relevant Pearson's r values (and no convertible statistics), return null.

        h) If a statistic has a value that is implausible for its type (e.g., a standardized beta with absolute value > 1.0), exclude it from the aggregation.

        i) If a pooled/overall sample statistic is available alongside sub-sample breakdowns from the same data, prefer the sub-sample statistics to avoid double-counting -- UNLESS the sub-samples are non-overlapping subgroups (e.g., male vs. female), in which case use the sub-sample statistics. If in doubt, use the pooled statistic only.

        ## 4. STUDY DATA (JSON)

        Each study object contains:
        - `study_id`: the identifier
        - `sub_studies` -> `statistics`: the extracted data points
        - Each statistic has `predictor_construct` and `criterion_construct` fields indicating which constructs it measures

        Match each statistic's `predictor_construct` and `criterion_construct` against the target construct pair from the QUICK REFERENCE table.

        ## 5. REQUIRED OUTPUT

        Use code execution to perform the Fisher's z transformations and aggregations programmatically. Do NOT do the math by hand.

        For each study, briefly note which r values you selected and why, then compute the aggregate.

        For each of the studies in the following json, give me the aggregate pearson r.

        {all_papers_raw_string}

        """
        return prompt
    
    deep_research_context = build_single_call_prompt(construct_defs_text, all_papers_raw_string, study_constructs, research_question_prototype=DEEP_RESEARCH_RESEARCH_QUESTION_FOR_BATCH)

    with safe_open(f"{OUTPUT_DIRECTORY_PATH}/deep_research_context_preview.txt", "w") as f:
        f.write(deep_research_context)

    dr_test = client.interactions.create(
        agent="deep-research-pro-preview-12-2025",
        background=True,
        input=(deep_research_context),
        tools=[
            {"type": "google_search"},
            {"type": "code_execution"}
        ],
    )

    while True:
        status_check = client.interactions.get(dr_test.id)
        safe_print(f"  Status: {status_check.status}")

        if status_check.status == "completed":
            safe_print(status_check.outputs[-1].text)
            with safe_open(f"{OUTPUT_DIRECTORY_PATH}/deep_research_filesearch_output.txt", "w") as f:
                f.write(status_check.outputs[-1].text)
            break
        elif status_check.status in ["failed", "cancelled"]:
            safe_print(f"  [FAIL] {status_check.status}")
            break
        time.sleep(30)

    deep_research_response = status_check.outputs[-1].text

    final_prompt = f"""
    Here is a study report I need to pull the aggregate effect sizes from: {deep_research_response}

    Convert the aggregate effect size by study information in the report to csv with the following format:

    studyid,aggregateeffectsize
    study1,0.23
    study2,-0.11
    study3,0.00
    """

    MAX_RETRIES = 5
    INITIAL_RETRY_DELAY = 30  # seconds

    success = False
    response = None
    for attempt in range(MAX_RETRIES):
        try:

            response = client.models.generate_content(
                model=MODEL_ID,
                contents=final_prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are a data extraction assistant. Return only raw CSV text -- no markdown, no code fences, no explanation.",
                    thinking_config=types.ThinkingConfig(thinking_level="high"),
                    response_mime_type="text/plain",
                ),
            )

            success = True
            break  # Exit retry loop on success

        except Exception as e:
            error_str = str(e)
            if "503" in error_str and attempt < MAX_RETRIES - 1:
                retry_delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                safe_print(f"  [WARN] 503 error on attempt {attempt+1}/{MAX_RETRIES}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                safe_print(f"  [FAIL] Failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                if attempt == MAX_RETRIES - 1 or "503" not in error_str:
                    break

    if not success or response is None:
        safe_print("[FAIL] Could not get final CSV response from model after all retries.")
        # Return a fallback dataframe with nulls
        fallback_df = pd.DataFrame({
            "studyid": list(study_constructs.keys()),
            "aggregateeffectsize": [0.0] * len(study_constructs)
        })
        fallback_df.to_csv(f"{OUTPUT_DIRECTORY_PATH}/final_deep_research_submission_test_agent_2_weighted_imputed.csv", index=False)
        return fallback_df

    raw = response.text.strip()

    # 1. Strip markdown code-fence markers
    cleaned = raw.strip().removeprefix("```csv").removesuffix("```").strip()

    # 2. Read into DataFrame
    df = pd.read_csv(StringIO(cleaned))

    # 3. If columns aren't named correctly, fix them
    expected_cols = ["studyid", "aggregateeffectsize"]

    if list(df.columns) != expected_cols:
        df = pd.read_csv(StringIO(cleaned), header=None, names=expected_cols)

    # 4. Fill NaN with 0
    df.fillna(0, inplace=True)

    # 5. Save to CSV
    df.to_csv(f"{OUTPUT_DIRECTORY_PATH}/final_deep_research_submission_test_agent_2_na_to_0.csv", index=False)

    safe_print(f"Saved {len(df)} rows.")
    safe_print(str(df.head(10)))

    # ============================================================
    # Parse raw CSV again, but do NOT fill NaN yet (for weighted imputation)
    # ============================================================
    cleaned = raw.strip().removeprefix("```csv").removesuffix("```").strip()
    df = pd.read_csv(StringIO(cleaned))

    expected_cols = ["studyid", "aggregateeffectsize"]
    if list(df.columns) != expected_cols:
        df = pd.read_csv(StringIO(cleaned), header=None, names=expected_cols)

    # Convert "null" strings to actual NaN
    df["aggregateeffectsize"] = pd.to_numeric(df["aggregateeffectsize"], errors="coerce")

    safe_print(f"Total rows: {len(df)}")
    safe_print(f"Rows with NaN: {df['aggregateeffectsize'].isna().sum()}")
    safe_print(f"NaN study IDs: {df.loc[df['aggregateeffectsize'].isna(), 'studyid'].tolist()}")

    # ============================================================
    # Load the extracted JSON to get sample sizes per study
    # ============================================================
    with safe_open(f"{OUTPUT_DIRECTORY_PATH}/all_papers_raw_test_agent_2_no_nulls_cleaned.json", "r") as f:
        all_papers_data = json.load(f)

    study_sample_sizes = {}
    if isinstance(all_papers_data, dict) and "papers" in all_papers_data:
        papers_list = all_papers_data["papers"]
    elif isinstance(all_papers_data, list):
        papers_list = all_papers_data
    else:
        papers_list = all_papers_data.get("papers", [])

    for paper in papers_list:
        sid = paper["study_id"]
        total_n = 0
        for sub in paper.get("sub_studies", []):
            n = sub.get("sample_size")
            if n is None or n == 0:
                stat_ns = [
                    s.get("sample_size_total", 0)
                    for s in sub.get("statistics", [])
                    if s.get("sample_size_total") is not None
                ]
                n = max(stat_ns) if stat_ns else 0
            total_n += (n or 0)
        study_sample_sizes[sid] = total_n

    safe_print(f"\nSample sizes loaded for {len(study_sample_sizes)} studies")

    # ============================================================
    # Add construct pair info & sample size to the dataframe
    # ============================================================
    df["construct1"] = df["studyid"].map(
        lambda sid: study_constructs.get(sid, {}).get("construct1", None)
    )
    df["construct2"] = df["studyid"].map(
        lambda sid: study_constructs.get(sid, {}).get("construct2", None)
    )
    # Use ASCII arrow to avoid encoding issues
    df["construct_pair"] = df["construct1"] + " -> " + df["construct2"]
    df["sample_size"] = df["studyid"].map(study_sample_sizes)

    safe_print("\nConstruct pairs for NaN rows:")
    safe_print(str(df.loc[df["aggregateeffectsize"].isna(), ["studyid", "construct_pair", "sample_size"]]))

    # ============================================================
    # Compute sample-size-weighted mean effect size per construct pair
    # ============================================================
    valid = df.dropna(subset=["aggregateeffectsize"]).copy()

    def weighted_mean_by_pair(group):
        r = group["aggregateeffectsize"]
        n = group["sample_size"]
        mask = r.notna() & n.notna() & (n > 0)
        if mask.sum() == 0:
            return np.nan
        return np.average(r[mask], weights=n[mask])

    pair_weighted_means = valid.groupby("construct_pair").apply(weighted_mean_by_pair)
    pair_weighted_means.name = "weighted_mean_r"

    safe_print("\n=== Sample-size-weighted mean r by construct pair ===")
    for pair, wmean in pair_weighted_means.items():
        count = valid.loc[valid["construct_pair"] == pair].shape[0]
        safe_print(f"  {pair}: weighted mean r = {wmean:.4f}  (k={count} studies)")

    # ============================================================
    # Impute NaN rows with the weighted mean of their construct pair
    # ============================================================
    nan_mask = df["aggregateeffectsize"].isna()

    for idx in df.index[nan_mask]:
        pair = df.loc[idx, "construct_pair"]
        imputed_value = pair_weighted_means.get(pair, 0.0)
        df.loc[idx, "aggregateeffectsize"] = round(imputed_value, 2)
        safe_print(f"\n  Imputed {df.loc[idx, 'studyid']} ({pair}): {imputed_value:.4f} -> rounded to {round(imputed_value, 2)}")

    # ============================================================
    # Verify no NaN remains, then save
    # ============================================================
    assert df["aggregateeffectsize"].isna().sum() == 0, "Still have NaN values!"

    output_df = df[["studyid", "aggregateeffectsize"]].copy()

    output_df.to_csv(
        f"{OUTPUT_DIRECTORY_PATH}/final_deep_research_submission_test_agent_2_weighted_imputed.csv",
        index=False,
    )

    safe_print(f"\nSaved {len(output_df)} rows.")
    safe_print(str(output_df.head(10)))

    return output_df

if __name__ == '__main__':

    ##### SET UP YOUR CONSTANTS 

    MODEL_ID = "gemini-3.1-flash-lite-preview" #"gemini-3-flash-preview" #"gemini-3.1-pro-preview" #"gemini-3-flash-preview" # TODO update
    STUDY_DIRECTORY_PATH = "../input_data/TEST_SET_3_ATTEMPTS/test_articles" # TODO update
    OUTPUT_DIRECTORY_PATH = "../output_data/TEST_SET_3_ATTEMPTS" # TODO update

    GEMINI_API_KEY =os.getenv("JOHN_GEMINI_API_KEY") # TODO update

    TEST_ARTICLES_CSV_PATH = "../input_data/TEST_SET_3_ATTEMPTS/test_articles.csv" # TODO update
    CONSTRUCT_DEFINTIONS_PATH = "../input_data/TEST_SET_3_ATTEMPTS/test_construct_definitions.csv" # TODO update

    DEEP_RESEARCH_RESEARCH_QUESTION_FOR_BATCH = "What is the bivariate association between [Construct1] and [Construct2]?" # TODO update 

    #### TODO Add validation

    try:
        ret_df = run_DR_agent_loop(
            MODEL_ID = MODEL_ID,
            STUDY_DIRECTORY_PATH = STUDY_DIRECTORY_PATH,
            OUTPUT_DIRECTORY_PATH = OUTPUT_DIRECTORY_PATH,
            GEMINI_API_KEY = GEMINI_API_KEY,
            TEST_ARTICLES_CSV_PATH = TEST_ARTICLES_CSV_PATH,
            CONSTRUCT_DEFINTIONS_PATH = CONSTRUCT_DEFINTIONS_PATH,
            DEEP_RESEARCH_RESEARCH_QUESTION_FOR_BATCH = DEEP_RESEARCH_RESEARCH_QUESTION_FOR_BATCH,
            autofill_bivariate_relationship_for_research_question = False
        )

        safe_print(f"ret_df:")
        safe_print(f"{ret_df.head()}")

    except Exception as e:
        safe_print(f"{e}")
