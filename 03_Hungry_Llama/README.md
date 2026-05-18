# Third Place: Hungry Llama

**Private Test Set Final MSE: 0.011416**

## Team Members

- Jennifer Gibson @ Fors Marsh
- Joe Luchman @ Fors Marsh
- Shane Halder @ Fors Marsh
- Ron Vega @ Fors Marsh

## Approach

- Google Colab Python Notebook
- Gemini API:
    - Structured Output feature
    - Gemini 3 Flash model
- Carefully crafted prompt template by SMEs
- Refined construct definitions by SMEs
- Prompt includes:
    - Piped in constructs and definitions
    - Valence detection for constructs
    - Reverse scoring detection for constructs
    - Effect size extraction rules
- Structured Output:
    - Study information (title, author, etc.)
    - List of effects for extraction
- Boolean logic to align the effect size direction

## Requirements

- Google Colab
- Pandas
- Gemini API (with API key)
- Pydantic
- Typing
- Numpy
- Competition test data files

## How to Run

- Upload the notebook to Google Colab
- Setup a **GEMINI_API_KEY** secret key in Colab
- Create a **test** folder and upload the following files to this folder:
    - test_articles.csv
    - test_construct_definitions8.csv
    - PDF research articles 
- Run the notebook in Colab
- Output files:
    - Intermediate output:
        - test_temp_processed.xlsx
        - test_temp_post_processed.xlsx
    - Submission file:
        - test_submit_YYYYMMDD_HHmmss.csv
