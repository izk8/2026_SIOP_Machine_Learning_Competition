# 2026 SIOP Machine Learning Competition

Data and Winning Code for the 2026 SIOP Machine Learning Competition

## Introduction

Can AI automate meta-analysis? Teams built end-to-end pipelines that take a research article as a PDF, along with descriptions of the research question, predictor, and dependent variable, and return a single aggregate effect size per study. When a study contained multiple relevant effect sizes, the true study-level value was the average of the relevant observed correlations. Submissions were a CSV with two columns: `studyid` and `aggregateeffectsize`. Scoring used Mean Squared Error (MSE) against held-out ground-truth values on a private test leaderboard.

## Task

Given a PDF article and a research question (with predictor and dependent variable descriptions), build a fully automated pipeline that:

1. Extracts the relevant predictor–outcome effect size(s) from the paper
2. Converts extracted effect sizes to Pearson's *r* if needed (and inverts if reverse-coded)
3. Averages them into a single aggregate effect size per study

The pipeline must be implemented in a single file, use freely shareable code, and rely only on publicly accessible models or APIs.

## Design Constraints

- Implemented in a single file (excluding imports and remote library installations)
- Uses freely shareable code with no license or copyright restrictions
- Uses models/APIs that are publicly available (paid/token-based APIs are allowed)
- Applies the same research question, predictor, and outcome descriptions for all papers in the same research question
- Custom pretrained models must be remotely installable (e.g., hosted on Hugging Face, CRAN, or GitHub)

## Submission Format

Upload a CSV with exactly two columns:

| Column | Description |
|--------|-------------|
| `studyid` | Unique study ID from the dataset |
| `aggregateeffectsize` | Predicted aggregate Pearson's *r* for that study |

Dev submissions: 5/day, 100 total. Test submissions: 3 total — submit carefully, no exceptions.

## Scoring

Pipelines are evaluated using Mean Squared Error between predicted and true aggregate effect sizes:

**MSE = (1/N) Σ (r̂ᵢ − rᵢ)²**

## Timeline

| Date | Event |
|------|-------|
| Mar 7 | Competition begins; dev dataset released |
| Apr 4 | Test dataset released |
| Apr 11 | Competition ends |
| Apr 12 | Winners notified |
| Apr 30 | Winning solutions presented at SIOP 2026 |

## Competition Portal

Full details on data, scoring, and submissions: https://computationaloutreach.com/siopmlcompetition2026

# Winners

---

[Competition Overview and Awards Presentation](https://github.com/izk8/2026_SIOP_Machine_Learning_Competition/blob/main/SIOP%202026%20ML%20Competition%20Deck.pdf)

## [First Place: goforit](01_goforit)

Nga Do @ University of Minnesota

Michael Hazboun @ University of Minnesota

Private Test Set Final MSE = **0.009233**

## [Second Place: Putka's Army](02_Putkas_Army)

Ammar Ansari @ HumRRO

Daniel Barstow @ HumRRO

Anoop Javalagi @ HumRRO

Jiayi Liu @ HumRRO

Karla Castillo-Guerra @ HumRRO

John Little @ HumRRO

Robert Wellman @ HumRRO

Lilang Chen @ HumRRO

Private Test Set Final MSE = **0.010497**

## [Third Place: Hungry Llama](03_Hungry_Llama)

Jennifer Gibson @ Fors Marsh

Joe Luchman @ Fors Marsh

Shane Halder @ Fors Marsh

Ron Vega @ Fors Marsh

Private Test Set Final MSE = **0.011416**

## [Fourth Place: MetaML](04_MetaML)

Pengda Wang @ Rice University

Private Test Set Final MSE = **0.012888**

# Organizers

---

Ivan Hernandez @ Virginia Tech

Isaac Thompson @ Amazon

Egyn Zhu @ Amazon

# How to Cite

Hernandez, I., Thompson, I., & Zhu, E. *The 2026 SIOP Machine Learning Competition.* Presented at the 41st Annual Society for Industrial and Organizational Psychology Conference in New Orleans, LA.

# License

This repository is licensed under [CC-BY-SA-4.0](LICENSE). The research articles referenced in the dataset are **not included** — participants must obtain them through their own institutional access, and those articles remain under the copyright of their original publishers.
