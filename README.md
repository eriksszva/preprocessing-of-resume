# Resume Screening Classifier — Preprocessing Module

This repository is part of a larger project titled **Resume Screening Classifier for Data Scientist Applications**, which aims to streamline and automate the process of identifying resumes that are relevant for data science roles using natural language processing (NLP).

This particular module focuses on the **preprocessing and exploratory data analysis (EDA)** phase, where raw resume data is cleaned, transformed, and prepared for downstream model training and evaluation.

## Dataset Overview

The dataset used in this project is sourced from [Kaggle: Resume Dataset](https://www.kaggle.com/datasets/saugataroyarghya/resume-dataset/data). It contains thousands of resumes represented in **semi-structured tabular format**, with columns covering various aspects of a candidate’s profile, such as:

* `career_objective`
* `skills`
* `major_field_of_studies`
* `positions`
* `responsibilities`

## Problem Statement

Manual resume screening is often:

* **Time-consuming**, especially with large applicant pools
* **Inconsistent**, due to subjective human evaluation
* **Prone to bias**, whether conscious or unconscious

To address these challenges, the goal of this module is to:

* Preprocess raw resume data into clean, usable format
* Combine multiple key columns into a unified textual representation (`resume_text`)
* Automatically **label each resume** as:

  * ✅ Relevant to be a Data Scientist
  * ❌ Not Relevant to be a Data Scientist
* Generate structured and semantically rich data to support **machine learning-based resume classification**

## 📁 Project Structure

```bash
.
├── .github/
│   └── workflows/
│       └── main.yaml                        # CI pipeline definition
├── raw_data/
│   └── resume_data.csv                      # Original raw dataset
├── preprocessing/
│   ├── cleaned_data/
│   │   └── resume_data_cleaned-labeled.csv  # Output: cleaned & labeled dataset
│   ├── utils/
│   │   └── ds_keywords.py                   # List of data science keywords for labeling
│   ├── automate_preprocessing.py            # Main script for preprocessing
│   └── Eksperimen_Erika.ipynb               # Notebook for exploration & testing
```

## Preprocessing Pipeline

The preprocessing pipeline is modular and robust, consisting of the following steps:

### 1. Feature Selection

Identify and isolate only the key resume components that matter most for relevance classification:

```python
['career_objective', 'skills', 'major_field_of_studies', 'positions', 'responsibilities']
```

These columns are selected for their high semantic value in representing a candidate's competency.

### 2. Handling Missing Values

Resume data often contains **missing (`NaN`) or placeholder values**, especially in optional fields. Instead of dropping such rows entirely, a **contextual strategy** is used:

* Rows with missing `skills`, `positions`, or `major_field_of_studies` are retained if other columns are rich in context.
* In some cases, missing `skills` are filled based on **grouped values from similar `positions`**, assuming common role-specific competencies.

### 3. Parsing List-Like Strings

Several columns use Python-style list strings such as:

```
"['Python', 'SQL']"
```

These are cleaned and converted into plain, comma-separated values:

```
"Python, SQL"
```

This helps flatten the structure for easier processing and modeling.

### 4. Text Cleaning

A custom `clean_text` function standardizes the text by:

* Lowercasing all entries
* Removing noise like `"N/A"`, `"null"`, `"None"`
* Replacing inconsistent separators (slashes, tabs, newlines)
* Stripping extra spaces and redundant punctuation

This ensures uniformity and improves the quality of textual representation for embeddings.

### 5. Renaming for Clarity

The column `positions` is renamed to `previous_positions` to avoid confusion with job postings or desired positions. This naming emphasizes that the field refers to **past roles held** by the candidate.

### 6. Unified Text Field: `resume_text`

All relevant resume fields are **merged into a single, structured text column**:

```
"Objective: ..., Skills: ..., Field of studies: ..., Previous positions: ..., Responsibilities: ..."
```

This approach allows for:

* Simple input into embedding models
* Retention of important semantic context
* Flexibility despite partial missing values in individual columns

### 7. Label Generation

Using a curated list of data science–related keywords (e.g., "machine learning", "data analysis", "python", "NLP"), each resume is labeled as:

* `1` → Relevant
* `0` → Not Relevant

This weak labeling approach enables supervised training for a binary classification model without needing full manual annotation.

### 8. Output

The cleaned and labeled dataset is exported to:

```
preprocessing/cleaned_data/resume_data_cleaned-labeled.csv
```

Example output row:

```csv
resume_text,label
"Objective: to work in machine learning, Skills: Python, SQL, Field of studies: CS, ..., Responsibilities: model deployment",1
```

## Sample Output Preview

| `resume_text` (truncated)                                                   | `label` |
| --------------------------------------------------------------------------- | ------- |
| Objective: big data analytics... Responsibilities: troubleshooting, support | 1       |
| Objective: fresher looking... Responsibilities: documentation, team tasks   | 0       |
| Objective: to enter AI field... Responsibilities: model development         | 1       |
| Objective: backend engineer... Responsibilities: debugging, system design   | 0       |

## CI/CD Integration

The module is integrated with **GitHub Actions** to automate data processing:

* Every push to the main branch will:

  * Trigger the preprocessing pipeline
  * Update the cleaned and labeled dataset automatically
* This ensures always up-to-date training data is available for downstream model development

## Requirements

To run the preprocessing module, install dependencies with:

```bash
pip install -r requirements.txt
```

Example dependencies include:

```txt
pandas
numpy
```

