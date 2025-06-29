""" 
This script automates the preprocessing of resume data for a Data Scientist role.
It includes feature selection, handling missing values, parsing and cleaning text,
renaming columns, creating new features, labeling, and saving the cleaned data.
"""


import pandas as pd
import numpy as np
import ast  # parsing array-string
import re
from utils.ds_keywords import ds_keywords

def feature_selection(df):
    selected_fields = [
    'career_objective',
    'skills',
    'major_field_of_studies',
    'positions',
    'responsibilities'
    ]
    return df[selected_fields].copy()

def handling_missing_values(df):
    def handling_nan_skills(df):
        df_skills_nan = df[df['skills'].isna()].copy()
        df_skills_nan['career_objective'] = df_skills_nan['career_objective'].fillna("")
        df_skills_nan['skills'] = df_skills_nan['skills'].fillna("")

        group_cols = ['career_objective', 'skills', 'major_field_of_studies', 'positions']
        df_grouped = df_skills_nan.groupby(group_cols).agg({
            'responsibilities': lambda x: list(set(x.dropna()))
        }).reset_index()
        df_non_nan = df.dropna(subset=['skills'])
        df_cleaned = pd.concat([df_non_nan, df_grouped], ignore_index=True)
        return df_cleaned

    def handling_nan_positions(df):
        df_positions_nan = df[df['positions'].isna()].copy()
        df_positions_nan['positions'] = df_positions_nan['positions'].fillna("")

        group_cols = ['career_objective', 'skills', 'major_field_of_studies', 'positions']
        df_grouped = df_positions_nan.groupby(group_cols).agg({
            'responsibilities': lambda x: list(set(x.dropna()))
        }).reset_index()
        df_cleaned = pd.concat([df, df_grouped], ignore_index=True)
        df_cleaned = df_cleaned.dropna(subset=['positions'])
        return df_cleaned

    def handling_nan_major_field_of_studies(df):
        df_major_nan = df[df['major_field_of_studies'].isna()].copy()
        df_major_nan['career_objective'] = df_major_nan['career_objective'].fillna("")
        df_major_nan['major_field_of_studies'] = df_major_nan['major_field_of_studies'].fillna("")

        group_cols = ['career_objective', 'skills', 'major_field_of_studies', 'positions']
        df_grouped = df_major_nan.groupby(group_cols).agg({
            'responsibilities': lambda x: list(set(x.dropna()))
        }).reset_index()
        df_cleaned = pd.concat([df, df_grouped], ignore_index=True)
        df_cleaned = df_cleaned.dropna(subset=['major_field_of_studies'])
        return df_cleaned

    # cleaning order
    df = handling_nan_skills(df)
    df = handling_nan_positions(df)
    df = handling_nan_major_field_of_studies(df)
    return df.reset_index(drop=True)

def clean_text(text):
    if isinstance(text, list):
        # If it's a list, clean each item and join them
        cleaned_items = [clean_text(item) for item in text]
        return ", ".join(cleaned_items)
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    # clean nan or empty values
    if text in ["n/a", "na", "none", "-", "null"]:
        return ""
    text = str(text).replace("\n", ", ").replace("\r", ", ").replace("\t", " ")
    # change slashes to commas
    text = re.sub(r'\s*/\s*', ', ', text)
    # delete double punctuation
    text = re.sub(r'([,;:\.\-])\s*\1+', r'\1', text)  # change ,, ;; .. to be one
    # delete unusual combinations like ,., or .,, to be one comma/period only
    text = re.sub(r'([,;:\.])[\s]*([,;:\.])+', r'\1', text)
    # delete excessive spaces
    text = re.sub(r'\s+', ' ', text)
    # delete commas/periods at the beginning/end of the string
    text = text.strip(" ,.;:-")
    return text.strip()

def parse_list_column(col):
    # convert string list like "['A', 'B']" to actual python list
    def clean_list(x):
        try:
            if pd.isna(x) or x == "":
                return ""
            # attempt to parse the string as a list
            parsed = ast.literal_eval(x) if isinstance(x, str) else x
            if isinstance(parsed, list):
                # filter out None and empty strings before joining
                return ", ".join([
                    str(item).strip()
                    for item in parsed
                    if item is not None and str(item).strip() != ""
                ])
            else:
                return str(parsed).strip()
        except (ValueError, SyntaxError):
            # if literal_eval fails, treat it as a single string and clean it
            return clean_text(x)
        except Exception:
            return ""

    return col.apply(clean_list)

def rename_column(df):
    df.rename(columns={"positions": "previous_positions"}, inplace=True)
    return df

def new_feature(df):
    df["resume_text"] = df.apply(
    lambda x: ", ".join([
        f"Objective: {x['career_objective']}" if x["career_objective"] else "",
        f"Skills: {x['skills']}" if x["skills"] else "",
        f"Field of studies: {x['major_field_of_studies']}" if x["major_field_of_studies"] else "",
        f"Previous positions: {x['previous_positions']}" if x["previous_positions"] else "",
        f"Responsibilities: {x['responsibilities']}" if x["responsibilities"] else ""
    ]).strip(", "), axis=1
    )
    return df

def label(text):
    if not isinstance(text, str) or not text.strip():
        return 'Not Relevant to be a Data Scientist'
    text_lower = text.lower()
    match_count = sum(kw in text_lower for kw in ds_keywords)
    return 'Relevant to be a Data Scientist' if match_count >= 4 else 'Not Relevant to be a Data Scientist'

def encode_label(df):
    df['label'] = df['label'].map(lambda x: {'Relevant to be a Data Scientist': 1, 'Not Relevant to be a Data Scientist': 0}.get(x, x))
    return df

def save_cleaned_data(df, file_path='preprocessing/cleaned_data/resume_data_cleaned-labeled.csv'):
    df[["resume_text", "label"]].to_csv(file_path, index=False)
    print(f"Cleaned data saved to {file_path}")





if __name__ == '__main__':
    
    # --- Load Data ---
    file_path = 'raw_data/resume_data.csv'
    df = pd.read_csv(file_path)
    
    # --- Feature Selection & Handling Missing Values ---
    df = feature_selection(df)
    df_cleaned = handling_missing_values(df)
    
    # --- Parsing & Cleaning ---
    df_cleaned["skills"] = parse_list_column(df_cleaned["skills"])
    df_cleaned["major_field_of_studies"] = parse_list_column(df_cleaned["major_field_of_studies"])
    df_cleaned["positions"] = parse_list_column(df_cleaned["positions"])

    df_cleaned["career_objective"] = df_cleaned["career_objective"].apply(clean_text)
    df_cleaned["skills"] = df_cleaned["skills"].apply(clean_text)
    df_cleaned["major_field_of_studies"] = df_cleaned["major_field_of_studies"].apply(clean_text)
    df_cleaned["positions"] = df_cleaned["positions"].apply(clean_text)
    df_cleaned["responsibilities"] = df_cleaned["responsibilities"].apply(clean_text)
    
    # --- Rename Columns ---
    df_cleaned = rename_column(df_cleaned)
    
    # --- New Feature ---
    df_cleaned = new_feature(df_cleaned)
    
    # --- Labeling ---
    df_cleaned['label'] = df_cleaned['resume_text'].apply(label)
    
    # --- Encode Label ---
    df_cleaned = encode_label(df_cleaned)
    
    # --- Save the cleaned DataFrame ---
    save_cleaned_data(df_cleaned)