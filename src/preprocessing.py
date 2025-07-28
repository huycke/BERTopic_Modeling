# -*- coding: utf-8 -*-
"""
This module contains the primary function for loading, cleaning, and preprocessing
text data for the BERTopic pipeline. It is designed to be configured and called
from a Jupyter notebook (e.g., 01_data_preprocessing.ipynb).
"""

import pandas as pd
import logging
import re
import html
from typing import Dict, Any, List, Optional

def _clean_text(text: str, config: Dict[str, Any]) -> str:
    """
    Applies a series of cleaning steps to a single text string based on a configuration dictionary.
    This is a helper function for the main load_and_preprocess_data function.

    Args:
        text: The raw text string to be cleaned.
        config: A dictionary containing boolean flags and parameters for each cleaning step.

    Returns:
        The cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    # --- Fix Text Encoding & Markdown ---
    # 1. Attempt to fix "double-encoded" UTF-8 characters (e.g., â€, â€™).
    # This happens when text saved as UTF-8 is incorrectly read as a single-byte encoding like latin1 or windows-1252.
    try:
        # This sequence re-encodes the wrongly decoded string back to its raw bytes, then correctly decodes as UTF-8.
        text = text.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        # This will fail if the string is already correct UTF-8 or another encoding, which is fine.
        pass

    # 2. Remove markdown-style links `[link text](URL)`, keeping only the link text.
    # This is configurable via the YAML file. Defaults to True if not specified.
    if config.get('clean_apply_markdown_removal', True):
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # --- Standard Cleaning Steps from Config ---
    # Each step is controlled by a boolean flag in the config dictionary.
    if config.get('clean_apply_unescape', False):
        text = html.unescape(text)

    if config.get('clean_apply_url_removal', False):
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

    if config.get('clean_apply_html_tag_removal', False):
        text = re.sub(r'<.*?>', '', text)

    if config.get('clean_apply_quote_normalization', False):
        # Normalizes multiple consecutive quotes into a single one.
        text = re.sub(r'"+', '"', text)

    if config.get('clean_apply_html_entity_removal', False):
        # Removes leftover HTML entities like &amp;
        text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)

    if config.get('clean_apply_char_filtering', False) and config.get('clean_char_filter_regex'):
        # Removes characters that don't match the provided regex.
        text = re.sub(config['clean_char_filter_regex'], ' ', text)

    if config.get('clean_apply_lowercase', False):
        text = text.lower()

    # Finally, collapse multiple whitespace characters into a single space and strip leading/trailing whitespace.
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_and_preprocess_data(
    file_path: str,
    text_source_columns: List[str],
    unique_id_column: str,
    required_columns_for_docs_creation: List[str],
    dropped_rows_output_path: str,
    data_type_specific_df_processing: Optional[Any] = None,
    **kwargs: Any
) -> Optional[pd.DataFrame]:
    """
    Loads data from a CSV, preprocesses the text, and applies filters based on parameters.

    This function orchestrates the entire preprocessing pipeline, from loading the data
    with the correct encoding to applying cleaning and filtering rules defined in the
    configuration dictionary (`kwargs`).

    Args:
        file_path: The absolute path to the raw data CSV file.
        text_source_columns: A list of column names to be combined into the main document text.
        unique_id_column: The name of the column that serves as a unique identifier for each row.
        required_columns_for_docs_creation: A list of columns that must not be empty. Rows with
                                            missing values in these columns will be dropped.
        dropped_rows_output_path: Path to save the log of rows that were dropped.
        data_type_specific_df_processing: (Optional) A function to perform dataset-specific
                                          transformations before the main processing.
        **kwargs: A dictionary containing all other configuration parameters for cleaning and filtering,
                  which will be passed to helper functions.

    Returns:
        A pandas DataFrame containing the processed and filtered data, or None if an error occurs.
    """
    print(f"Starting preprocessing for: {file_path}")

    # --- 1. Load Data with Correct Encoding ---
    try:
        # **CRUCIAL FIX**: Read the file explicitly with UTF-8 encoding to prevent character errors.
        # `encoding_errors='ignore'` will skip any characters that still cause issues.
        df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='ignore')
        print(f"Original dataset shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error reading CSV file {file_path}: {e}")
        raise

    # --- 2. Handle Required Columns & Create 'docs' Column ---
    print(f"Checking for missing values in required columns: {required_columns_for_docs_creation}")
    df_initial = df.copy()
    
    # Drop rows where any of the required columns have NaN values.
    df.dropna(subset=required_columns_for_docs_creation, inplace=True)
    
    # Also drop rows where the required text is just an empty string or whitespace.
    for col in required_columns_for_docs_creation:
        df = df[df[col].astype(str).str.strip() != '']

    # Log the rows that were dropped for transparency.
    dropped_rows = df_initial.loc[~df_initial.index.isin(df.index)]
    if not dropped_rows.empty:
        print(f"Dropped {len(dropped_rows)} rows due to missing/empty values in required columns: {required_columns_for_docs_creation}.")
        try:
            dropped_rows.to_csv(dropped_rows_output_path, index=False, encoding='utf-8')
            print(f"Saved {len(dropped_rows)} dropped rows to: {dropped_rows_output_path}")
        except Exception as e:
            logging.error(f"Failed to save dropped rows log: {e}")

    # Combine specified source columns into a single 'docs' column for modeling.
    df['docs'] = df[text_source_columns].astype(str).agg(' '.join, axis=1)
    print(f"Created 'docs' column from: {text_source_columns}")
    print(f"Using '{unique_id_column}' as reference ID.")

    # --- 3. Apply Configurable Text Cleaning ---
    print("Applying configurable text cleaning to 'docs' column...")
    # Pass the kwargs dictionary, which contains all cleaning parameters, to the helper function.
    df['docs'] = df['docs'].apply(lambda x: _clean_text(x, kwargs))
    
    # After cleaning, some 'docs' might become empty. Remove these.
    initial_rows = len(df)
    df.dropna(subset=['docs'], inplace=True)
    df = df[df['docs'].str.strip() != '']
    rows_after_cleaning = len(df)
    if initial_rows > rows_after_cleaning:
        print(f"Dropped {initial_rows - rows_after_cleaning} rows with empty 'docs' after text cleaning.")

    # --- 4. Apply Configurable Filters ---
    df_filtered = df.copy()
    
    # Length-based document filtering
    if kwargs.get('apply_length_filter'):
        min_len = kwargs.get('min_doc_length', 0)
        max_len = kwargs.get('max_doc_length', float('inf'))
        original_count = len(df_filtered)
        # Calculate length based on words (split by space).
        df_filtered['doc_len'] = df_filtered['docs'].str.split().str.len()
        df_filtered = df_filtered[df_filtered['doc_len'].between(min_len, max_len)]
        removed = original_count - len(df_filtered)
        if removed > 0:
            print(f"Filtered by length (words in 'docs', min: {min_len}, max: {max_len}): Removed {removed} documents. Kept {len(df_filtered)} documents.")

    # Duplicate document filtering
    if kwargs.get('apply_duplicate_removal'):
        col_to_check = kwargs.get('column_for_duplicate_checking', 'docs')
        original_count = len(df_filtered)
        df_filtered.drop_duplicates(subset=[col_to_check], keep='first', inplace=True)
        removed = original_count - len(df_filtered)
        if removed > 0:
            print(f"Removed duplicates based on '{col_to_check}': Removed {removed} documents. Kept {len(df_filtered)}.")

    # Score-based document filtering (e.g., for Reddit upvotes)
    if kwargs.get('apply_score_filter'):
        score_col = kwargs.get('score_column_for_filtering')
        min_score = kwargs.get('min_score_for_filtering')
        max_score = kwargs.get('max_score_for_filtering')
        if score_col and (min_score is not None or max_score is not None):
            original_count = len(df_filtered)
            query_parts = []
            if min_score is not None:
                query_parts.append(f"`{score_col}` >= {min_score}")
            if max_score is not None:
                query_parts.append(f"`{score_col}` <= {max_score}")
            
            if query_parts:
                df_filtered = df_filtered.query(" and ".join(query_parts))
                removed = original_count - len(df_filtered)
                if removed > 0:
                    print(f"Filtered by score (column: '{score_col}', range: {min_score}-{max_score}): Removed {removed} documents. Kept {len(df_filtered)}.")

    print(f"Finished preprocessing. Processed dataset shape: {df_filtered.shape}")
    return df_filtered