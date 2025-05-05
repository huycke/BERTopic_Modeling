# -*- coding: utf-8 -*-
"""
Preprocessing module for the BERTopic pipeline.

Handles loading, cleaning, filtering, and unitizing text data from various sources.
Includes detailed logging of removed rows during filtering.
"""

import os
import re
import html # For HTML entity unescaping
import logging
import pandas as pd
import numpy as np
# --- ADDED IMPORT FOR TYPE HINTING ---
from typing import Any, Dict, List, Optional, Union, Tuple
# --- END ADDED IMPORT ---


# NLTK is used for sentence tokenization, stop words, lemmatization
# Users might need to download NLTK data packs the first time:
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    nltk_punkt_available = True
except LookupError:
    logging.warning("NLTK 'punkt' tokenizer data not found. Sentence splitting will fail. Run nltk.download('punkt')")
    nltk_punkt_available = False
try:
    # Ensure other necessary NLTK data is checked/downloaded if used
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError as e:
    logging.warning(f"Missing NLTK data ({e}). Some cleaning steps might fail. Run nltk.download() for the missing packages.")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer


# Import utility functions
try:
    from src import utils # Assumes src is in the python path or running from root
except ImportError:
    import utils # Fallback for direct execution or different environment setup


# --- Constants ---
# Define standard internal column names
DEFAULT_OUTPUT_ID_COL = 'doc_id'
DEFAULT_OUTPUT_TEXT_COL = 'text' # Column name after combining inputs
DEFAULT_OUTPUT_UNIT_ID_COL = 'unit_id' # Unique ID for each text unit (doc, para, sent)
DEFAULT_OUTPUT_TEXT_UNIT_COL = 'text_unit' # Column containing the final text unit
REMOVAL_LOG_SUFFIX = "_removal_log.csv" # Suffix for the removal log file


# --- Main Preprocessing Function ---

def load_and_preprocess_data(
    file_path: str,
    output_path: str,
    config_path: Optional[str] = None,
    force_recompute: bool = False,
    **kwargs: Any
) -> pd.DataFrame:
    """
    Loads data, applies cleaning, filtering (with logging), and unitization,
    then saves the result and a removal log.

    Args:
        file_path: Path to the raw data file (CSV, JSON, JSONL, TXT, XLSX).
        output_path: Path to save the processed DataFrame (CSV).
        config_path: (Optional) Path to YAML config file specifying parameters.
        force_recompute: If True, ignore existing output file and recompute.
        **kwargs: Parameters passed directly, overriding config file values.

    Returns:
        A pandas DataFrame containing the processed data.

    Raises:
        FileNotFoundError: If input file not found.
        ValueError: For invalid configurations or unsupported file types.
        Exception: For other processing errors.
    """
    logging.info(f"Starting preprocessing for: {file_path}")

    # 1. Load Configuration (Merge file and kwargs)
    config = {}
    if config_path:
        try:
            config = utils.load_config(config_path)
            logging.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logging.error(f"Failed to load config from {config_path}: {e}")
            raise
    config.update(kwargs)
    logging.debug(f"Using effective configuration: {config}")

    # Set force_recompute based on config or argument
    force_recompute = config.get('force_recompute', force_recompute)

    # Define removal log path
    output_dir, output_filename = os.path.split(output_path)
    output_basename, _ = os.path.splitext(output_filename)
    removal_log_path = os.path.join(output_dir, output_basename + REMOVAL_LOG_SUFFIX)
    save_removal_log = config.get('save_removal_log', True) # Option to disable saving log

    # 2. Check Cache (Check both processed file and removal log if enabled)
    processed_exists = utils.check_cache(output_path)
    log_exists = not save_removal_log or utils.check_cache(removal_log_path) # Log exists if not saving or file present
    if not force_recompute and processed_exists and log_exists:
        logging.info(f"Processed file and removal log exist: {output_path}, {removal_log_path}. Loading cached version.")
        try:
            processed_df = pd.read_csv(output_path)
            # Basic validation of cached file
            id_col = config.get('column_mapping', {}).get('output_id_col', DEFAULT_OUTPUT_ID_COL)
            required_cached_cols = [id_col, DEFAULT_OUTPUT_UNIT_ID_COL, DEFAULT_OUTPUT_TEXT_UNIT_COL]
            if not all(col in processed_df.columns for col in required_cached_cols):
                 raise ValueError("Cached file missing required columns")
            logging.info(f"Successfully loaded cached processed data from {output_path}")
            return processed_df
        except Exception as e:
            logging.warning(f"Could not load or validate cached file {output_path}: {e}. Recomputing.")


    # --- Start Processing ---
    removal_log_data = [] # Initialize list to store removal details

    # 3. Load Raw Data
    logging.info(f"Loading raw data from: {file_path}")
    try:
        raw_df = _load_raw_data(file_path, config)
    except Exception as e:
        logging.error(f"Failed to load raw data from {file_path}: {e}")
        raise

    # 4. Standardize Columns
    logging.info("Standardizing column names...")
    try:
        standardized_df = _standardize_columns(raw_df, config)
    except Exception as e:
        logging.error(f"Failed to standardize columns: {e}")
        raise

    # 5. Combine Text Columns
    logging.info("Combining text columns...")
    try:
        combined_df = _combine_text_columns(standardized_df, config)
    except Exception as e:
        logging.error(f"Failed to combine text columns: {e}")
        raise

    # 6. Handle Missing Essential Data
    logging.info("Handling missing essential data...")
    processed_df, removed_missing_essential = _handle_missing_essential(combined_df, config)
    if not removed_missing_essential.empty:
        removal_log_data.append(removed_missing_essential)
    if processed_df.empty:
         logging.warning("DataFrame is empty after handling missing essential data. No output generated.")
         if save_removal_log: _save_log(removal_log_data, removal_log_path)
         open(output_path, 'a').close() # Create empty file for cache
         return processed_df

    # 7. Apply Text Cleaning
    logging.info("Applying text cleaning...")
    text_col_to_clean = config.get('column_mapping', {}).get('output_text_col', DEFAULT_OUTPUT_TEXT_COL)
    if text_col_to_clean not in processed_df.columns:
         raise ValueError(f"Text column '{text_col_to_clean}' not found for cleaning.")
    processed_df = _clean_text(processed_df, text_col_to_clean, config)

    # 8. Unitize Text
    logging.info("Unitizing text based on granularity...")
    processed_df = _split_text(processed_df, text_col_to_clean, config)
    if processed_df.empty:
        logging.warning("DataFrame is empty after text unitization. No output generated.")
        if save_removal_log: _save_log(removal_log_data, removal_log_path)
        open(output_path, 'a').close() # Create empty file for cache
        return processed_df

    # 9. Apply Filtering (Now returns filtered_df and removal details)
    logging.info("Applying filtering...")
    processed_df, removed_filtered = _filter_data(processed_df, config)
    if not removed_filtered.empty:
        removal_log_data.append(removed_filtered)
    if processed_df.empty:
        logging.warning("DataFrame is empty after filtering. No output generated.")
        if save_removal_log: _save_log(removal_log_data, removal_log_path)
        open(output_path, 'a').close() # Create empty file for cache
        return processed_df

    # --- End Processing Steps ---

    # 10. Save Processed Data and Removal Log
    logging.info(f"Saving processed data to: {output_path}")
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        # Define expected columns for final output
        id_col = config.get('column_mapping', {}).get('output_id_col', DEFAULT_OUTPUT_ID_COL)
        metadata_cols = config.get('metadata_cols', [])
        final_expected_cols = [id_col, DEFAULT_OUTPUT_UNIT_ID_COL, DEFAULT_OUTPUT_TEXT_UNIT_COL] + \
                              [col for col in metadata_cols if col in processed_df.columns]
        # Save only existing columns in the expected order
        cols_to_save = [col for col in final_expected_cols if col in processed_df.columns]
        processed_df[cols_to_save].to_csv(output_path, index=False)
        logging.info(f"Successfully saved processed data ({len(processed_df)} rows).")

        # Save the removal log
        if save_removal_log:
            _save_log(removal_log_data, removal_log_path)

    except Exception as e:
        logging.error(f"Failed to save processed data or removal log: {e}")
        raise

    return processed_df


# --- Helper Functions ---

def _save_log(log_data_list: List[pd.DataFrame], log_path: str) -> None:
    """Combines and saves the removal log data."""
    if not log_data_list:
        logging.info("No rows removed during processing, removal log not saved.")
        return
    try:
        full_log_df = pd.concat(log_data_list, ignore_index=True)
        log_dir = os.path.dirname(log_path)
        if log_dir: os.makedirs(log_dir, exist_ok=True)
        full_log_df.to_csv(log_path, index=False)
        logging.info(f"Successfully saved removal log ({len(full_log_df)} rows) to: {log_path}")
    except Exception as e:
        logging.error(f"Failed to save removal log to {log_path}: {e}")


def _load_raw_data(file_path: str, config: Dict[str, Any]) -> pd.DataFrame:
    """Loads data from various file formats."""
    # (Implementation from v1 - unchanged)
    file_format = config.get('file_format', None)
    if not file_format: _, ext = os.path.splitext(file_path); file_format = ext.lower().strip('.')
    logging.debug(f"Attempting to load file as format: {file_format}")
    if file_format == 'csv': return pd.read_csv(file_path, low_memory=False)
    elif file_format in ['json', 'jsonl']:
        try: return pd.read_json(file_path, lines=True)
        except ValueError: logging.warning("Failed JSONL read, trying standard JSON."); return pd.read_json(file_path)
    elif file_format == 'xlsx': return pd.read_excel(file_path)
    elif file_format == 'txt':
        logging.warning("Loading .txt: one doc per line, using index as ID.");
        with open(file_path, 'r', encoding='utf-8') as f: lines = [ln.strip() for ln in f if ln.strip()]
        df = pd.DataFrame(lines, columns=['text']); df[DEFAULT_OUTPUT_ID_COL] = df.index; return df
    else: raise ValueError(f"Unsupported file format: {file_format}")

def _standardize_columns(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Renames input columns to standard internal names based on config."""
    # (Implementation from v1 - unchanged)
    mapping = config.get('column_mapping', {}); output_id_col = mapping.get('output_id_col', DEFAULT_OUTPUT_ID_COL)
    if not mapping: logging.warning("No 'column_mapping' provided."); return df
    input_id_col = mapping.get('input_id_col'); input_text_cols = mapping.get('input_text_cols', [])
    output_text_col = mapping.get('output_text_col', DEFAULT_OUTPUT_TEXT_COL)
    if input_id_col:
        if input_id_col not in df.columns: raise ValueError(f"Input ID column '{input_id_col}' not found.")
        if input_id_col != output_id_col: df = df.rename(columns={input_id_col: output_id_col}); logging.debug(f"Renamed ID '{input_id_col}'->'{output_id_col}'")
    elif output_id_col not in df.columns: logging.warning(f"No input ID col & '{output_id_col}' not found. Using index."); df[output_id_col] = df.index
    if not isinstance(input_text_cols, list): raise ValueError("'input_text_cols' must be a list.")
    missing_text_cols = [c for c in input_text_cols if c not in df.columns]; valid_input_text_cols = [c for c in input_text_cols if c in df.columns]
    if missing_text_cols: logging.warning(f"Input text columns not found: {missing_text_cols}. Ignored.")
    if not valid_input_text_cols and output_text_col not in df.columns: raise ValueError(f"No valid text columns found or specified.")
    metadata_cols_to_keep = config.get('metadata_cols', [])
    if not isinstance(metadata_cols_to_keep, list): raise ValueError("'metadata_cols' must be a list.")
    cols_to_keep_set = {output_id_col} | set(valid_input_text_cols) | set(metadata_cols_to_keep)
    final_columns = [c for c in df.columns if c in cols_to_keep_set or c == output_id_col]
    missing_metadata = [c for c in metadata_cols_to_keep if c not in df.columns]
    if missing_metadata: logging.warning(f"Metadata columns not found: {missing_metadata}. Ignored.")
    df_standardized = df[final_columns].copy(); logging.debug(f"Cols kept: {df_standardized.columns.tolist()}")
    return df_standardized

def _combine_text_columns(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Combines multiple text columns into a single standard text column."""
    # (Implementation from v1 - unchanged)
    mapping = config.get('column_mapping', {}); output_text_col = mapping.get('output_text_col', DEFAULT_OUTPUT_TEXT_COL)
    input_text_cols = mapping.get('input_text_cols', []); valid_input_text_cols = [c for c in input_text_cols if c in df.columns]
    if not valid_input_text_cols:
        if output_text_col in df.columns: logging.info(f"Using existing column '{output_text_col}'."); df[output_text_col] = df[output_text_col].astype(str).fillna(''); return df
        else: raise ValueError("No valid text columns found or specified.")
    elif len(valid_input_text_cols) == 1:
        input_col = valid_input_text_cols[0]
        if input_col != output_text_col: df = df.rename(columns={input_col: output_text_col}); logging.debug(f"Renamed text col '{input_col}'->'{output_text_col}'")
        df[output_text_col] = df[output_text_col].astype(str).fillna('')
    else:
        logging.info(f"Combining text from: {valid_input_text_cols} into '{output_text_col}'")
        df[output_text_col] = df[valid_input_text_cols].astype(str).fillna('').agg(' '.join, axis=1)
    return df

def _handle_missing_essential(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Handles rows with missing essential data (ID or Text).

    Returns the filtered DataFrame and a DataFrame containing removed rows with reasons.
    """
    # (Implementation from v5 - unchanged, uses Tuple correctly now)
    mapping = config.get('column_mapping', {});
    id_col = mapping.get('output_id_col', DEFAULT_OUTPUT_ID_COL)
    text_col = mapping.get('output_text_col', DEFAULT_OUTPUT_TEXT_COL)
    skip_missing = config.get('skip_missing_essential', True)
    removed_rows_log = pd.DataFrame() # Initialize empty log

    missing_id_mask = df[id_col].isnull()
    missing_text_mask = df[text_col].isnull() | (df[text_col].astype(str).str.strip() == '')
    rows_to_drop_mask = missing_id_mask | missing_text_mask

    if rows_to_drop_mask.any():
        num_missing = rows_to_drop_mask.sum()
        if skip_missing:
            removed_df = df[rows_to_drop_mask].copy()
            removed_df['removal_reason'] = np.where(missing_id_mask[rows_to_drop_mask], 'missing_id', 'missing_or_empty_text')
            # Select only ID columns for the log
            id_cols_for_log = [id_col, DEFAULT_OUTPUT_UNIT_ID_COL] # unit_id might not exist yet
            id_cols_present = [c for c in id_cols_for_log if c in removed_df.columns]
            removed_rows_log = removed_df[id_cols_present + ['removal_reason']]

            df = df[~rows_to_drop_mask].copy()
            logging.warning(f"Removed {num_missing} rows due to missing ID or empty text.")
        else:
             logging.error(f"Found {num_missing} rows with missing ID/text, but 'skip_missing_essential' is False.")

    if len(df) == 0 and num_missing > 0:
        logging.warning("All rows were removed due to missing essential data.")

    return df, removed_rows_log


def _clean_text(df: pd.DataFrame, text_col: str, config: Dict[str, Any]) -> pd.DataFrame:
    """Applies various text cleaning steps based on the configuration."""
    # (Implementation from v2 - unchanged)
    opts = config.get('cleaning_options', {}); logging.debug(f"Applying cleaning options: {opts}")
    df[text_col] = df[text_col].astype(str)
    if opts.get('html_unescape', True): logging.debug("HTML unescaping..."); df[text_col] = df[text_col].apply(html.unescape)
    if opts.get('remove_urls', True): logging.debug("Removing URLs..."); url_pattern = r'https?://\S+|www\.\S+'; df[text_col] = df[text_col].str.replace(url_pattern, '', regex=True)
    if opts.get('remove_emails', True): logging.debug("Removing emails..."); email_pattern = r'\S+@\S+\.\S+'; df[text_col] = df[text_col].str.replace(email_pattern, '', regex=True)
    if opts.get('lowercase', True): logging.debug("Lowercasing..."); df[text_col] = df[text_col].str.lower()
    case_sensitive = not opts.get('lowercase', True); boilerplate_patterns = opts.get('boilerplate_remove', [r'\[deleted\]', r'\[removed\]'])
    if boilerplate_patterns:
        logging.debug(f"Removing boilerplate: {boilerplate_patterns}")
        for pattern in boilerplate_patterns:
            try: df[text_col] = df[text_col].str.replace(pattern, '', regex=True, case=case_sensitive)
            except re.error as e: logging.warning(f"Invalid regex '{pattern}': {e}. Skipping.")
    custom_regex_list = opts.get('custom_regex_remove', [])
    if custom_regex_list:
        logging.debug(f"Applying custom regex: {custom_regex_list}")
        for pattern in custom_regex_list:
            try: df[text_col] = df[text_col].str.replace(pattern, '', regex=True, case=case_sensitive)
            except re.error as e: logging.warning(f"Invalid regex '{pattern}': {e}. Skipping.")
    if opts.get('remove_punctuation', False): logging.debug("Removing punctuation..."); punct_pattern = r'[^\w\s]'; df[text_col] = df[text_col].str.replace(punct_pattern, '', regex=True)
    logging.debug("Removing extra whitespace..."); df[text_col] = df[text_col].str.replace(r'\s+', ' ', regex=True).str.strip()
    if opts.get('remove_stop_words', False):
        language = opts.get('language', 'english'); logging.debug(f"Removing stop words: {language}")
        try:
            stop_words_set = set(stopwords.words(language)); custom_stops = opts.get('custom_stop_words', []); stop_words_set.update(custom_stops)
            def _remove_stops(text): words = word_tokenize(text); filtered = [w for w in words if w.lower() not in stop_words_set]; return ' '.join(filtered)
            df[text_col] = df[text_col].apply(_remove_stops)
        except OSError: logging.error(f"NLTK stopwords '{language}' not found. Skipping."); pass
        except Exception as e: logging.error(f"Stop word removal error: {e}. Skipping."); pass
    if opts.get('lemmatize', False):
        logging.debug("Applying lemmatization...")
        try:
            lemmatizer = WordNetLemmatizer()
            def _lemmatize_text(text): words = word_tokenize(text); lemmatized = [lemmatizer.lemmatize(w) for w in words]; return ' '.join(lemmatized)
            df[text_col] = df[text_col].apply(_lemmatize_text)
        except LookupError: logging.error("NLTK WordNet data not found. Skipping lemmatization."); pass
        except Exception as e: logging.error(f"Lemmatization error: {e}. Skipping."); pass
    if opts.get('stem', False) and not opts.get('lemmatize', False): logging.warning("Stemming not implemented yet.")
    df[text_col] = df[text_col].str.strip(); logging.info("Text cleaning applied.")
    return df

def _split_text(df: pd.DataFrame, text_col: str, config: Dict[str, Any]) -> pd.DataFrame:
    """Splits documents into paragraphs or sentences based on granularity config."""
    # (Implementation from v3 - unchanged)
    granularity = config.get('granularity', 'document')
    id_col = config.get('column_mapping', {}).get('output_id_col', DEFAULT_OUTPUT_ID_COL)
    metadata_cols = [col for col in df.columns if col not in [text_col, id_col]]
    logging.info(f"Applying granularity: {granularity}")
    if granularity == 'document':
        df = df.rename(columns={text_col: DEFAULT_OUTPUT_TEXT_UNIT_COL})
        df[DEFAULT_OUTPUT_UNIT_ID_COL] = df[id_col]
        logging.debug("Granularity 'document': Renamed text column and copied doc_id.")
        # Ensure column order consistency
        cols_ordered = [id_col, DEFAULT_OUTPUT_UNIT_ID_COL, DEFAULT_OUTPUT_TEXT_UNIT_COL] + metadata_cols
        return df[[col for col in cols_ordered if col in df.columns]] # Keep only existing columns

    new_rows = []
    for index, row in df.iterrows():
        doc_id = row[id_col]; original_text = row[text_col]; metadata = row[metadata_cols].to_dict()
        if pd.isna(original_text) or not isinstance(original_text, str) or not original_text.strip():
            logging.debug(f"Skipping row {index} (doc_id {doc_id}) due to empty/invalid text."); continue
        units = []
        if granularity == 'paragraph':
            pattern = config.get('paragraph_split_pattern', r'\n\s*\n')
            try: units = [p.strip() for p in re.split(pattern, original_text) if p.strip()]; logging.debug(f"Doc {doc_id}: Split {len(units)} paragraphs.")
            except re.error as e: logging.error(f"Invalid regex '{pattern}': {e}. Skipping split for doc {doc_id}."); units = [original_text.strip()]
            except Exception as e: logging.error(f"Error splitting paragraphs for doc {doc_id}: {e}. Skipping."); units = [original_text.strip()]
        elif granularity == 'sentence':
            if not nltk_punkt_available: logging.error("NLTK 'punkt' missing. Treating as single unit."); units = [original_text.strip()]
            else:
                try: units = [s.strip() for s in sent_tokenize(original_text) if s.strip()]; logging.debug(f"Doc {doc_id}: Split {len(units)} sentences.")
                except Exception as e: logging.error(f"Error tokenizing sentences for doc {doc_id}: {e}. Skipping."); units = [original_text.strip()]
        else: logging.warning(f"Unsupported granularity '{granularity}'. Treating as 'document'."); units = [original_text.strip()]
        for i, unit_text in enumerate(units):
            unit_id = f"{doc_id}_{granularity[0]}{i}"; new_row_data = {id_col: doc_id, DEFAULT_OUTPUT_UNIT_ID_COL: unit_id, DEFAULT_OUTPUT_TEXT_UNIT_COL: unit_text, **metadata}; new_rows.append(new_row_data)
    if not new_rows: logging.warning("No text units generated after splitting."); final_cols = [id_col, DEFAULT_OUTPUT_UNIT_ID_COL, DEFAULT_OUTPUT_TEXT_UNIT_COL] + metadata_cols; return pd.DataFrame(columns=final_cols)
    split_df = pd.DataFrame(new_rows); logging.info(f"Expanded DataFrame from {len(df)} to {len(split_df)} rows after splitting.")
    final_cols = [id_col, DEFAULT_OUTPUT_UNIT_ID_COL, DEFAULT_OUTPUT_TEXT_UNIT_COL] + metadata_cols
    split_df = split_df[[col for col in final_cols if col in split_df.columns]] # Ensure order and existence
    return split_df


# --- Filtering Implementation (with Logging) ---

def _filter_data(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies filtering based on text length, metadata, and duplicates.

    Returns the filtered DataFrame and a DataFrame logging removed rows.
    """
    # (Implementation from v5 - unchanged, uses Tuple correctly now)
    opts = config.get('filtering_options', {})
    if not opts:
        logging.info("No filtering options specified. Skipping filtering.")
        return df, pd.DataFrame() # Return empty log

    logging.debug(f"Applying filtering options: {opts}")
    initial_rows = len(df)
    df_to_filter = df.copy() # Work on a copy to identify removed rows
    removal_details = [] # List to store dicts {unit_id: ..., reason: ...}

    text_unit_col = DEFAULT_OUTPUT_TEXT_UNIT_COL
    unit_id_col = DEFAULT_OUTPUT_UNIT_ID_COL
    doc_id_col = config.get('column_mapping', {}).get('output_id_col', DEFAULT_OUTPUT_ID_COL)

    if text_unit_col not in df_to_filter.columns or unit_id_col not in df_to_filter.columns:
        logging.error(f"Required columns '{text_unit_col}' or '{unit_id_col}' not found for filtering.")
        raise ValueError(f"Required columns for filtering not found.")

    # --- Helper to log removals ---
    def log_removal(ids_to_remove: pd.Index, reason: str):
        count = len(ids_to_remove)
        if count > 0:
            logging.info(f"Identified {count} rows to remove based on: {reason}")
            # Store unit_id and doc_id if available
            for unit_id in ids_to_remove:
                 log_entry = {'unit_id': unit_id, 'removal_reason': reason}
                 # Try to get corresponding doc_id
                 doc_id_val = df_to_filter.loc[df_to_filter[unit_id_col] == unit_id, doc_id_col].iloc[0] if doc_id_col in df_to_filter.columns else None
                 if doc_id_val is not None: log_entry['doc_id'] = doc_id_val
                 removal_details.append(log_entry)


    # 1. Filter by Text Length (Characters)
    min_chars = opts.get('min_char_length', None)
    max_chars = opts.get('max_char_length', None)
    if min_chars is not None or max_chars is not None:
        lengths = df_to_filter[text_unit_col].astype(str).str.len()
        mask = True
        reason_parts = []
        if min_chars is not None:
            mask &= (lengths >= min_chars)
            reason_parts.append(f"char_len<{min_chars}")
        if max_chars is not None:
            mask &= (lengths <= max_chars)
            reason_parts.append(f"char_len>{max_chars}")
        failed_mask = ~mask
        if failed_mask.any():
             log_removal(df_to_filter[failed_mask][unit_id_col], " | ".join(reason_parts))
             df_to_filter = df_to_filter[mask]
        if df_to_filter.empty: # Early exit if empty
             log_df = pd.DataFrame(removal_details) if removal_details else pd.DataFrame()
             logging.warning("DataFrame empty after character length filtering.")
             return df_to_filter, log_df

    # 2. Filter by Text Length (Words)
    min_words = opts.get('min_word_length', None)
    max_words = opts.get('max_word_length', None)
    if min_words is not None or max_words is not None:
        word_counts = df_to_filter[text_unit_col].astype(str).str.split().str.len()
        mask = True
        reason_parts = []
        if min_words is not None:
            mask &= (word_counts >= min_words)
            reason_parts.append(f"word_len<{min_words}")
        if max_words is not None:
            mask &= (word_counts <= max_words)
            reason_parts.append(f"word_len>{max_words}")
        failed_mask = ~mask
        if failed_mask.any():
            log_removal(df_to_filter[failed_mask][unit_id_col], " | ".join(reason_parts))
            df_to_filter = df_to_filter[mask]
        if df_to_filter.empty:
             log_df = pd.DataFrame(removal_details) if removal_details else pd.DataFrame()
             logging.warning("DataFrame empty after word length filtering.")
             return df_to_filter, log_df

    # 3. Filter by Metadata
    metadata_filters = opts.get('metadata_filters', None)
    if metadata_filters:
        if isinstance(metadata_filters, dict): metadata_filters = [metadata_filters]
        if isinstance(metadata_filters, list):
            logging.debug(f"Applying metadata filters: {metadata_filters}...")
            for f_dict in metadata_filters:
                if not isinstance(f_dict, dict) or 'column' not in f_dict or 'condition' not in f_dict:
                     logging.warning(f"Invalid metadata filter format: {f_dict}. Skipping."); continue
                col, cond = f_dict['column'], f_dict['condition']
                if col not in df_to_filter.columns:
                    logging.warning(f"Metadata filter column '{col}' not found. Skipping."); continue
                # Basic validation
                if not re.fullmatch(r'[a-zA-Z0-9_]+', col): logging.warning(f"Invalid col name '{col}'. Skipping."); continue
                if not re.match(r'^([><=!]=?|in)\s+.*$', cond.strip()): logging.warning(f"Unsafe condition '{cond}'. Skipping."); continue

                query_part = f"`{col}` {cond}" # Use backticks for safety with df.query
                reason = f"metadata_filter:({query_part})"
                try:
                    # Identify rows that *pass* the condition
                    passing_mask = df_to_filter.eval(query_part, engine='python')
                    failed_mask = ~passing_mask
                    if failed_mask.any():
                         log_removal(df_to_filter[failed_mask][unit_id_col], reason)
                         df_to_filter = df_to_filter[passing_mask]
                    if df_to_filter.empty:
                         log_df = pd.DataFrame(removal_details) if removal_details else pd.DataFrame()
                         logging.warning(f"DataFrame empty after metadata filter: {query_part}")
                         return df_to_filter, log_df
                except Exception as e:
                    logging.error(f"Error applying metadata query '{query_part}': {e}. Skipping this filter.")
        else:
             logging.warning(f"metadata_filters invalid format: {type(metadata_filters)}. Skipping.")


    # 4. Remove Exact Duplicates (based on text_unit)
    if opts.get('deduplicate_exact', True):
        duplicates_mask = df_to_filter.duplicated(subset=[text_unit_col], keep='first')
        if duplicates_mask.any():
            log_removal(df_to_filter[duplicates_mask][unit_id_col], 'exact_duplicate_text')
            df_to_filter = df_to_filter[~duplicates_mask]
        if df_to_filter.empty:
             log_df = pd.DataFrame(removal_details) if removal_details else pd.DataFrame()
             logging.warning("DataFrame empty after exact duplicate removal.")
             return df_to_filter, log_df

    # 5. Remove Similar Duplicates (Placeholder)
    if opts.get('deduplicate_similar_threshold', None) is not None:
        logging.warning("Similarity-based deduplication not implemented yet.")

    # --- Finalize ---
    final_rows = len(df_to_filter)
    total_removed = initial_rows - final_rows
    logging.info(f"Filtering complete. Final row count: {final_rows} (Removed {total_removed} rows in total).")

    # Convert log list to DataFrame
    removal_log_df = pd.DataFrame(removal_details) if removal_details else pd.DataFrame()

    return df_to_filter, removal_log_df
