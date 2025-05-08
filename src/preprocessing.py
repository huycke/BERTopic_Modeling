# src/preprocessing.py
import pandas as pd
import re
from html import unescape
import os

# --- Text Cleaning Utilities --- (Identical to v6_no_nltk)
def remove_html_tags(text):
    if not isinstance(text, str): return ""
    return re.sub(re.compile('<.*?>'), '', text)

def remove_urls(text):
    if not isinstance(text, str): return ""
    return re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.IGNORECASE)

def normalize_quotes(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"â€™|‘|’", "'", text)
    text = re.sub(r'“|”', '"', text)
    return text

def filter_characters(text, char_filter_regex=None):
    if not isinstance(text, str): return ""
    if char_filter_regex: text = re.sub(char_filter_regex, "", text)
    return text

def remove_html_entities(text):
    if not isinstance(text, str): return ""
    return re.sub(r'&[a-zA-Z]+;', '', text)

def configurable_text_cleaning(
    text,
    apply_unescape=True, apply_url_removal=True, apply_html_tag_removal=True,
    apply_quote_normalization=True, apply_char_filtering=True,
    char_filter_regex=r"[^a-zA-Z0-9\s,.!?':;\"-]",
    apply_html_entity_removal=True, apply_lowercase=True
):
    if pd.isna(text) or not isinstance(text, str) or re.match(r'^\s*$', str(text)): return ''
    if apply_unescape: text = unescape(text)
    if apply_url_removal: text = remove_urls(text)
    if apply_html_tag_removal: text = remove_html_tags(text)
    if apply_quote_normalization: text = normalize_quotes(text)
    if apply_char_filtering and char_filter_regex: text = filter_characters(text, char_filter_regex)
    if apply_html_entity_removal: text = remove_html_entities(text)
    if apply_lowercase: text = text.lower()
    return ' '.join(text.split()).strip()

# --- Filtering Functions --- (Identical to v6_no_nltk)
def filter_by_length(df, text_column, min_len=None, max_len=None):
    if not isinstance(df, pd.DataFrame) or text_column not in df.columns:
        print("Error: Invalid DataFrame or text_column for length filtering.")
        return df
    original_count = len(df)
    df[text_column] = df[text_column].astype(str) 
    if min_len is not None: df = df[df[text_column].str.len() >= min_len]
    if max_len is not None: df = df[df[text_column].str.len() <= max_len]
    removed_count = original_count - len(df)
    log_msg = f"Filtered by length (column: '{text_column}', min: {min_len}, max: {max_len}): "
    log_msg += f"Removed {removed_count} documents. Kept {len(df)} documents." if removed_count > 0 else f"No documents removed. Kept {len(df)} documents."
    print(log_msg)
    return df

def filter_by_score(df, score_column, min_score=None, max_score=None):
    if not isinstance(df, pd.DataFrame): print("Error: Invalid DataFrame for score filtering."); return df
    if score_column is None or score_column not in df.columns: print(f"Score column '{score_column}' not found. Skipping score filter."); return df
    if df[score_column].isnull().all(): print(f"Score column '{score_column}' all NaN. Skipping score filter."); return df
    original_count = len(df)
    df[score_column] = pd.to_numeric(df[score_column], errors='coerce')
    rows_before_nan_drop = len(df)
    df.dropna(subset=[score_column], inplace=True) 
    nan_dropped_count = rows_before_nan_drop - len(df)
    if nan_dropped_count > 0: print(f"Dropped {nan_dropped_count} rows with non-numeric/NaN scores in '{score_column}'.")
    rows_before_score_filter = len(df)
    if min_score is not None: df = df[df[score_column] >= min_score]
    if max_score is not None: df = df[df[score_column] <= max_score]
    score_filtered_count = rows_before_score_filter - len(df)
    total_removed_overall = original_count - len(df)
    log_msg = f"Filtered by score (column: '{score_column}', min: {min_score}, max: {max_score}): "
    if score_filtered_count > 0: log_msg += f"Removed {score_filtered_count} documents by score condition. "
    elif nan_dropped_count > 0 and score_filtered_count == 0 : log_msg += f"No documents removed by score condition (only by NaN conversion). "
    elif score_filtered_count == 0 and nan_dropped_count == 0: log_msg += f"No documents removed. "
    log_msg += f"Total removed (including NaN conversion): {total_removed_overall}. Kept {len(df)} documents."
    print(log_msg)
    return df

def remove_duplicates_by_column(df, column_to_check_duplicates):
    if not isinstance(df, pd.DataFrame) or column_to_check_duplicates not in df.columns: print("Error: Invalid DataFrame/column for duplicate removal."); return df
    original_count = len(df)
    df = df.drop_duplicates(subset=[column_to_check_duplicates], keep='first')
    removed_count = original_count - len(df)
    log_msg = f"Removed duplicates based on '{column_to_check_duplicates}': "
    log_msg += f"Removed {removed_count} documents. Kept {len(df)}." if removed_count > 0 else f"No duplicates found. Kept {len(df)}."
    print(log_msg)
    return df

# --- Main Data Loading and Preprocessing Function ---
def load_and_preprocess_data(
    file_path,
    text_source_columns,
    unique_id_column=None,
    required_columns_for_docs_creation=None, # NEW: List of columns that must not be empty
    dropped_rows_output_path=None,          # NEW: Path to save rows dropped due to missing required columns
    data_type_specific_df_processing=None, 
    clean_apply_unescape=True,
    clean_apply_url_removal=True,
    clean_apply_html_tag_removal=True,
    clean_apply_quote_normalization=True,
    clean_apply_char_filtering=True, 
    clean_char_filter_regex=r"[^a-zA-Z0-9\s,.!?':;\"-]", 
    clean_apply_html_entity_removal=True,
    clean_apply_lowercase=True,
    apply_length_filter=False,
    min_doc_length=None,
    max_doc_length=None,
    apply_duplicate_removal=True,
    column_for_duplicate_checking='docs',
    apply_score_filter=False,
    score_column_for_filtering=None,
    min_score_for_filtering=None,
    max_score_for_filtering=None,
):
    print(f"Starting preprocessing for: {file_path}")
    try:
        df = pd.read_csv(file_path, low_memory=False) 
    except FileNotFoundError: print(f"Error: File not found at {file_path}"); return None
    except Exception as e: print(f"Error loading CSV '{file_path}': {e}"); return None

    print(f"Original dataset shape: {df.shape}")

    # --- NEW: Drop rows with missing essential data for 'docs' creation ---
    df_dropped_missing_required = pd.DataFrame() # Initialize an empty DataFrame for dropped rows
    if required_columns_for_docs_creation:
        print(f"Checking for missing values in required columns: {required_columns_for_docs_creation}")
        # Create a boolean mask for rows to keep. A row is kept if all required columns are non-empty.
        # A value is considered "empty" if it's NaN or an empty string after stripping whitespace.
        mask_to_keep = pd.Series(True, index=df.index)
        for col in required_columns_for_docs_creation:
            if col in df.columns:
                # Check for NaN or (empty string or whitespace-only string)
                is_empty_or_nan = df[col].isnull() | (df[col].astype(str).str.strip() == '')
                mask_to_keep &= ~is_empty_or_nan # Keep if NOT empty or NaN
            else:
                print(f"Warning: Required column '{col}' not found in DataFrame. Skipping this check for this column.")
        
        rows_to_drop_indices = df.index[~mask_to_keep]
        if not rows_to_drop_indices.empty:
            df_dropped_missing_required = df.loc[rows_to_drop_indices].copy()
            df.drop(rows_to_drop_indices, inplace=True)
            print(f"Dropped {len(df_dropped_missing_required)} rows due to missing/empty values in required columns: {required_columns_for_docs_creation}.")
            
            if dropped_rows_output_path and not df_dropped_missing_required.empty:
                try:
                    output_dir = os.path.dirname(dropped_rows_output_path)
                    if output_dir and not os.path.exists(output_dir): # Check if output_dir is not empty
                        os.makedirs(output_dir)
                        print(f"Created directory for dropped rows: {output_dir}")
                    df_dropped_missing_required.to_csv(dropped_rows_output_path, index=False)
                    print(f"Saved {len(df_dropped_missing_required)} dropped rows to: {dropped_rows_output_path}")
                except Exception as e:
                    print(f"Error saving dropped rows: {e}")
        else:
            print("No rows dropped due to missing/empty values in required columns.")
    
    if df.empty:
        print("DataFrame is empty after checking for required columns. No further processing will occur.")
        return df # Return the empty DataFrame

    # --- Create 'docs' column ---
    missing_cols = [col for col in text_source_columns if col not in df.columns]
    if missing_cols: print(f"Error: Text source columns {missing_cols} not found in the (potentially filtered) DataFrame."); return df
    
    df['docs'] = df[text_source_columns].fillna('').astype(str).agg(' '.join, axis=1)
    df['docs'] = df['docs'].str.strip()
    print(f"Created 'docs' column from: {text_source_columns}")

    # --- Preserve or report on unique_id_column ---
    if unique_id_column:
        if unique_id_column in df.columns:
            if df[unique_id_column].nunique() != len(df): print(f"Warning: unique_id_column '{unique_id_column}' not unique.")
            print(f"Using '{unique_id_column}' as reference ID.")
        else: print(f"Warning: unique_id_column '{unique_id_column}' not found.")
    else: print("No unique_id_column specified.")

    # --- Apply Data-Type Specific DataFrame Processing ---
    if data_type_specific_df_processing:
        print("Applying data-type specific DataFrame processing...")
        try:
            df = data_type_specific_df_processing(df, text_columns_to_check=text_source_columns) 
            if not isinstance(df, pd.DataFrame): print("Error: specific processing didn't return DataFrame.")
            else: print("Finished specific DataFrame processing.")
        except Exception as e: print(f"Error during specific DataFrame processing: {e}.")

    # --- Apply Configurable Text Cleaning to 'docs' column ---
    print("Applying configurable text cleaning to 'docs' column...")
    df['docs'] = df['docs'].apply(
        configurable_text_cleaning, 
        apply_unescape=clean_apply_unescape,
        apply_url_removal=clean_apply_url_removal,
        apply_html_tag_removal=clean_apply_html_tag_removal,
        apply_quote_normalization=clean_apply_quote_normalization,
        apply_char_filtering=clean_apply_char_filtering, 
        char_filter_regex=clean_char_filter_regex,       
        apply_html_entity_removal=clean_apply_html_entity_removal,
        apply_lowercase=clean_apply_lowercase
    )
    original_len_after_specific_processing = len(df)
    df = df[df['docs'] != ''] # Remove rows where 'docs' became empty after text cleaning
    if len(df) < original_len_after_specific_processing:
        print(f"Dropped {original_len_after_specific_processing - len(df)} rows with empty 'docs' after text cleaning.")

    # --- Apply Filtering ---
    if apply_length_filter:
        df = filter_by_length(df, 'docs', min_doc_length, max_doc_length)
    if apply_score_filter:
        df = filter_by_score(df, score_column_for_filtering, min_score_for_filtering, max_score_for_filtering)
    if apply_duplicate_removal and column_for_duplicate_checking in df.columns:
        df = remove_duplicates_by_column(df, column_for_duplicate_checking)
    elif apply_duplicate_removal: 
        print(f"Warning: Column '{column_for_duplicate_checking}' for duplicate check not found. Skipping duplicate removal.")

    print(f"Finished preprocessing. Processed dataset shape: {df.shape}")
    if df.empty: print("Warning: DataFrame empty after final preprocessing steps.")
    return df

# --- Example Data-Type Specific Processing Functions --- (Identical to v6_no_nltk)
def reddit_specific_df_processing(df, text_columns_to_check=None):
    print("Applying Reddit specific DataFrame processing (removing [deleted]/[removed] content)...")
    if text_columns_to_check is None: text_columns_to_check = ['body', 'title', 'selftext'] 
    patterns_to_remove = r'\[deleted\]|\[removed\]|\[deleted by user\]|\[removed by user\]'
    initial_rows = len(df)
    combined_mask = pd.Series([False] * len(df), index=df.index) 
    for column in text_columns_to_check:
        if column in df.columns:
            current_mask = df[column].fillna('').astype(str).str.contains(patterns_to_remove, na=False, regex=True)
            combined_mask = combined_mask | current_mask 
    df_filtered = df[~combined_mask].copy() 
    removed_count = initial_rows - len(df_filtered)
    if removed_count > 0: print(f"Reddit specific: Removed {removed_count} rows containing deleted/removed patterns from columns: {text_columns_to_check}.")
    return df_filtered

def s2_specific_df_processing(df, text_columns_to_check=None):
    print("Applied S2 specific DataFrame processing (placeholder).")
    return df

# --- Test Execution Block ---
if __name__ == '__main__':
    dummy_data_s2 = {
        'corpusid': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'title': ['Title 1', 'Title 2', 'Title 3', 'Title 4', 'Title 5', 'Title 6', 'Title 7', 'Title 8', 'Title 9 Has Abstract', 'Title 10 No Abstract'],
        'abstract': ['Abstract 1.', 'Abstract 2.', None, 'Abstract 4.', '', 'Abstract 6.', '   ', 'Abstract 8.', 'Real Abstract Here', None], # Added None, empty, and whitespace
        'year': [2020, 2021, 2020, 2020, 2022, 2023, 2021, 2022, 2023, 2024],
        's2_score': [10, 5, 0, 8, 12, -2, 7, 9, 10, 11] # Not used for S2 filtering in this test
    }
    dummy_df_s2 = pd.DataFrame(dummy_data_s2)
    dummy_s2_file_path = 'dummy_s2_data_test.csv'
    dummy_dropped_s2_path = 'dummy_s2_data_dropped_required.csv'
    dummy_df_s2.to_csv(dummy_s2_file_path, index=False)

    print("--- Testing S2 Data Preprocessing with Required Columns Check ---")
    processed_s2_df = load_and_preprocess_data(
        file_path=dummy_s2_file_path,
        text_source_columns=['title', 'abstract'],
        unique_id_column='corpusid',
        required_columns_for_docs_creation=['abstract'], # Require 'abstract' to be non-empty
        dropped_rows_output_path=dummy_dropped_s2_path,
        data_type_specific_df_processing=s2_specific_df_processing,
        clean_char_filter_regex=r"[^a-zA-Z0-9\s'-]", 
        clean_apply_char_filtering=True, 
        apply_length_filter=True, min_doc_length=10, max_doc_length=500,
        apply_duplicate_removal=False, # Turn off for this simple test to see all kept rows
        apply_score_filter=False 
    )

    if processed_s2_df is not None:
        print("\n--- S2 Processed DataFrame Head ---")
        print(processed_s2_df.head(10)) # Show more rows
        print("\n--- Example 'docs' content (S2) ---")
        for i, row_idx in enumerate(processed_s2_df.head().index):
            doc_text = processed_s2_df.loc[row_idx, 'docs']
            print(f"Doc (CorpusID {processed_s2_df.loc[row_idx, 'corpusid']}): {doc_text[:150]}...")
    
    # Check content of dropped file
    if os.path.exists(dummy_dropped_s2_path):
        print(f"\n--- Content of Dropped Rows File ({dummy_dropped_s2_path}) ---")
        df_dropped_check = pd.read_csv(dummy_dropped_s2_path)
        print(df_dropped_check)
    else:
        print(f"\nDropped rows file not created: {dummy_dropped_s2_path}")


    # Clean up dummy files
    try: 
        if os.path.exists(dummy_s2_file_path): os.remove(dummy_s2_file_path)
        if os.path.exists(dummy_dropped_s2_path): os.remove(dummy_dropped_s2_path)
        print(f"\nCleaned up dummy files.")
    except OSError as e: print(f"Error removing dummy files: {e}")

