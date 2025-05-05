import pandas as pd
import os
import logging
from typing import List, Optional

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def combine_reddit_csvs(
    input_directory: str,
    output_path: str,
    subreddits_to_include: List[str],
    required_cols_comments: Optional[List[str]] = None,
    required_cols_submissions: Optional[List[str]] = None,
    final_cols_to_keep: Optional[List[str]] = None,
    require_selftext_for_submissions: bool = True
) -> None:
    """
    Combines Reddit submission and comment CSV files from a directory into a
    single CSV file with a unified 'text' column and selected metadata.

    Args:
        input_directory: Path to the directory containing raw CSV files.
        output_path: Path to save the combined CSV file.
        subreddits_to_include: List of subreddit names (case-insensitive) to filter filenames by.
        required_cols_comments: List of columns to load from comment CSVs.
                                Defaults to essential columns.
        required_cols_submissions: List of columns to load from submission CSVs.
                                   Defaults to essential columns.
        final_cols_to_keep: List of columns to keep in the final combined output.
                            Defaults to essential columns plus 'text'.
        require_selftext_for_submissions: If True, drops submission rows where
                                          'selftext' is missing or empty.
    """
    logging.info(f"Starting data combination process.")
    logging.info(f"Input directory: {input_directory}")
    logging.info(f"Output path: {output_path}")
    logging.info(f"Filtering for subreddits: {subreddits_to_include}")

    # Define default essential columns if not provided
    if required_cols_comments is None:
        required_cols_comments = ['id', 'link_id', 'author', 'created_utc', 'subreddit', 'body', 'score']
    if required_cols_submissions is None:
        required_cols_submissions = ['id', 'permalink', 'author', 'created_utc', 'subreddit', 'title', 'selftext', 'score']
    if final_cols_to_keep is None:
        # Combine defaults, ensure 'text' is included, remove potential duplicates
        final_cols_to_keep = list(set(required_cols_comments + required_cols_submissions + ['text']) - {'body', 'title', 'selftext', 'permalink'})
        # Adjust based on your actual needs, e.g., keep 'link_id' or not

    logging.info(f"Required comment columns: {required_cols_comments}")
    logging.info(f"Required submission columns: {required_cols_submissions}")
    logging.info(f"Final columns to keep: {final_cols_to_keep}")


    dataframes = []
    subreddits_lower = [sub.lower() for sub in subreddits_to_include]

    files_processed = 0
    files_skipped = 0

    for filename in os.listdir(input_directory):
        # Check if filename matches any subreddit and ends with .csv
        if not filename.endswith(".csv") or not any(subreddit in filename.lower() for subreddit in subreddits_lower):
            continue

        filepath = os.path.join(input_directory, filename)
        logging.debug(f"Processing file: {filename}")

        try:
            # Determine file type and required columns
            is_comment_file = '_comments' in filename.lower() # Basic check
            required_cols = required_cols_comments if is_comment_file else required_cols_submissions

            # Load only necessary columns
            df = pd.read_csv(filepath, usecols=lambda col: col in required_cols, low_memory=False)

            # --- Data Standardization and Text Creation ---
            if is_comment_file:
                if 'body' not in df.columns:
                     logging.warning(f"Column 'body' not found in comment file {filename}. Skipping file.")
                     files_skipped += 1
                     continue
                # Rename 'body' to 'text'
                df.rename(columns={'body': 'text'}, inplace=True)
                # Ensure 'text' column exists even if empty after rename
                if 'text' not in df.columns: df['text'] = ''

            else: # Submission file
                if 'title' not in df.columns or 'selftext' not in df.columns:
                     logging.warning(f"Columns 'title' or 'selftext' not found in submission file {filename}. Skipping file.")
                     files_skipped += 1
                     continue

                # Handle missing selftext based on flag
                if require_selftext_for_submissions:
                    initial_rows = len(df)
                    # Treat NaN and empty strings as missing
                    df = df[df['selftext'].notna() & (df['selftext'].astype(str).str.strip() != '')]
                    removed_count = initial_rows - len(df)
                    if removed_count > 0:
                        logging.info(f"Removed {removed_count} rows from {filename} due to missing/empty 'selftext'.")

                if df.empty:
                     logging.info(f"No valid rows remaining in {filename} after filtering selftext. Skipping file.")
                     files_skipped += 1
                     continue

                # Combine title and selftext
                df['title'] = df['title'].fillna('')
                df['selftext'] = df['selftext'].fillna('') # Should already be filtered if require_selftext_for_submissions=True
                df['text'] = df['title'].astype(str) + ' ' + df['selftext'].astype(str)
                df['text'] = df['text'].str.strip() # Remove leading/trailing whitespace

            # --- Column Selection for Concatenation ---
            # Select only the final desired columns that actually exist in the df
            cols_to_append = [col for col in final_cols_to_keep if col in df.columns]
            dataframes.append(df[cols_to_append])
            files_processed += 1
            logging.debug(f"Added {len(df)} rows from {filename}")

        except FileNotFoundError:
            logging.error(f"File not found: {filepath}. Skipping.")
            files_skipped += 1
        except pd.errors.EmptyDataError:
            logging.warning(f"File is empty: {filepath}. Skipping.")
            files_skipped += 1
        except ValueError as ve: # Handles errors from usecols if a required col is missing
             logging.warning(f"ValueError reading {filepath} (likely missing required columns): {ve}. Skipping.")
             files_skipped += 1
        except KeyError as ke:
            logging.warning(f"Missing expected column in {filepath}: {ke}. Skipping.")
            files_skipped += 1
        except Exception as e:
            logging.error(f"Failed to process {filepath}: {e}. Skipping.")
            files_skipped += 1


    if not dataframes:
        logging.warning("No valid dataframes were created. No output file generated.")
        return

    # Concatenate all DataFrames
    logging.info(f"Concatenating data from {files_processed} files...")
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Ensure final columns exist before saving (some might be missing if only one file type was processed)
    final_columns_present = [col for col in final_cols_to_keep if col in combined_df.columns]

    # Save the combined DataFrame
    logging.info(f"Saving combined data ({len(combined_df)} rows) to {output_path}...")
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        combined_df[final_columns_present].to_csv(output_path, index=False)
        logging.info(f"Successfully saved combined file.")
        logging.info(f"Total files processed: {files_processed}, Total files skipped: {files_skipped}")
    except Exception as e:
        logging.error(f"Failed to save combined file to {output_path}: {e}")


# --- Example Usage ---
if __name__ == "__main__":
    # Define configuration for the combination process
    INPUT_DIR = 'rpg/csv_working/' # CHANGE TO YOUR INPUT DIRECTORY
    OUTPUT_CSV = 'rpg/rpg_combined_data.csv' # CHANGE TO YOUR DESIRED OUTPUT FILE
    SUBREDDITS = [
        "AskGameMasters", "DMAcademy", "dndhorrorstories",
        "DungeonMasters", "lfg", "rpg", "rpghorrorstories"
    ]
    # Specify only columns you absolutely need for preprocessing/modeling
    FINAL_COLS = ['id', 'author', 'created_utc', 'subreddit', 'score', 'text', 'link_id'] # Example

    # Run the combination function
    combine_reddit_csvs(
        input_directory=INPUT_DIR,
        output_path=OUTPUT_CSV,
        subreddits_to_include=SUBREDDITS,
        final_cols_to_keep=FINAL_COLS,
        require_selftext_for_submissions=True # Set to False if you want to keep title-only posts
    )
