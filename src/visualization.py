# -*- coding: utf-8 -*-
"""
Visualization module for the BERTopic pipeline.

Generates and saves various visualizations from a trained BERTopic model.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional

# Import BERTopic
try:
    from bertopic import BERTopic
except ImportError as e:
     logging.error(f"BERTopic library not found. Please install it. Error: {e}")
     BERTopic = None # Define placeholder

# Import Plotly for saving figures if needed (BERTopic usually handles this)
# Make sure plotly and kaleido are installed for static image saving:
# pip install plotly kaleido
try:
    import plotly.graph_objects as go
    plotly_available = True
except ImportError:
    logging.warning("Plotly library not found. Saving visualizations might fail.")
    plotly_available = False
    go = None # Placeholder

# Import utility functions
try:
    from src import utils # Assumes src is in the python path or running from root
except ImportError:
    import utils # Fallback for direct execution or different environment setup

# --- Constants ---
DEFAULT_TEXT_UNIT_COL = 'text_unit' # Matches output from preprocessing
DEFAULT_SAVE_FORMAT_INTERACTIVE = 'html'
DEFAULT_SAVE_FORMAT_STATIC = 'png' # Requires kaleido


# --- Visualization Generation ---

def generate_visualizations(
    model_path: str,
    output_dir: str,
    run_name: str,
    processed_data_path: Optional[str] = None,
    embeddings_path: Optional[str] = None,
    config_path: Optional[str] = None,
    **kwargs: Any
) -> None:
    """
    Loads a trained BERTopic model and generates specified visualizations.

    Args:
        model_path: Path to the saved BERTopic model (.pkl or .safetensors).
        output_dir: Base directory where visualizations subdirectory will be created.
        run_name: A unique name for the run (used for naming the subdirectory).
        processed_data_path: (Optional) Path to the processed data CSV file
                               (needed for docs/timestamps in some visualizations).
        embeddings_path: (Optional) Path to the pre-computed embeddings (.npy file)
                           (needed for visualize_documents).
        config_path: (Optional) Path to YAML config file specifying parameters.
        **kwargs: Parameters passed directly, overriding config file values.
                  Expected keys in 'visualization_options':
                  'visualizations_to_generate', 'save_format_interactive',
                  'save_format_static', 'timestamp_column', 'text_column',
                  plot-specific params (e.g., 'vis_topics_params', 'vis_docs_params').

    Returns:
        None. Saves files to disk.

    Raises:
        FileNotFoundError: If the model file or required data files are not found.
        ValueError: For invalid configurations.
        Exception: For errors during visualization generation or saving.
    """
    logging.info(f"--- Starting Visualization Generation for Run: {run_name} ---")

    if BERTopic is None:
        logging.error("BERTopic library not found. Cannot generate visualizations.")
        return
    if not plotly_available:
         logging.error("Plotly library not found. Cannot generate visualizations.")
         return

    # 1. Load Configuration
    config = {}
    if config_path:
        try:
            config = utils.load_config(config_path)
            logging.info(f"Loaded visualization configuration from {config_path}")
        except Exception as e:
            logging.error(f"Failed to load config from {config_path}: {e}"); return
    config.update(kwargs)
    vis_opts = config.get('visualization_options', {}) # Get sub-dictionary
    logging.debug(f"Using effective visualization options: {vis_opts}")

    # Create output directory
    vis_save_dir = os.path.join(output_dir, "visualizations", run_name)
    try:
        os.makedirs(vis_save_dir, exist_ok=True)
        logging.info(f"Ensured visualization output directory exists: {vis_save_dir}")
    except OSError as e:
        logging.error(f"Could not create visualization output directory {vis_save_dir}: {e}")
        return # Cannot proceed without output directory

    # 2. Load BERTopic Model
    logging.info(f"Loading BERTopic model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"BERTopic model file not found: {model_path}")
    try:
        topic_model = BERTopic.load(model_path)
        logging.info("BERTopic model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load BERTopic model from {model_path}: {e}")
        raise

    # 3. Load Required Data (Conditionally)
    visualizations_to_generate = vis_opts.get('visualizations_to_generate', [
        'topics', 'hierarchy', 'barchart', 'heatmap', 'documents', 'topics_over_time' # Default list
    ])
    if not isinstance(visualizations_to_generate, list):
        logging.warning("visualizations_to_generate should be a list. Using default.")
        visualizations_to_generate = ['topics', 'hierarchy', 'barchart', 'heatmap', 'documents', 'topics_over_time']


    docs = None
    timestamps = None
    embeddings = None
    text_column = vis_opts.get('text_column', DEFAULT_TEXT_UNIT_COL)
    timestamp_column = vis_opts.get('timestamp_column', None)

    needs_docs = any(viz in ['documents', 'topics_over_time'] for viz in visualizations_to_generate)
    needs_timestamps = 'topics_over_time' in visualizations_to_generate
    needs_embeddings = 'documents' in visualizations_to_generate

    if needs_docs or needs_timestamps:
        if not processed_data_path or not os.path.exists(processed_data_path):
            logging.error(f"Processed data file path required but not found or specified: {processed_data_path}")
            raise FileNotFoundError(f"Processed data file needed but not found: {processed_data_path}")
        try:
            logging.info(f"Loading processed data for visualizations: {processed_data_path}")
            df = pd.read_csv(processed_data_path)
            if text_column not in df.columns: raise ValueError(f"Text column '{text_column}' not found.")
            docs = df[text_column].astype(str).tolist()

            if needs_timestamps:
                if not timestamp_column or timestamp_column not in df.columns:
                    logging.warning(f"Timestamp column '{timestamp_column}' not found or specified. Cannot generate 'topics_over_time'.")
                    visualizations_to_generate = [v for v in visualizations_to_generate if v != 'topics_over_time'] # Remove it
                else:
                    timestamps = df[timestamp_column].tolist()
                    logging.info("Timestamps loaded for 'topics_over_time'.")
            logging.info("Documents loaded for visualizations.")
        except Exception as e:
            logging.error(f"Failed to load or process data from {processed_data_path}: {e}")
            raise

    if needs_embeddings:
        if not embeddings_path or not os.path.exists(embeddings_path):
            logging.error(f"Embeddings file path required but not found or specified: {embeddings_path}")
            raise FileNotFoundError(f"Embeddings file needed but not found: {embeddings_path}")
        try:
            logging.info(f"Loading embeddings for visualizations: {embeddings_path}")
            embeddings = np.load(embeddings_path)
            # Basic check - should match doc length if docs were loaded
            if docs is not None and len(docs) != embeddings.shape[0]:
                 logging.warning(f"Mismatch between loaded docs ({len(docs)}) and embeddings ({embeddings.shape[0]}) for visualize_documents.")
                 # Proceed? Or raise error? Let's warn and proceed, visualize_documents might handle it.
            logging.info("Embeddings loaded for visualizations.")
        except Exception as e:
            logging.error(f"Failed to load embeddings from {embeddings_path}: {e}")
            raise

    # 4. Generate and Save Visualizations
    save_html = vis_opts.get('save_format_interactive', DEFAULT_SAVE_FORMAT_INTERACTIVE).lower() == 'html'
    static_format = vis_opts.get('save_format_static', DEFAULT_SAVE_FORMAT_STATIC).lower()
    save_static = static_format in ['png', 'jpg', 'jpeg', 'svg', 'pdf'] # Kaleido supports these

    for viz_type in visualizations_to_generate:
        fig = None
        save_name_base = f"{run_name}_{viz_type}"
        html_path = os.path.join(vis_save_dir, f"{save_name_base}.html")
        static_path = os.path.join(vis_save_dir, f"{save_name_base}.{static_format}")

        logging.info(f"Generating visualization: {viz_type}")
        try:
            if viz_type == 'topics':
                params = vis_opts.get('vis_topics_params', {})
                fig = topic_model.visualize_topics(**params)
            elif viz_type == 'hierarchy':
                params = vis_opts.get('vis_hierarchy_params', {})
                fig = topic_model.visualize_hierarchy(**params)
            elif viz_type == 'barchart':
                params = vis_opts.get('vis_barchart_params', {})
                fig = topic_model.visualize_barchart(**params)
            elif viz_type == 'heatmap':
                params = vis_opts.get('vis_heatmap_params', {})
                fig = topic_model.visualize_heatmap(**params)
            elif viz_type == 'documents':
                if docs is None: logging.warning("Skipping 'documents' visualization: Docs not loaded."); continue
                # Embeddings are optional for visualize_documents, it can compute UMAP internally
                # But if provided, use them.
                params = vis_opts.get('vis_docs_params', {})
                fig = topic_model.visualize_documents(docs, embeddings=embeddings, **params)
            elif viz_type == 'topics_over_time':
                if docs is None or timestamps is None: logging.warning("Skipping 'topics_over_time': Docs or timestamps not available."); continue
                params = vis_opts.get('vis_topics_over_time_params', {})
                # This requires calculating topics over time first
                try:
                     topics_over_time_df = topic_model.topics_over_time(docs, timestamps)
                     fig = topic_model.visualize_topics_over_time(topics_over_time_df, **params)
                except Exception as e:
                     logging.error(f"Failed to calculate or visualize topics over time: {e}")
                     continue # Skip this plot
            else:
                logging.warning(f"Unknown visualization type requested: {viz_type}. Skipping.")
                continue

            # Save the generated figure
            if fig is not None:
                if save_html:
                    try:
                        logging.debug(f"Saving HTML to: {html_path}")
                        fig.write_html(html_path)
                    except Exception as e:
                        logging.error(f"Failed to save HTML for {viz_type}: {e}")
                if save_static:
                    try:
                        logging.debug(f"Saving static image ({static_format}) to: {static_path}")
                        fig.write_image(static_path)
                    except ValueError as ve:
                         if "requires the kaleido package" in str(ve):
                              logging.error(f"Failed to save static image ({static_format}) for {viz_type}: Kaleido package not found or not working. Please install/configure kaleido.")
                              # Disable static saving for future plots in this run?
                              # save_static = False # Optional: stop trying if kaleido fails once
                         else:
                              logging.error(f"Failed to save static image ({static_format}) for {viz_type}: {ve}")
                    except Exception as e:
                        logging.error(f"Failed to save static image ({static_format}) for {viz_type}: {e}")
            else:
                 logging.warning(f"No figure object generated for visualization type: {viz_type}")

        except Exception as e:
            logging.error(f"Failed to generate or save visualization '{viz_type}': {e}")
            # Continue to next visualization type

    logging.info(f"--- Finished Visualization Generation for Run: {run_name} ---")

