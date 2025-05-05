# -*- coding: utf-8 -*-
"""
Modeling module for the BERTopic pipeline.

Handles embedding generation and BERTopic model training.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple

# Import SentenceTransformer for embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logging.error("SentenceTransformers library not found. Please install it: pip install sentence-transformers")
    SentenceTransformer = None # Define as None to avoid NameError later if not installed

# Import BERTopic and its components
try:
    from bertopic import BERTopic
    from bertopic.representation import (
        BaseRepresentation, # Base class
        KeyBERTInspired,
        PartOfSpeech,
        MaximalMarginalRelevance,
        ZeroShotClassification, # Example of another model
        TextGeneration # Example for generative models
        # Import other representation models as needed
    )
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
except ImportError as e:
     logging.error(f"Required libraries (bertopic, umap-learn, hdbscan-learn, scikit-learn) not found. Please install them. Error: {e}")
     # Define placeholders to avoid NameErrors if imports fail but code proceeds
     BERTopic = None
     UMAP = None
     HDBSCAN = None
     CountVectorizer = None
     BaseRepresentation = None
     KeyBERTInspired = None
     PartOfSpeech = None
     MaximalMarginalRelevance = None
     ZeroShotClassification = None
     TextGeneration = None


# Import utility functions
try:
    from src import utils # Assumes src is in the python path or running from root
except ImportError:
    import utils # Fallback for direct execution or different environment setup

# --- Constants ---
DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2' # A common default
DEFAULT_TEXT_UNIT_COL = 'text_unit' # Matches output from preprocessing
DEFAULT_DOC_ID_COL = 'doc_id' # Matches output from preprocessing
DEFAULT_UNIT_ID_COL = 'unit_id' # Matches output from preprocessing


# --- Embedding Generation ---

def generate_embeddings(
    processed_data_path: str,
    output_path: str,
    config_path: Optional[str] = None,
    force_recompute: bool = False,
    **kwargs: Any
) -> np.ndarray:
    """
    Generates sentence embeddings for the processed text data.
    (Implementation from previous step - unchanged)
    """
    logging.info(f"Starting embedding generation for: {processed_data_path}")
    if SentenceTransformer is None: raise ImportError("SentenceTransformers library required.")
    config = {};
    if config_path: config = utils.load_config(config_path)
    config.update(kwargs); logging.debug(f"Using embedding config: {config}")
    model_name = config.get('embedding_model_name', DEFAULT_EMBEDDING_MODEL)
    text_column = config.get('text_column', DEFAULT_TEXT_UNIT_COL)
    batch_size = config.get('batch_size', 32)
    show_progress_bar = config.get('show_progress_bar', True)
    force_recompute = config.get('force_recompute', force_recompute)
    if not output_path.endswith('.npy'): output_path += '.npy'
    if not force_recompute and utils.check_cache(output_path):
        logging.info(f"Loading cached embeddings: {output_path}")
        try: return np.load(output_path)
        except Exception as e: logging.warning(f"Failed load cached embeddings: {e}. Recomputing.")
    logging.info(f"Loading processed data: {processed_data_path}")
    if not os.path.exists(processed_data_path): raise FileNotFoundError(f"Data not found: {processed_data_path}")
    try: df = pd.read_csv(processed_data_path)
    except Exception as e: logging.error(f"Failed load processed CSV: {e}"); raise
    if text_column not in df.columns: raise ValueError(f"Text column '{text_column}' not found.")
    texts_to_embed = df[text_column].fillna('').astype(str).tolist()
    if not texts_to_embed: logging.warning("No text data found."); embeddings = np.array([])
    else:
        logging.info(f"Loaded {len(texts_to_embed)} text units.")
        logging.info(f"Initializing SentenceTransformer: {model_name}")
        try: model = SentenceTransformer(model_name); logging.info(f"Using device: {model.device}")
        except Exception as e: logging.error(f"Failed init SentenceTransformer: {e}"); raise
        logging.info("Generating embeddings...");
        try:
            embeddings = model.encode(texts_to_embed, batch_size=batch_size, show_progress_bar=show_progress_bar)
            logging.info(f"Generated embeddings shape: {embeddings.shape}")
        except Exception as e: logging.error(f"Embedding generation error: {e}"); raise
    logging.info(f"Saving embeddings to: {output_path}")
    try:
        output_dir = os.path.dirname(output_path);
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        np.save(output_path, embeddings); logging.info("Embeddings saved.")
    except Exception as e: logging.error(f"Failed save embeddings: {e}"); raise
    return embeddings


# --- BERTopic Training ---

def _initialize_component(component_class: type, params: Optional[Dict[str, Any]], component_name: str) -> Optional[object]:
    """Helper to initialize a BERTopic component (UMAP, HDBSCAN, etc.) from parameters."""
    if params is None:
        logging.debug(f"No parameters provided for {component_name}. BERTopic will use its default.")
        return None # Let BERTopic handle the default
    if not isinstance(params, dict):
        logging.warning(f"Parameters for {component_name} should be a dictionary, got {type(params)}. Using default.")
        return None

    try:
        # Special handling for random_state if needed? UMAP/HDBSCAN usually accept it directly.
        instance = component_class(**params)
        logging.info(f"Initialized {component_name} with parameters: {params}")
        return instance
    except Exception as e:
        logging.error(f"Failed to initialize {component_name} with params {params}: {e}. Using BERTopic default.")
        return None # Fallback to default on error

def _initialize_representation_model(config: Dict[str, Any]) -> Optional[Union[BaseRepresentation, Dict[str, BaseRepresentation]]]:
    """Initializes the representation model(s) based on config."""
    rep_model_config = config.get('representation_params', None)
    if rep_model_config is None:
        logging.debug("No representation_params specified. Using BERTopic default (c-TF-IDF).")
        return None

    # Map model names (strings) to actual classes
    model_class_map = {
        "KeyBERTInspired": KeyBERTInspired,
        "PartOfSpeech": PartOfSpeech,
        "MaximalMarginalRelevance": MaximalMarginalRelevance,
        "ZeroShotClassification": ZeroShotClassification,
        "TextGeneration": TextGeneration,
        # Add other BERTopic representation models here
    }

    # Handle single model config (dict) or multiple models (list of dicts -> BERTopic handles as dict)
    if isinstance(rep_model_config, dict):
        # Check if it's a single model definition or already a dict for BERTopic's multicall
        if 'model_name' in rep_model_config: # Assume single model definition
            model_name = rep_model_config.get('model_name')
            model_params = rep_model_config.get('params', {})
            model_class = model_class_map.get(model_name)
            if model_class:
                 logging.info(f"Initializing single representation model: {model_name}")
                 try:
                     return model_class(**model_params)
                 except Exception as e:
                      logging.error(f"Failed to initialize {model_name} with params {model_params}: {e}. Using default.")
                      return None
            else:
                logging.warning(f"Unknown representation model name: '{model_name}'. Using default.")
                return None
        else: # Assume it's already a dictionary for multi-aspect representation {aspect_name: model_instance}
             logging.info("Using pre-defined dictionary for multi-aspect representation models.")
             # We expect the values in this dict to be actual model instances,
             # which is hard to configure via YAML. This structure is better suited
             # for direct Python definition. For YAML, prefer the list format below.
             # Let's return None here if it's not a single model def, forcing list usage for YAML.
             logging.warning("Direct dictionary configuration for representation_params is complex via YAML. Use list format instead. Using default.")
             return None

    elif isinstance(rep_model_config, list): # List of model definitions for multi-aspect
        representation_models = {}
        logging.info("Initializing multiple representation models from list...")
        for i, model_def in enumerate(rep_model_config):
            if not isinstance(model_def, dict) or 'model_name' not in model_def:
                 logging.warning(f"Invalid format in representation_params list item {i}: {model_def}. Skipping.")
                 continue

            model_name = model_def.get('model_name')
            model_params = model_def.get('params', {})
            aspect_name = model_def.get('aspect_name', f"aspect_{i}") # Assign default name if missing
            model_class = model_class_map.get(model_name)

            if model_class:
                 try:
                     instance = model_class(**model_params)
                     representation_models[aspect_name] = instance
                     logging.info(f"Initialized representation model '{aspect_name}': {model_name} with params {model_params}")
                 except Exception as e:
                      logging.error(f"Failed to initialize {model_name} (aspect '{aspect_name}') with params {model_params}: {e}. Skipping this aspect.")
            else:
                 logging.warning(f"Unknown representation model name '{model_name}' for aspect '{aspect_name}'. Skipping.")

        if not representation_models:
             logging.warning("No valid representation models initialized from list. Using default.")
             return None
        return representation_models # Return dict {aspect_name: model_instance}
    else:
        logging.warning(f"Invalid format for 'representation_params': {type(rep_model_config)}. Expected dict or list. Using default.")
        return None


def train_bertopic_model(
    processed_data_path: str,
    embeddings_path: str,
    output_dir: str,
    run_name: str,
    config_path: Optional[str] = None,
    **kwargs: Any
) -> Optional[BERTopic]:
    """
    Trains a BERTopic model using pre-computed embeddings and specified configurations.

    Initializes components, fits the model, saves the model and associated outputs.

    Args:
        processed_data_path: Path to the processed data CSV file.
        embeddings_path: Path to the pre-computed embeddings (.npy file).
        output_dir: Base directory to save outputs (models, topic_info, logs).
        run_name: A unique name for this run (used for naming output files).
        config_path: (Optional) Path to YAML config file specifying parameters.
        **kwargs: Parameters passed directly, overriding config file values.

    Returns:
        The trained BERTopic model object, or None if training failed.
    """
    logging.info(f"--- Starting BERTopic Training Run: {run_name} ---")

    if BERTopic is None:
        logging.error("BERTopic library not found. Cannot train model.")
        return None

    # 1. Load Configuration
    config = {}
    if config_path:
        try:
            config = utils.load_config(config_path)
            logging.info(f"Loaded training configuration from {config_path}")
        except Exception as e:
            logging.error(f"Failed to load config from {config_path}: {e}"); return None
    config.update(kwargs)
    logging.debug(f"Using effective training configuration: {config}")

    # Create output directories
    model_save_dir = os.path.join(output_dir, "models")
    topic_info_save_dir = os.path.join(output_dir, "topic_info")
    log_save_dir = os.path.join(output_dir, "logs")
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(topic_info_save_dir, exist_ok=True)
    os.makedirs(log_save_dir, exist_ok=True)

    # 2. Load Data and Embeddings
    try:
        logging.info(f"Loading processed data: {processed_data_path}")
        df = pd.read_csv(processed_data_path)
        # Identify required columns based on config/defaults
        text_column = config.get('text_column', DEFAULT_TEXT_UNIT_COL)
        timestamp_column = config.get('timestamp_column', None) # Optional for dynamic topic modeling
        if text_column not in df.columns: raise ValueError(f"Text column '{text_column}' not found.")
        if timestamp_column and timestamp_column not in df.columns: logging.warning(f"Timestamp column '{timestamp_column}' not found. Dynamic modeling disabled."); timestamp_column = None

        docs = df[text_column].astype(str).tolist()
        timestamps = df[timestamp_column].tolist() if timestamp_column else None

        logging.info(f"Loading embeddings: {embeddings_path}")
        embeddings = np.load(embeddings_path)

        if len(docs) != embeddings.shape[0]:
             raise ValueError(f"Mismatch between number of documents ({len(docs)}) and embeddings ({embeddings.shape[0]}).")
        logging.info(f"Loaded {len(docs)} documents/units and embeddings.")

    except FileNotFoundError as e:
        logging.error(f"Data or embeddings file not found: {e}"); return None
    except ValueError as e:
        logging.error(f"Data loading/validation error: {e}"); return None
    except Exception as e:
        logging.error(f"Unexpected error loading data/embeddings: {e}"); return None


    # 3. Initialize BERTopic Components from Config
    logging.info("Initializing BERTopic components...")
    umap_params = config.get('umap_params', None)
    hdbscan_params = config.get('hdbscan_params', None)
    vectorizer_params = config.get('vectorizer_params', None)
    # Representation params handled by helper
    bertopic_params = config.get('bertopic_params', {}) # Other direct BERTopic params

    umap_model = _initialize_component(UMAP, umap_params, "UMAP")
    hdbscan_model = _initialize_component(HDBSCAN, hdbscan_params, "HDBSCAN")
    vectorizer_model = _initialize_component(CountVectorizer, vectorizer_params, "CountVectorizer")
    representation_model = _initialize_representation_model(config)


    # 4. Instantiate BERTopic Model
    logging.info("Instantiating BERTopic model...")
    try:
        topic_model = BERTopic(
            # Pass components if they were successfully initialized, otherwise let BERTopic use defaults
            umap_model=umap_model if umap_model else None,
            hdbscan_model=hdbscan_model if hdbscan_model else None,
            vectorizer_model=vectorizer_model if vectorizer_model else None,
            representation_model=representation_model if representation_model else None,
            # Pass other BERTopic parameters directly from config
            verbose=bertopic_params.get('verbose', True),
            language=bertopic_params.get('language', 'english'), # Ensure consistency or let BERTopic decide
            calculate_probabilities=bertopic_params.get('calculate_probabilities', True), # Needed for saving probabilities
            # Add other relevant BERTopic params here: min_topic_size, nr_topics, etc.
            min_topic_size=bertopic_params.get('min_topic_size', 10),
            nr_topics=bertopic_params.get('nr_topics', None),
            # embedding_model = None # We are passing pre-computed embeddings
            **{k: v for k, v in bertopic_params.items() if k not in ['verbose', 'language', 'calculate_probabilities', 'min_topic_size', 'nr_topics']} # Pass remaining params
        )
    except Exception as e:
        logging.error(f"Failed to instantiate BERTopic model: {e}")
        return None

    # 5. Fit Model
    logging.info("Fitting BERTopic model...")
    try:
        # Use timestamps if available
        topics, probs = topic_model.fit_transform(docs, embeddings=embeddings, y=timestamps) # Pass timestamps to 'y' if doing DTM
        logging.info(f"Model fitting complete. Found {len(topic_model.get_topic_info()) - 1} topics.") # -1 for outlier topic
    except Exception as e:
        logging.error(f"Error during BERTopic model fitting: {e}")
        return None

    # 6. Save Model and Outputs
    logging.info("Saving model and results...")
    save_format = config.get('bertopic_save_serialization', 'safetensors') # Default to safetensors
    model_filename = f"{run_name}_model.{'safetensors' if save_format=='safetensors' else 'pkl'}"
    model_save_path = os.path.join(model_save_dir, model_filename)

    topic_info_filename = f"{run_name}_topic_info.csv"
    topic_info_path = os.path.join(topic_info_save_dir, topic_info_filename)

    doc_info_filename = f"{run_name}_document_info.csv"
    doc_info_path = os.path.join(topic_info_save_dir, doc_info_filename)

    probs_filename = f"{run_name}_probabilities.npy"
    probs_path = os.path.join(topic_info_save_dir, probs_filename)

    config_log_filename = f"{run_name}_config.yaml"
    config_log_path = os.path.join(log_save_dir, config_log_filename)

    try:
        # Save BERTopic model
        logging.info(f"Saving model to: {model_save_path} (format: {save_format})")
        topic_model.save(model_save_path, serialization=save_format, save_embedding_model=False) # Don't save embedder

        # Save Topic Info
        logging.info(f"Saving topic info to: {topic_info_path}")
        topic_info_df = topic_model.get_topic_info()
        topic_info_df.to_csv(topic_info_path, index=False)

        # Save Document Info (mapping docs to topics)
        logging.info(f"Saving document info to: {doc_info_path}")
        # Get document info, ensuring original doc_id and unit_id are included
        doc_info_df = topic_model.get_document_info(docs, df=df[[config.get('column_mapping', {}).get('output_id_col', DEFAULT_DOC_ID_COL), DEFAULT_UNIT_ID_COL]]) # Pass original IDs
        doc_info_df.to_csv(doc_info_path, index=False)

        # Save Probabilities (if calculated)
        if probs is not None and topic_model.calculate_probabilities:
             logging.info(f"Saving probabilities to: {probs_path}")
             np.save(probs_path, probs)
        elif not topic_model.calculate_probabilities:
             logging.info("Probabilities were not calculated, skipping save.")
        else:
             logging.info("Probabilities object is None, skipping save.")


        # Save Run Configuration Log
        logging.info(f"Saving run configuration log to: {config_log_path}")
        utils.save_config_log(config, config_log_path, format='yaml')

        logging.info("All outputs saved successfully.")

    except Exception as e:
        logging.error(f"Error during saving of model or outputs: {e}")
        # Don't return None here, model was trained, just saving failed. Maybe return model anyway?
        # Let's return the model but log the error prominently.
        logging.error("!!! Model training succeeded, but saving outputs failed !!!")

    logging.info(f"--- Finished BERTopic Training Run: {run_name} ---")
    return topic_model

