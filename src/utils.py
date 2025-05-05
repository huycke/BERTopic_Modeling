# -*- coding: utf-8 -*-
"""
Utility functions for the BERTopic pipeline project.

This module contains helper functions for configuration management,
file checking, and potentially logging or other common tasks.
"""

import os
import yaml
import json
import logging
from typing import Any, Dict, Optional

# --- Configuration Management ---

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary containing the configuration parameters.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the config file is invalid YAML.
        Exception: For other potential loading errors.
    """
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if config is None: # Handle empty YAML file case
             config = {}
        logging.info(f"Successfully loaded configuration from: {config_path}")
        return config
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {config_path}: {e}")
        raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred loading config {config_path}: {e}")
        raise Exception(f"An unexpected error occurred loading config {config_path}: {e}")

def save_config_log(config_dict: Dict[str, Any], output_path: str, format: str = 'yaml') -> None:
    """
    Saves the used configuration parameters to a log file (YAML or JSON).

    Ensures the output directory exists.

    Args:
        config_dict: The dictionary containing configuration parameters.
        output_path: The full path where the log file should be saved.
        format: The format to save in ('yaml' or 'json'). Defaults to 'yaml'.

    Raises:
        ValueError: If an unsupported format is specified.
        IOError: If the file cannot be written.
        Exception: For other unexpected errors.
    """
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir: # Check if output_dir is not empty (i.e., not saving in root)
             os.makedirs(output_dir, exist_ok=True)

        # Write the file in the specified format
        with open(output_path, 'w', encoding='utf-8') as f:
            if format.lower() == 'yaml':
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            elif format.lower() == 'json':
                json.dump(config_dict, f, indent=4)
            else:
                raise ValueError(f"Unsupported format specified: {format}. Use 'yaml' or 'json'.")
        logging.info(f"Configuration log saved to: {output_path} in {format} format.")

    except IOError as e:
        logging.error(f"Error writing config log to {output_path}: {e}")
        raise IOError(f"Error writing config log to {output_path}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred saving config log: {e}")
        raise Exception(f"An unexpected error occurred saving config log: {e}")


# --- File System Checks ---

def check_cache(file_path: str) -> bool:
    """
    Checks if a file exists at the given path.

    Args:
        file_path: The path to the file.

    Returns:
        True if the file exists, False otherwise.
    """
    exists = os.path.exists(file_path)
    if exists:
        logging.debug(f"Cache hit: File found at {file_path}")
    else:
        logging.debug(f"Cache miss: File not found at {file_path}")
    return exists

# --- (Optional) Basic Logging Setup ---
# You might want a function to configure logging centrally if needed later.
# Example:
# def setup_logging(level=logging.INFO, log_file: Optional[str] = None):
#     """Configures basic logging."""
#     log_format = '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
#     handlers = [logging.StreamHandler()] # Log to console by default
#     if log_file:
#         # Ensure log directory exists
#         log_dir = os.path.dirname(log_file)
#         if log_dir:
#              os.makedirs(log_dir, exist_ok=True)
#         handlers.append(logging.FileHandler(log_file))

#     logging.basicConfig(level=level, format=log_format, handlers=handlers)
#     logging.info("Logging configured.")

# Example usage (typically called once at the start of a script/notebook):
# setup_logging(log_file='pipeline.log')

