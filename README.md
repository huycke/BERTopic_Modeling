# BERTopic Modeling Repository for [Your Project Area, e.g., Reddit RPG Subreddits]

This repository contains code and notebooks for conducting topic modeling experiments using BERTopic, primarily focused on analyzing [Your Data Source, e.g., Reddit comments and submissions]. The project aims to [Your Main Goal, e.g., identify key discussion themes, track topic evolution, compare topics across communities].

## Project Goal

(Expand on this section. What specific questions are you trying to answer? What is the expected outcome or contribution of this analysis?)
* Example: Identify the primary topics discussed within selected RPG-related subreddits.
* Example: Compare the prevalence and nature of topics between subreddits like DMAcademy and rpghorrorstories.
* Example: Analyze how discussion themes change over time.

## Repository Structure

The project follows a structured approach to facilitate reproducibility and code reuse:

BERTopic_Modeling/
├── .gitignore           # Specifies intentionally untracked files Git should ignore
├── README.md            # This file: Project overview, setup, usage instructions
├── notebooks/           # Jupyter notebooks for testing, analysis, visualization
│   ├── 00_combine_raw_data.ipynb # (Optional) Example notebook for combining raw data
│   ├── 01_data_preprocessing.ipynb # Notebook to run/test preprocessing
│   ├── 02_embedding_generation.ipynb # Notebook to run/test embedding generation
│   ├── 03_topic_modeling.ipynb     # Notebook to run/test BERTopic training
│   └── 04_visualization_analysis.ipynb # Notebook for visualization & further analysis
├── src/                 # Source code (reusable Python modules)
│   ├── init.py      # Makes 'src' a Python package
│   ├── utils.py         # Utility functions (config loading, saving, etc.)
│   ├── preprocessing.py # Data loading, cleaning, filtering, unitization
│   ├── modeling.py      # Embedding generation, BERTopic model training
│   └── visualization.py # Visualization generation functions
├── data/                # Data files (Ignored by Git - Use DVC/LFS if needed)
│   ├── raw/             # Raw input data (e.g., CSVs from Pushshift)
│   ├── combined/        # (Optional) Combined raw data files
│   └── processed/       # Preprocessed data ready for modeling & removal logs
├── models/              # Saved trained models & embeddings (Ignored by Git - Use DVC/LFS)
│   └── bertopic/        # Subdirectory for BERTopic models & embeddings
├── results/             # Output files like topic info, plots (Ignored by Git)
│   ├── topic_info/      # CSV/text files with topic details, document assignments
│   ├── visualizations/  # Saved plots (HTML, PNG, etc.)
│   └── logs/            # Configuration logs for runs
├── configs/             # Configuration files for pipeline parameters (YAML)
└── requirements.txt     # Python package dependencies
## Workflow Overview

1.  **(Optional) Combine Raw Data:** Use `scripts/combine_data.py` (or a similar notebook like `notebooks/00_combine_raw_data.ipynb`) to merge multiple raw CSV files (e.g., comments and submissions from different subreddits) into a single input file (e.g., saved in `data/combined/`).
2.  **Preprocess Data:** Run `notebooks/01_data_preprocessing.ipynb`. This notebook uses `src/preprocessing.py` to:
    * Load the raw (or combined) data.
    * Apply cleaning steps (configurable).
    * Unitize text (document, paragraph, sentence - configurable).
    * Apply filters (length, metadata, duplicates - configurable).
    * Save the processed data to `data/processed/`.
    * Save a detailed removal log (`*_removal_log.csv`) to `data/processed/` for reproducibility.
3.  **Generate Embeddings:** Run `notebooks/02_embedding_generation.ipynb`. This uses `src/modeling.py` to:
    * Load the processed data.
    * Generate text embeddings using a specified Sentence Transformer model (configurable).
    * Save embeddings to `models/bertopic/`.
4.  **Train BERTopic Model:** Run `notebooks/03_topic_modeling.ipynb`. This uses `src/modeling.py` to:
    * Load processed data and embeddings.
    * Configure BERTopic components (UMAP, HDBSCAN, Vectorizer, Representation models) via parameters or YAML.
    * Train the BERTopic model.
    * Save the trained model object to `models/bertopic/`.
    * Save topic info, document assignments, probabilities, and the run configuration log to `results/`.
5.  **Visualize & Analyze:** Run `notebooks/04_visualization_analysis.ipynb`. This uses `src/visualization.py` to:
    * Load a saved BERTopic model.
    * Generate various interactive and static visualizations (configurable).
    * Save visualizations to `results/visualizations/`.
    * Perform further analysis, comparisons, or prepare data for downstream tasks (e.g., LLM analysis).

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    # Replace 'your-username' with your actual GitHub username
    git clone [https://github.com/your-username/BERTopic_Modeling.git](https://github.com/your-username/BERTopic_Modeling.git)
    cd BERTopic_Modeling
    ```
2.  **Create and Activate Environment (Recommended):**
    * **Using venv:**
      ```bash
      python -m venv .venv
      # On Windows (PowerShell):
      .\.venv\Scripts\Activate.ps1
      # On Windows (cmd):
      .\.venv\Scripts\activate.bat
      # On macOS/Linux:
      source .venv/bin/activate
      ```
    * **Using Conda:**
      ```bash
      conda create -n bertopic_env python=3.9 # Or your preferred Python version
      conda activate bertopic_env
      ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For GPU support (PyTorch, cuML), manual installation steps are required. See comments in `requirements.txt`.*
4.  **Download NLTK Data:** Run this Python code once (e.g., in a notebook cell or Python interpreter within your activated environment):
    ```python
    import nltk
    import ssl
    # Handle potential SSL issues
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError: pass
    else: ssl._create_default_https_context = _create_unverified_https_context
    # Download necessary packages
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    ```
5.  **(If using DVC/LFS):** Add instructions here for pulling large data/model files.

## Configuration

Pipeline steps (preprocessing, embedding, modeling, visualization) are configured using parameters passed directly in notebooks or, preferably, via YAML files stored in the `configs/` directory. See example configurations within the notebooks and the function definitions in `src/` for available options.

## Running the Pipeline

Execute the Jupyter notebooks in the `notebooks/` directory sequentially (01 -> 04) to run the standard workflow. Modify the configuration parameters within each notebook (or point them to different YAML files) to run experiments.

## Reproducibility & Logging

* **Configuration Logs:** For each BERTopic model training run (`03_topic_modeling.ipynb`), the exact configuration parameters used are saved to a YAML file in `results/logs/`.
* **Removal Logs:** The preprocessing step (`01_data_preprocessing.ipynb`) generates a detailed CSV log (`*_removal_log.csv`) in `data/processed/`, listing each text unit removed during filtering and the reason why. This ensures transparency in the data cleaning process.

## Contributing

(Optional: Add guidelines if others might contribute, e.g., pull request process, coding standards.)

## License

(Optional: Specify your chosen license, e.g., MIT License. If you haven't chosen one, you can add it later.)

