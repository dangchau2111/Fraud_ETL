"""
etl_pipeline.py

Main functions:
    - load_data(): Read dataset from CSV, clean NaN + duplicates, log before/after, return clean DataFrame
    - run_etl(): Run full ETL (extract -> transform -> load) and save the output
"""

import os
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

# -----------------------------
# Logging configuration
# -----------------------------
logger = logging.getLogger("fraud_etl")
logger.setLevel(logging.INFO)

if not logger.handlers:
    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = logs_dir / "etl_pipeline.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # File handler
    fh = logging.FileHandler(log_file_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)


# -----------------------------
# Helper: log basic DataFrame information
# -----------------------------
def log_dataframe_info(df: pd.DataFrame, title: str) -> None:
    """
    Log basic information about a DataFrame:
        - Shape (rows, columns)
        - NaN count per column
        - Data types per column
    """
    logger.info("=== %s ===", title)
    logger.info("Shape: %s (rows, cols)", df.shape)

    missing = df.isna().sum()
    if missing.sum() == 0:
        logger.info("Missing values: no NaN values detected")
    else:
        logger.info("Missing values per column:\n%s", missing.to_string())

    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.astype(str)
    }).to_string(index=False)

    logger.info("Column dtypes:\n%s", dtype_df)
    logger.info("=== End of %s ===", title)


# -----------------------------
# Core ETL helper functions
# -----------------------------
def get_data_path(custom_path: Optional[str] = None) -> Path:
    """
    Resolve the input CSV path.

    Priority:
        1. custom_path argument (if provided)
        2. FRAUD_DATA_PATH environment variable
        3. Default: data/creditcard.csv
    """
    if custom_path:
        return Path(custom_path)

    env_path = os.getenv("FRAUD_DATA_PATH")
    if env_path:
        return Path(env_path)

    return Path("data") / "creditcard.csv"


def get_output_path(custom_path: Optional[str] = None) -> Path:
    """
    Resolve the output Parquet path.

    Priority:
        1. custom_path argument (if provided)
        2. FRAUD_OUTPUT_PATH environment variable
        3. Default: data/creditcard_clean.parquet
    """
    if custom_path:
        return Path(custom_path)

    env_path = os.getenv("FRAUD_OUTPUT_PATH")
    if env_path:
        return Path(env_path)

    return Path("output") / "creditcard_clean.parquet"


def extract(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Read raw data from CSV and log initial information.

    Parameters
    ----------
    data_path : str, optional
        Custom path to the CSV file. If None, get_data_path() is used.

    Returns
    -------
    df : pandas.DataFrame
        Raw DataFrame read from CSV.
    """
    path = get_data_path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path.resolve()}")

    logger.info("Reading data from CSV: %s", path)
    df = pd.read_csv(path)

    logger.info("Loaded data with %d rows and %d columns", df.shape[0], df.shape[1])
    log_dataframe_info(df, "Raw data information (before cleaning)")

    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic transformation:
        - Drop rows containing any NaN value
        - Drop duplicated rows
        - Log information before and after cleaning

    Parameters
    ----------
    df : pandas.DataFrame
        Raw input DataFrame.

    Returns
    -------
    df_clean : pandas.DataFrame
        Cleaned DataFrame.
    """
    
    logger.info("Starting data transformation: handling NaN and duplicates...")

    # Log info before cleaning
    # log_dataframe_info(df, "Data BEFORE cleaning NaN & duplicates")

    # 1. Drop rows with any NaN
    before_drop_na = df.shape[0]
    total_nan_before = df.isna().sum().sum()
    df = df.dropna()
    after_drop_na = df.shape[0]
    total_nan_after = df.isna().sum().sum()

    logger.info(
        "Drop NaN: total NaN before = %d, after = %d | rows: %d -> %d",
        total_nan_before,
        total_nan_after,
        before_drop_na,
        after_drop_na,
    )

    # 2. Drop duplicate rows
    before_drop_dup = df.shape[0]
    df = df.drop_duplicates()
    after_drop_dup = df.shape[0]

    logger.info("Drop duplicates: %d -> %d rows", before_drop_dup, after_drop_dup)

    # Log info after cleaning
    # log_dataframe_info(df, "Data AFTER cleaning NaN & duplicates")

    logger.info("Finished transformation. Final row count: %d", df.shape[0])
    return df


def load(df: pd.DataFrame, output_path: Optional[str] = None) -> Path:
    """
    Save the cleaned DataFrame to a Parquet file.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned DataFrame to be saved.
    output_path : str, optional
        Custom output file path. If None, get_output_path() is used.

    Returns
    -------
    Path
        Path to the saved Parquet file.
    """
    path = get_output_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Writing cleaned data to Parquet: %s", path)
    # Requires pyarrow or fastparquet in your environment
    df.to_parquet(path, index=False)
    logger.info("Successfully wrote Parquet file: %s", path.resolve())
    return path


# -----------------------------
# Public API for tests / scripts
# -----------------------------
def load_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience API for unit tests or scripts.

    Steps:
        - Read CSV using pandas
        - Log initial data information
        - Clean NaN + duplicates and log final information
        - Return cleaned DataFrame
    """
    df_raw = extract(data_path=data_path)
    df_clean = transform(df_raw)
    return df_clean


def run_etl(
    data_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Path:
    """
    Run the full ETL pipeline: extract -> transform -> load.

    Parameters
    ----------
    data_path : str, optional
        Custom CSV input path. If None, default resolution is used.
    output_path : str, optional
        Custom Parquet output path. If None, default resolution is used.

    Returns
    -------
    Path
        Path to the Parquet output file.
    """
    logger.info(16*"===")
    logger.info("=== STARTING FRAUD DETECTION ETL PIPELINE ===")
    df = load_data(data_path=data_path)
    logger.info("Data after ETL has %d rows and %d columns", df.shape[0], df.shape[1])

    out_path = load(df, output_path=output_path)
    logger.info("=== ETL PIPELINE COMPLETED SUCCESSFULLY ===")
    logger.info(16*"===")
    return out_path


if __name__ == "__main__":
    # Allow running directly: python etl_pipeline.py
    try:
        run_etl()
    except Exception as e:
        logger.exception("ETL pipeline failed: %s", e)
        raise
