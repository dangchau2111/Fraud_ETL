# tests/test_etl_pipeline.py

import sys
import os

# Ensure project root is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import pandas.api.types as ptypes

from etl_pipeline import load_data, get_data_path


def test_fraud_data_row_count():
    """
    Test whether the cleaned fraud dataset has the expected number of rows.

    The load_data() function performs:
        - CSV extraction
        - Cleaning (drop NaN + drop duplicates)
        - Logging data info (before/after)

    Expected row count is determined from the known creditcard.csv dataset.
    """
    df = load_data()  # Execute ETL extraction + cleaning

    row_count = len(df)
    expected_rows = 283726

    assert (
        row_count == expected_rows
    ), f"Row count mismatch: expected {expected_rows}, got {row_count}"


def test_schema_column_count():
    """
    Test that the cleaned dataset has the expected number of columns.
    The original creditcard.csv dataset has 31 columns.
    """
    df = load_data()
    expected_columns = 31
    assert (
        df.shape[1] == expected_columns
    ), f"Column count mismatch: expected {expected_columns}, got {df.shape[1]}"


def test_required_columns_exist():
    """
    Test that required business-critical columns are present after ETL.
    """
    df = load_data()
    required_columns = ["Time", "Amount", "Class"]

    for col in required_columns:
        assert col in df.columns, f"Missing required column after ETL: {col}"


def test_no_nan_after_cleaning():
    """
    Test that there are no NaN values remaining in the cleaned dataset.
    """
    df = load_data()
    total_nan = df.isna().sum().sum()
    assert total_nan == 0, f"Dataset still contains NaN values after cleaning: {total_nan}"


def test_no_duplicates():
    """
    Test that no duplicate rows remain after ETL.
    """
    df = load_data()
    duplicate_count = df.duplicated().sum()
    assert duplicate_count == 0, f"Dataset still contains duplicate rows: {duplicate_count}"


def test_column_dtypes_are_correct():
    """
    Test that key columns have the expected data types.
    - Time and Amount should be numeric
    - Class should be integer-like (binary label)
    """
    df = load_data()

    assert "Time" in df.columns, "Time column is missing from the dataset"
    assert "Amount" in df.columns, "Amount column is missing from the dataset"
    assert "Class" in df.columns, "Class column is missing from the dataset"

    assert ptypes.is_numeric_dtype(df["Time"]), "Time column must be numeric"
    assert ptypes.is_numeric_dtype(df["Amount"]), "Amount column must be numeric"
    assert ptypes.is_integer_dtype(df["Class"]), "Class column must be integer"


def test_class_values_are_binary():
    """
    Test that the Class column only contains binary values {0, 1}.
    """
    df = load_data()
    unique_classes = set(df["Class"].unique())
    expected_classes = {0, 1}

    assert (
        unique_classes == expected_classes
    ), f"Class column contains unexpected values: {unique_classes}"


def test_non_negative_time_and_amount():
    """
    Test that Time and Amount columns do not contain negative values.
    """
    df = load_data()

    assert (df["Time"] >= 0).all(), "Time column contains negative values"
    assert (df["Amount"] >= 0).all(), "Amount column contains negative values"


def test_etl_does_not_drop_columns():
    """
    Test that the ETL process does not drop or add columns compared to the raw CSV.
    Only row-level cleaning (NaN/duplicates) should be applied.
    """
    raw_path = get_data_path()
    df_raw = pd.read_csv(raw_path)
    df_clean = load_data()

    raw_columns = set(df_raw.columns)
    clean_columns = set(df_clean.columns)

    assert (
        raw_columns == clean_columns
    ), f"Column set changed during ETL. Raw: {raw_columns}, Clean: {clean_columns}"
