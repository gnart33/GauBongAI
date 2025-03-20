import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from gauai.core.data_management.data_ingestion import DataIngestionManager
import tempfile
import os


def test_init():
    """Test DataIngestionManager initialization."""
    manager = DataIngestionManager()
    assert manager.datasets == {}
    assert manager.metadata == {}


def test_load_csv(heart_attack_file, sample_heart_data):
    """Test loading heart attack CSV file."""
    manager = DataIngestionManager()
    df = manager.load_csv(heart_attack_file, usecols=sample_heart_data.columns)

    # Check if first few rows match our sample data
    pd.testing.assert_frame_equal(df.head(4).reset_index(drop=True), sample_heart_data)

    # Check if dataset was stored with correct name
    expected_name = Path(heart_attack_file).stem
    assert expected_name in manager.datasets


def test_load_csv_with_custom_name(heart_attack_file, sample_heart_data):
    """Test loading heart attack CSV file with custom dataset name."""
    manager = DataIngestionManager()
    custom_name = "heart_data"
    df = manager.load_csv(
        heart_attack_file, name=custom_name, usecols=sample_heart_data.columns
    )

    # Check if DataFrame was stored with custom name
    assert custom_name in manager.datasets
    pd.testing.assert_frame_equal(df.head(4).reset_index(drop=True), sample_heart_data)


def test_extract_metadata(sample_heart_data, heart_data_metadata):
    """Test metadata extraction for heart attack data."""
    manager = DataIngestionManager()
    dataset_name = "heart_data"

    # Load dataset and check metadata
    manager.datasets[dataset_name] = sample_heart_data
    manager._extract_metadata(dataset_name, sample_heart_data)

    assert dataset_name in manager.metadata
    metadata = manager.metadata[dataset_name]

    assert metadata["shape"] == heart_data_metadata["shape"]
    assert metadata["columns"] == heart_data_metadata["columns"]
    assert metadata["dtypes"] == heart_data_metadata["dtypes"]
    assert metadata["missing_values"] == heart_data_metadata["missing_values"]


def test_get_dataset(sample_heart_data):
    """Test retrieving heart attack dataset."""
    manager = DataIngestionManager()
    dataset_name = "heart_data"

    # Store dataset and retrieve it
    manager.datasets[dataset_name] = sample_heart_data
    retrieved_df = manager.get_dataset(dataset_name)

    pd.testing.assert_frame_equal(retrieved_df, sample_heart_data)


def test_get_nonexistent_dataset():
    """Test retrieving non-existent dataset."""
    manager = DataIngestionManager()
    assert manager.get_dataset("nonexistent") is None


def test_get_metadata(sample_heart_data, heart_data_metadata):
    """Test retrieving metadata for heart attack data."""
    manager = DataIngestionManager()
    dataset_name = "heart_data"

    # Store dataset and metadata
    manager.datasets[dataset_name] = sample_heart_data
    manager.metadata[dataset_name] = heart_data_metadata

    retrieved_metadata = manager.get_metadata(dataset_name)
    assert retrieved_metadata == heart_data_metadata


def test_list_datasets(sample_heart_data):
    """Test listing available datasets."""
    manager = DataIngestionManager()
    dataset_names = ["heart_data_train", "heart_data_test", "heart_data_val"]

    # Store multiple datasets
    for name in dataset_names:
        manager.datasets[name] = sample_heart_data

    available_datasets = manager.list_datasets()
    assert set(available_datasets) == set(dataset_names)


def test_load_csv_with_missing_values():
    """Test loading CSV with missing values."""
    manager = DataIngestionManager()

    # Create temporary CSV with missing values
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        f.write("Age,Gender,Cholesterol,BloodPressure,HeartRate,Outcome\n")
        f.write("31,Male,,162,71,No Heart Attack\n")
        f.write("69,,208,148,,No Heart Attack\n")
        f.write("34,Female,132,,94,Heart Attack\n")
        temp_path = f.name

    try:
        df = manager.load_csv(temp_path)
        metadata = manager.get_metadata(Path(temp_path).stem)

        # Check if missing values are correctly identified
        assert metadata["missing_values"]["Cholesterol"] == 1
        assert metadata["missing_values"]["Gender"] == 1
        assert metadata["missing_values"]["HeartRate"] == 1
        assert metadata["missing_values"]["BloodPressure"] == 1
    finally:
        os.unlink(temp_path)


def test_load_csv_with_custom_params(heart_attack_file):
    """Test loading heart attack CSV with custom parameters."""
    manager = DataIngestionManager()

    # Test with custom parameters
    df = manager.load_csv(
        heart_attack_file,
        usecols=["Age", "Gender"],  # Only load specific columns
        nrows=10,  # Only load first 10 rows
    )

    assert list(df.columns) == ["Age", "Gender"]
    assert len(df) == 10
