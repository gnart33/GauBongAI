import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from gaubongai.data_management.interfaces import DataInfo, DataCategory


@pytest.fixture
def heart_attack_file():
    """Path to the heart attack dataset."""
    return "examples/heart_attack/heart_attack_dataset.csv"


@pytest.fixture
def sample_heart_data():
    """Create a sample DataFrame with first few rows of heart attack data."""
    return pd.DataFrame(
        {
            "Age": [31, 69, 34, 53],
            "Gender": ["Male", "Male", "Female", "Male"],
            "Cholesterol": [194, 208, 132, 268],
            "BloodPressure": [162, 148, 161, 134],
            "HeartRate": [71, 93, 94, 91],
            "Outcome": [
                "No Heart Attack",
                "No Heart Attack",
                "Heart Attack",
                "No Heart Attack",
            ],
        }
    )


@pytest.fixture
def heart_data_metadata():
    """Create sample metadata dictionary for heart attack data."""
    return {
        "shape": (4, 6),
        "columns": [
            "Age",
            "Gender",
            "Cholesterol",
            "BloodPressure",
            "HeartRate",
            "Outcome",
        ],
        "dtypes": {
            "Age": "int64",
            "Gender": "object",
            "Cholesterol": "int64",
            "BloodPressure": "int64",
            "HeartRate": "int64",
            "Outcome": "object",
        },
        "missing_values": {
            "Age": 0,
            "Gender": 0,
            "Cholesterol": 0,
            "BloodPressure": 0,
            "HeartRate": 0,
            "Outcome": 0,
        },
    }


@pytest.fixture
def sample_heart_notes():
    """Create sample analysis notes for heart attack data."""
    return [
        {
            "content": "High cholesterol levels observed in male patients",
            "category": "observation",
        },
        {
            "content": "Check correlation between blood pressure and heart rate",
            "category": "todo",
        },
        {"content": "Potential outlier in cholesterol values", "category": "warning"},
    ]


@pytest.fixture
def sample_csv_path(tmp_path) -> Path:
    """Create a sample CSV file for testing."""
    file_path = tmp_path / "test.csv"
    df = pd.DataFrame(
        {
            "id": range(1, 6),
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "score": [85.5, 90.0, 88.5, 92.0, 87.5],
            "active": [True, False, True, True, False],
        }
    )
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "id": range(1, 6),
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "score": [85.5, 90.0, 88.5, 92.0, 87.5],
            "active": [True, False, True, True, False],
        }
    )


@pytest.fixture
def sample_data_info(sample_dataframe) -> DataInfo:
    """Create a sample DataInfo object."""
    return DataInfo(
        data=sample_dataframe,
        metadata={
            "rows": len(sample_dataframe),
            "columns": list(sample_dataframe.columns),
            "dtypes": sample_dataframe.dtypes.astype(str).to_dict(),
        },
        category=DataCategory.TABULAR,
        source_path=Path("test.csv"),
    )


@pytest.fixture
def sample_text_data() -> str:
    """Create sample text data for testing."""
    return """
    This is a sample text with some special characters: @#$%
    It has multiple lines and extra   spaces.
    Some Numbers: 123, 456
    And some Mixed CASE words.
    """


@pytest.fixture
def sample_text_data_info(sample_text_data) -> DataInfo:
    """Create a sample DataInfo object for text data."""
    return DataInfo(
        data=sample_text_data,
        metadata={
            "type": "text",
            "length": len(sample_text_data),
            "lines": len(sample_text_data.splitlines()),
        },
        category=DataCategory.TEXT,
        source_path=Path("test.txt"),
    )


@pytest.fixture
def sample_mixed_data() -> pd.DataFrame:
    """Create a sample DataFrame with mixed data types and quality issues."""
    return pd.DataFrame(
        {
            "id": [1, 2, np.nan, 4, 5],
            "name": ["Alice", None, "Charlie", "David", "Eve"],
            "date": ["2023-01-01", "2023-02-02", "invalid", "2023-04-04", None],
            "value": ["1.5", "2.0", "three", "4.0", "5.5"],
            "category": ["A", "B", "A", "B", "C"],
        }
    )


@pytest.fixture
def sample_mixed_data_info(sample_mixed_data) -> DataInfo:
    """Create a sample DataInfo object with mixed data types."""
    return DataInfo(
        data=sample_mixed_data,
        metadata={
            "rows": len(sample_mixed_data),
            "columns": list(sample_mixed_data.columns),
            "dtypes": sample_mixed_data.dtypes.astype(str).to_dict(),
            "missing_values": sample_mixed_data.isna().sum().to_dict(),
        },
        category=DataCategory.MIXED,
        source_path=Path("mixed_data.csv"),
    )
