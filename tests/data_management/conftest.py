import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os


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
