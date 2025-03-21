import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from gaubongai.data_management.interfaces import DataInfo, DataCategory


@pytest.fixture
def sample_complete_df():
    """Create a complete DataFrame without missing values."""
    return pd.DataFrame(
        {
            "id": range(1, 101),
            "name": [f"Item {i}" for i in range(1, 101)],
            "value": np.random.rand(100),
            "category": np.random.choice(["A", "B", "C"], 100),
            "date": pd.date_range(start="2024-01-01", periods=100),
        }
    )


@pytest.fixture
def sample_incomplete_df():
    """Create a DataFrame with missing values."""
    df = pd.DataFrame(
        {
            "id": range(1, 101),
            "name": [f"Item {i}" for i in range(1, 101)],
            "value": np.random.rand(100),
            "category": np.random.choice(["A", "B", "C"], 100),
            "date": pd.date_range(start="2024-01-01", periods=100),
        }
    )

    # Introduce missing values
    df.loc[0:9, "name"] = None  # 10% missing in name
    df.loc[20:29, "value"] = None  # 10% missing in value
    df.loc[40:49, "category"] = None  # 10% missing in category

    return df


@pytest.fixture
def complete_data_info(sample_complete_df):
    """Create DataInfo object with complete data."""
    return DataInfo(
        data=sample_complete_df,
        metadata={"source": "test", "created_at": datetime.now().isoformat()},
        category=DataCategory.TABULAR,
    )


@pytest.fixture
def incomplete_data_info(sample_incomplete_df):
    """Create DataInfo object with incomplete data."""
    return DataInfo(
        data=sample_incomplete_df,
        metadata={"source": "test", "created_at": datetime.now().isoformat()},
        category=DataCategory.TABULAR,
    )


@pytest.fixture
def invalid_data_info():
    """Create DataInfo object with invalid data type."""
    return DataInfo(
        data="This is not a DataFrame",
        metadata={"source": "test"},
        category=DataCategory.TEXT,
    )
