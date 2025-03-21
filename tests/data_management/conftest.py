"""Test fixtures for data management tests."""

import pytest
import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Any, List, Type

from gaubongai.data_management.interfaces import (
    DataInfo,
    DataCategory,
    DataPlugin,
    DataTransformation,
    Pipeline,
    PluginVariant,
)

# ============================================================================
# Data Sample Fixtures
# ============================================================================


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample pandas DataFrame for testing."""
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
def sample_polars_df() -> pl.DataFrame:
    """Create a sample polars DataFrame for testing."""
    return pl.DataFrame(
        {
            "id": range(1, 6),
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "score": [85.5, 90.0, 88.5, 92.0, 87.5],
            "active": [True, False, True, True, False],
        }
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


# ============================================================================
# File Fixtures
# ============================================================================


@pytest.fixture
def sample_csv_file(tmp_path, sample_dataframe) -> Path:
    """Create a sample CSV file for testing."""
    file_path = tmp_path / "test.csv"
    sample_dataframe.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def mock_file_path() -> Path:
    """Fixture providing a mock file path."""
    return Path("test.mock")


# ============================================================================
# DataInfo Fixtures
# ============================================================================


@pytest.fixture
def sample_data_info(sample_dataframe) -> DataInfo:
    """Create a sample DataInfo object with pandas DataFrame."""
    return DataInfo(
        data=sample_dataframe,
        metadata={
            "rows": len(sample_dataframe),
            "columns": list(sample_dataframe.columns),
            "dtypes": sample_dataframe.dtypes.astype(str).to_dict(),
            "implementation": "pandas",
        },
        category=DataCategory.TABULAR,
    )


@pytest.fixture
def sample_polars_data_info(sample_polars_df) -> DataInfo:
    """Create a sample DataInfo object with polars DataFrame."""
    return DataInfo(
        data=sample_polars_df,
        metadata={
            "rows": sample_polars_df.height,
            "columns": sample_polars_df.columns,
            "implementation": "polars",
            "memory_usage": sample_polars_df.estimated_size(),
        },
        category=DataCategory.TABULAR,
    )


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
    )


# ============================================================================
# Plugin Fixtures
# ============================================================================


class MockPlugin(DataPlugin):
    """Mock plugin for testing."""

    name = "mock"
    variant = PluginVariant.DEFAULT
    supported_extensions = [".mock"]
    data_category = DataCategory.TABULAR
    priority = 1

    @classmethod
    def can_handle(cls, file_path: Path) -> bool:
        return file_path.suffix in cls.supported_extensions

    def load(self, file_path: Path, **kwargs) -> DataInfo:
        data = {"test": [1, 2, 3]}
        metadata = {"rows": 3, "columns": ["test"], "implementation": "mock"}
        return DataInfo(data=data, metadata=metadata, category=self.data_category)


class MockMemoryEfficientPlugin(MockPlugin):
    """Memory-efficient variant of mock plugin."""

    variant = PluginVariant.MEMORY_EFFICIENT
    priority = 2

    def load(self, file_path: Path, **kwargs) -> DataInfo:
        data_info = super().load(file_path, **kwargs)
        data_info.metadata["implementation"] = "mock_memory_efficient"
        data_info.metadata["memory_usage"] = 100
        return data_info


class MockPerformancePlugin(MockPlugin):
    """High-performance variant of mock plugin."""

    variant = PluginVariant.PERFORMANCE
    priority = 3

    def load(self, file_path: Path, **kwargs) -> DataInfo:
        data_info = super().load(file_path, **kwargs)
        data_info.metadata["implementation"] = "mock_performance"
        data_info.metadata["processing_time"] = 0.001
        return data_info


@pytest.fixture
def mock_plugin() -> MockPlugin:
    """Fixture providing a mock plugin."""
    return MockPlugin()


@pytest.fixture
def mock_memory_efficient_plugin() -> MockMemoryEfficientPlugin:
    """Fixture providing a memory-efficient mock plugin."""
    return MockMemoryEfficientPlugin()


@pytest.fixture
def mock_performance_plugin() -> MockPerformancePlugin:
    """Fixture providing a high-performance mock plugin."""
    return MockPerformancePlugin()


@pytest.fixture
def mock_plugins() -> List[Type[DataPlugin]]:
    """Fixture providing a list of mock plugins with different variants."""
    return [MockPlugin, MockMemoryEfficientPlugin, MockPerformancePlugin]


# ============================================================================
# Pipeline Fixtures
# ============================================================================


class MockTransformation(DataTransformation):
    """Mock transformation for testing."""

    name = "mock_transform"
    supported_categories = [DataCategory.TABULAR]

    def can_transform(self, data_info: DataInfo) -> bool:
        return data_info.category in self.supported_categories

    def transform(self, data_info: DataInfo) -> DataInfo:
        return data_info


class MockPipeline(Pipeline):
    """Mock pipeline for testing."""

    name = "mock_pipeline"
    steps = ["mock_transform"]

    def execute(self, data_info: DataInfo) -> DataInfo:
        return data_info


@pytest.fixture
def mock_transformation() -> MockTransformation:
    """Fixture providing a mock transformation."""
    return MockTransformation()


@pytest.fixture
def mock_pipeline() -> MockPipeline:
    """Fixture providing a mock pipeline."""
    return MockPipeline()


# ============================================================================
# Heart Attack Dataset Fixtures
# ============================================================================


@pytest.fixture
def heart_attack_file() -> Path:
    """Path to the heart attack dataset."""
    return Path("examples/heart_attack/heart_attack_dataset.csv")


@pytest.fixture
def sample_heart_data() -> pd.DataFrame:
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
def heart_data_metadata() -> Dict[str, Any]:
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
