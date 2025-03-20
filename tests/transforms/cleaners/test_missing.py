import pytest
import pandas as pd
import numpy as np
from gauai.transforms.cleaners import MissingValueFiller
from gauai.core.data_management.interfaces import DataInfo, DataCategory


@pytest.fixture
def sample_data_with_missing():
    """Create sample data with missing values."""
    return pd.DataFrame(
        {
            "numeric": [1.0, np.nan, 3.0, np.nan, 5.0],
            "categorical": ["A", None, "B", "A", None],
            "integer": [1, 2, np.nan, 4, 5],
        }
    )


@pytest.fixture
def data_info_with_missing(sample_data_with_missing):
    """Create DataInfo object with missing data."""
    return DataInfo(
        data=sample_data_with_missing,
        metadata={},
        category=DataCategory.TABULAR,
        source_path="test.csv",
    )


def test_missing_value_filler_initialization():
    """Test MissingValueFiller initialization with different strategies."""
    # Test valid strategy
    filler = MissingValueFiller(strategy="mean")
    assert filler.strategy == "mean"

    # Test invalid strategy
    with pytest.raises(ValueError):
        MissingValueFiller(strategy="invalid")


def test_can_transform(data_info_with_missing):
    """Test can_transform method."""
    filler = MissingValueFiller()

    # Test with valid data
    assert filler.can_transform(data_info_with_missing)

    # Test with invalid data type
    invalid_data = DataInfo(
        data="not a dataframe",
        metadata={},
        category=DataCategory.TABULAR,
        source_path="test.txt",
    )
    assert not filler.can_transform(invalid_data)


def test_mean_strategy(data_info_with_missing):
    """Test mean strategy for filling missing values."""
    filler = MissingValueFiller(strategy="mean")
    result = filler.transform(data_info_with_missing)

    # Check numeric column
    assert result.data["numeric"].isna().sum() == 0
    assert result.data["numeric"].mean() == 3.0

    # Check that non-numeric columns still have missing values
    assert result.data["categorical"].isna().sum() == 2


def test_median_strategy(data_info_with_missing):
    """Test median strategy for filling missing values."""
    filler = MissingValueFiller(strategy="median")
    result = filler.transform(data_info_with_missing)

    # Check numeric column
    assert result.data["numeric"].isna().sum() == 0
    assert result.data["numeric"].median() == 3.0


def test_mode_strategy(data_info_with_missing):
    """Test mode strategy for filling missing values."""
    filler = MissingValueFiller(strategy="mode")
    result = filler.transform(data_info_with_missing)

    # Check all columns have no missing values
    assert result.data.isna().sum().sum() == 0
    # Check categorical column mode
    assert result.data["categorical"].value_counts()["A"] >= 2


def test_drop_strategy(data_info_with_missing):
    """Test drop strategy for handling missing values."""
    filler = MissingValueFiller(strategy="drop")
    result = filler.transform(data_info_with_missing)

    # Check no missing values remain
    assert result.data.isna().sum().sum() == 0
    # Check reduced number of rows
    assert len(result.data) < len(data_info_with_missing.data)


def test_metadata_updates(data_info_with_missing):
    """Test that metadata is properly updated after transformation."""
    filler = MissingValueFiller(strategy="mean")
    result = filler.transform(data_info_with_missing)

    # Check transform was recorded in metadata
    assert "transforms" in result.metadata
    transform_record = result.metadata["transforms"][0]

    assert transform_record["name"] == "missing_value_filler"
    assert transform_record["type"] == "cleaner"
    assert transform_record["parameters"] == {"strategy": "mean"}
    assert "changes" in transform_record


def test_get_parameters():
    """Test get_parameters method."""
    filler = MissingValueFiller(strategy="median")
    params = filler.get_parameters()

    assert params == {"strategy": "median"}
