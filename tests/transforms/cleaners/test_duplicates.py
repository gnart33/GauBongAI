import pytest
import pandas as pd
import numpy as np
from gauai.transforms.cleaners import DuplicateCleaner
from gauai.core.data_management.interfaces import DataInfo, DataCategory


@pytest.fixture
def sample_data_with_duplicates():
    """Create sample data with duplicates."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 1, 2, 4],
            "value": [10, 20, 30, 10, 25, 40],
            "category": ["A", "B", "C", "A", "B", "D"],
        }
    )


@pytest.fixture
def data_info_with_duplicates(sample_data_with_duplicates):
    """Create DataInfo object with duplicate data."""
    return DataInfo(
        data=sample_data_with_duplicates,
        metadata={},
        category=DataCategory.TABULAR,
        source_path="test.csv",
    )


def test_duplicate_cleaner_initialization():
    """Test DuplicateCleaner initialization."""
    # Test valid initialization
    cleaner = DuplicateCleaner(subset=["id"], keep="first")
    assert cleaner.subset == ["id"]
    assert cleaner.keep == "first"

    # Test invalid keep option
    with pytest.raises(ValueError):
        DuplicateCleaner(keep="invalid")


def test_can_transform(data_info_with_duplicates):
    """Test can_transform method."""
    cleaner = DuplicateCleaner()

    # Test with valid data
    assert cleaner.can_transform(data_info_with_duplicates)

    # Test with invalid data type
    invalid_data = DataInfo(
        data="not a dataframe",
        metadata={},
        category=DataCategory.TABULAR,
        source_path="test.txt",
    )
    assert not cleaner.can_transform(invalid_data)


def test_keep_first(data_info_with_duplicates):
    """Test keeping first occurrence of duplicates."""
    cleaner = DuplicateCleaner(subset=["id"], keep="first")
    result = cleaner.transform(data_info_with_duplicates)

    # Check number of rows
    assert len(result.data) == 4  # Should have 4 unique IDs

    # Check that first occurrences were kept
    assert result.data.loc[result.data["id"] == 1, "value"].iloc[0] == 10
    assert result.data.loc[result.data["id"] == 2, "value"].iloc[0] == 20


def test_keep_last(data_info_with_duplicates):
    """Test keeping last occurrence of duplicates."""
    cleaner = DuplicateCleaner(subset=["id"], keep="last")
    result = cleaner.transform(data_info_with_duplicates)

    # Check number of rows
    assert len(result.data) == 4  # Should have 4 unique IDs

    # Check that last occurrences were kept
    assert result.data.loc[result.data["id"] == 1, "value"].iloc[0] == 10
    assert result.data.loc[result.data["id"] == 2, "value"].iloc[0] == 25


def test_drop_all_duplicates(data_info_with_duplicates):
    """Test dropping all duplicates."""
    cleaner = DuplicateCleaner(subset=["id"], keep=False)
    result = cleaner.transform(data_info_with_duplicates)

    # Check that only unique rows remain
    assert len(result.data) == 2  # Only IDs 3 and 4 have no duplicates


def test_multiple_column_subset(data_info_with_duplicates):
    """Test deduplication based on multiple columns."""
    cleaner = DuplicateCleaner(subset=["id", "value"])
    result = cleaner.transform(data_info_with_duplicates)

    # Check results
    assert len(result.data) == 5  # One pair of (id, value) is duplicate


def test_no_duplicates():
    """Test behavior when there are no duplicates."""
    data = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
    data_info = DataInfo(
        data=data, metadata={}, category=DataCategory.TABULAR, source_path="test.csv"
    )

    cleaner = DuplicateCleaner()
    result = cleaner.transform(data_info)

    # Check that data remains unchanged
    assert len(result.data) == len(data)
    assert result.metadata == data_info.metadata


def test_ignore_index(data_info_with_duplicates):
    """Test index handling options."""
    # Test with ignore_index=True
    cleaner_ignore = DuplicateCleaner(subset=["id"], ignore_index=True)
    result_ignore = cleaner_ignore.transform(data_info_with_duplicates)
    assert result_ignore.data.index.equals(pd.RangeIndex(len(result_ignore.data)))

    # Test with ignore_index=False
    cleaner_keep = DuplicateCleaner(subset=["id"], ignore_index=False)
    result_keep = cleaner_keep.transform(data_info_with_duplicates)
    assert not result_keep.data.index.equals(pd.RangeIndex(len(result_keep.data)))


def test_metadata_updates(data_info_with_duplicates):
    """Test that metadata is properly updated after transformation."""
    cleaner = DuplicateCleaner(subset=["id"])
    result = cleaner.transform(data_info_with_duplicates)

    # Check transform was recorded in metadata
    assert "transforms" in result.metadata
    transform_record = result.metadata["transforms"][0]

    assert transform_record["name"] == "duplicate_cleaner"
    assert transform_record["type"] == "cleaner"
    assert transform_record["parameters"] == {
        "subset": ["id"],
        "keep": "first",
        "ignore_index": True,
    }

    # Check changes were recorded
    changes = transform_record["changes"]
    assert changes["rows_before"] == 6
    assert changes["rows_after"] == 4
    assert changes["total_duplicates"] == 2


def test_get_parameters():
    """Test get_parameters method."""
    params = {"subset": ["col1", "col2"], "keep": "last", "ignore_index": False}
    cleaner = DuplicateCleaner(**params)
    assert cleaner.get_parameters() == params


def test_duplicate_groups_stats(data_info_with_duplicates):
    """Test duplicate group statistics in metadata."""
    cleaner = DuplicateCleaner(subset=["id"])
    result = cleaner.transform(data_info_with_duplicates)

    changes = result.metadata["transforms"][0]["changes"]

    assert changes["unique_duplicate_groups"] == 2  # IDs 1 and 2 have duplicates
    assert changes["max_duplicates_in_group"] == 2  # Each duplicate ID appears twice
    assert isinstance(changes["duplicate_group_sizes"], dict)
