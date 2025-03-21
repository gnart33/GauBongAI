import pytest
import pandas as pd
from gaubongai.data_management.transforms.cleaner import TextCleaner
from gaubongai.data_management.interfaces import DataInfo, DataCategory


def test_text_cleaner_initialization():
    """Test TextCleaner initialization."""
    # Test with default parameters
    cleaner = TextCleaner()
    assert cleaner.columns is None

    # Test with specific columns
    columns = ["col1", "col2"]
    cleaner = TextCleaner(columns=columns)
    assert cleaner.columns == columns


def test_can_transform_method(sample_data_info, sample_text_data_info):
    """Test can_transform method with different data types."""
    cleaner = TextCleaner()

    # Should handle DataFrame
    assert cleaner.can_transform(sample_data_info)

    # Should handle text data
    assert cleaner.can_transform(sample_text_data_info)

    # Should not handle unsupported types
    invalid_data = DataInfo(
        data=42, metadata={}, category=DataCategory.TABULAR  # Number is not supported
    )
    assert not cleaner.can_transform(invalid_data)


def test_clean_text_basic():
    """Test basic text cleaning operations."""
    cleaner = TextCleaner()

    # Test various text cleaning scenarios
    assert cleaner._clean_text("Hello, World!") == "hello world"
    assert cleaner._clean_text("  Extra  Spaces  ") == "extra spaces"
    assert cleaner._clean_text("Special@#$Characters") == "special characters"
    assert cleaner._clean_text("MIXED case") == "mixed case"


def test_transform_string_data(sample_text_data_info):
    """Test transforming string data."""
    cleaner = TextCleaner()
    result = cleaner.transform(sample_text_data_info)

    # Check cleaned text
    cleaned_text = result.data
    assert isinstance(cleaned_text, str)
    assert cleaned_text.islower()  # Should be lowercase
    assert "@#$%" not in cleaned_text  # Special characters should be removed
    assert "  " not in cleaned_text  # Extra spaces should be removed

    # Check metadata
    assert "cleaning_history" in result.metadata
    assert len(result.metadata["cleaning_history"]) == 1
    assert result.metadata["cleaning_history"][0]["processor"] == "text_cleaner"


def test_transform_dataframe(sample_data_info):
    """Test transforming DataFrame data."""
    cleaner = TextCleaner(columns=["name"])
    result = cleaner.transform(sample_data_info)

    # Check cleaned DataFrame
    assert isinstance(result.data, pd.DataFrame)
    assert all(result.data["name"].str.islower())  # Names should be lowercase

    # Non-specified columns should remain unchanged
    pd.testing.assert_series_equal(sample_data_info.data["age"], result.data["age"])

    # Check metadata
    assert "cleaning_history" in result.metadata
    history = result.metadata["cleaning_history"][0]
    assert history["processor"] == "text_cleaner"
    assert history["columns_processed"] == ["name"]


@pytest.mark.xfail(reason="TODO: Fix numeric string handling in mixed data cleaning")
def test_transform_mixed_data(sample_mixed_data_info):
    """Test transforming data with mixed types and missing values."""
    cleaner = TextCleaner()
    result = cleaner.transform(sample_mixed_data_info)

    # Check that string columns were cleaned
    assert all(result.data["name"].dropna().str.islower())
    assert all(result.data["category"].str.islower())

    # Check that missing values were preserved
    assert (
        result.data["name"].isna().sum()
        == sample_mixed_data_info.data["name"].isna().sum()
    )

    # Check that non-string columns were converted and cleaned
    assert all(result.data["value"].str.islower())
    assert not any(result.data["value"].str.contains(r"[^\w\s]"))


def test_transform_specific_columns(sample_mixed_data_info):
    """Test transforming only specified columns."""
    cleaner = TextCleaner(columns=["category"])
    result = cleaner.transform(sample_mixed_data_info)

    # Check that only specified column was cleaned
    assert all(result.data["category"].str.islower())

    # Check that other columns remain unchanged
    pd.testing.assert_series_equal(
        sample_mixed_data_info.data["name"], result.data["name"]
    )
    pd.testing.assert_series_equal(
        sample_mixed_data_info.data["value"], result.data["value"]
    )


@pytest.mark.xfail(
    reason="TODO: Fix error handling for DataFrame with no string columns"
)
def test_transform_error_handling():
    """Test error handling during transformation."""
    cleaner = TextCleaner()

    # Test with invalid data type
    invalid_data = DataInfo(data=42, metadata={}, category=DataCategory.TABULAR)
    with pytest.raises(ValueError):
        cleaner.transform(invalid_data)

    # Test with non-existent columns
    df = pd.DataFrame({"a": [1, 2, 3]})
    data_info = DataInfo(data=df, metadata={}, category=DataCategory.TABULAR)
    cleaner = TextCleaner(columns=["non_existent"])
    result = cleaner.transform(data_info)
    # Should not modify DataFrame when columns don't exist
    pd.testing.assert_frame_equal(df, result.data)
