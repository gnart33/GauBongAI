import pytest
import pandas as pd
import numpy as np
from gauai.transforms.cleaners import DataTypeCleaner
from gauai.core.data_management.interfaces import DataInfo, DataCategory


@pytest.fixture
def sample_mixed_data():
    """Create sample data with mixed types."""
    return pd.DataFrame(
        {
            "integers": ["1", "2", "3", "4", "5"],
            "floats": ["1.1", "2.2", "3.3", "4.4", "5.5"],
            "booleans": ["true", "false", "True", "0", "1"],
            "dates": [
                "2021-01-01",
                "2021-02-02",
                "2021-03-03",
                "2021-04-04",
                "2021-05-05",
            ],
            "categories": ["A", "B", "A", "B", "A"],
            "mixed": ["1", "2.2", "three", "4", "5.5"],
        }
    )


@pytest.fixture
def data_info_mixed(sample_mixed_data):
    """Create DataInfo object with mixed type data."""
    return DataInfo(
        data=sample_mixed_data,
        metadata={},
        category=DataCategory.TABULAR,
        source_path="test.csv",
    )


def test_data_type_cleaner_initialization():
    """Test DataTypeCleaner initialization."""
    # Test with valid type map
    cleaner = DataTypeCleaner(type_map={"col1": "int", "col2": "float"})
    assert cleaner.type_map == {"col1": "int", "col2": "float"}

    # Test with invalid type
    with pytest.raises(ValueError):
        DataTypeCleaner(type_map={"col1": "invalid_type"})


def test_can_transform(data_info_mixed):
    """Test can_transform method."""
    cleaner = DataTypeCleaner()

    # Test with valid data
    assert cleaner.can_transform(data_info_mixed)

    # Test with invalid data type
    invalid_data = DataInfo(
        data="not a dataframe",
        metadata={},
        category=DataCategory.TABULAR,
        source_path="test.txt",
    )
    assert not cleaner.can_transform(invalid_data)


def test_infer_types(data_info_mixed):
    """Test type inference."""
    cleaner = DataTypeCleaner()
    result = cleaner.transform(data_info_mixed)

    # Check inferred types
    assert pd.api.types.is_integer_dtype(result.data["integers"].dtype)
    assert pd.api.types.is_float_dtype(result.data["floats"].dtype)
    assert pd.api.types.is_bool_dtype(result.data["booleans"].dtype)
    assert pd.api.types.is_datetime64_any_dtype(result.data["dates"].dtype)
    assert pd.api.types.is_categorical_dtype(result.data["categories"].dtype)
    assert pd.api.types.is_string_dtype(result.data["mixed"].dtype)


def test_explicit_type_conversion():
    """Test explicit type conversion with type_map."""
    data = pd.DataFrame({"col1": ["1", "2", "3"], "col2": ["1.1", "2.2", "3.3"]})
    data_info = DataInfo(
        data=data, metadata={}, category=DataCategory.TABULAR, source_path="test.csv"
    )

    cleaner = DataTypeCleaner(type_map={"col1": "int", "col2": "float"})
    result = cleaner.transform(data_info)

    assert pd.api.types.is_integer_dtype(result.data["col1"].dtype)
    assert pd.api.types.is_float_dtype(result.data["col2"].dtype)


def test_boolean_conversion(data_info_mixed):
    """Test boolean conversion with different formats."""
    cleaner = DataTypeCleaner(type_map={"booleans": "bool"})
    result = cleaner.transform(data_info_mixed)

    assert pd.api.types.is_bool_dtype(result.data["booleans"].dtype)
    assert result.data["booleans"].tolist() == [True, False, True, False, True]


def test_datetime_conversion(data_info_mixed):
    """Test datetime conversion."""
    cleaner = DataTypeCleaner(type_map={"dates": "datetime"})
    result = cleaner.transform(data_info_mixed)

    assert pd.api.types.is_datetime64_any_dtype(result.data["dates"].dtype)
    assert result.data["dates"].dt.year.tolist() == [2021] * 5


def test_category_conversion(data_info_mixed):
    """Test category conversion."""
    cleaner = DataTypeCleaner(type_map={"categories": "category"})
    result = cleaner.transform(data_info_mixed)

    assert pd.api.types.is_categorical_dtype(result.data["categories"].dtype)
    assert set(result.data["categories"].cat.categories) == {"A", "B"}


def test_metadata_updates(data_info_mixed):
    """Test that metadata is properly updated after transformation."""
    cleaner = DataTypeCleaner(type_map={"integers": "int"})
    result = cleaner.transform(data_info_mixed)

    # Check transform was recorded in metadata
    assert "transforms" in result.metadata
    transform_record = result.metadata["transforms"][0]

    assert transform_record["name"] == "data_type_cleaner"
    assert transform_record["type"] == "cleaner"
    assert transform_record["parameters"] == {"type_map": {"integers": "int"}}
    assert "changes" in transform_record


def test_get_parameters():
    """Test get_parameters method."""
    type_map = {"col1": "int", "col2": "float"}
    cleaner = DataTypeCleaner(type_map=type_map)
    params = cleaner.get_parameters()

    assert params == {"type_map": type_map}
