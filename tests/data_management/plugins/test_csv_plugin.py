import pytest
from pathlib import Path
import pandas as pd
from gaubongai.data_management.plugins.csv_plugin import CSVPlugin
from gaubongai.data_management.interfaces import DataCategory


def test_csv_plugin_initialization():
    """
    Test CSVPlugin initialization and class attributes.

    This test verifies:
    1. Plugin name is correct
    2. Supported extensions are correct
    3. Data category is correct
    """
    plugin = CSVPlugin()

    assert plugin.name == "csv_plugin"
    assert plugin.supported_extensions == [".csv"]
    assert plugin.data_category == DataCategory.TABULAR


def test_can_handle():
    """
    Test file handling capability detection.

    This test verifies:
    1. Plugin correctly identifies CSV files
    2. Plugin correctly rejects non-CSV files
    3. Case-insensitive extension handling
    """
    # Test with different path formats
    assert CSVPlugin.can_handle(Path("test.csv"))
    assert CSVPlugin.can_handle(Path("test.CSV"))
    assert CSVPlugin.can_handle(Path("/path/to/data.csv"))

    # Test with non-CSV files
    assert not CSVPlugin.can_handle(Path("test.txt"))
    assert not CSVPlugin.can_handle(Path("test"))
    assert not CSVPlugin.can_handle(Path("test.csv.txt"))


def test_load_csv(sample_csv_path):
    """
    Test loading CSV file.

    This test verifies:
    1. CSV data is loaded correctly
    2. Metadata is generated correctly
    3. DataInfo object is properly constructed
    """
    plugin = CSVPlugin()
    data_info = plugin.load(sample_csv_path)

    # Check data
    assert isinstance(data_info.data, pd.DataFrame)
    assert len(data_info.data) > 0
    assert list(data_info.data.columns) == ["id", "name", "age", "score", "active"]

    # Check metadata
    assert "rows" in data_info.metadata
    assert data_info.metadata["rows"] == len(data_info.data)
    assert "columns" in data_info.metadata
    assert data_info.metadata["columns"] == list(data_info.data.columns)
    assert "missing_values" in data_info.metadata
    assert "dtypes" in data_info.metadata

    # Check DataInfo attributes
    assert data_info.category == DataCategory.TABULAR
    assert data_info.source_path == sample_csv_path


def test_load_csv_with_options(tmp_path):
    """
    Test loading CSV with different options.

    This test verifies:
    1. Custom separator handling
    2. Custom encoding handling
    3. Custom NA values handling
    """
    # Create a test CSV with specific format
    file_path = tmp_path / "test_custom.csv"
    content = "col1;col2;col3\n1;NA;value1\n2;N/A;value2\n3;;value3"
    file_path.write_text(content, encoding="utf-8")

    plugin = CSVPlugin()

    # Test with custom options
    data_info = plugin.load(
        file_path, separator=";", encoding="utf-8", na_values=["NA", "N/A", ""]
    )

    # Check data loading
    assert len(data_info.data) == 3
    assert data_info.data["col2"].isna().sum() == 3  # All values should be NA


def test_load_csv_with_missing_values(tmp_path):
    """
    Test handling of missing values in CSV.

    This test verifies:
    1. Missing values are properly detected
    2. Missing value counts are correctly reported in metadata
    """
    # Create a test CSV with missing values
    file_path = tmp_path / "test_missing.csv"
    df = pd.DataFrame(
        {"A": [1, None, 3], "B": ["x", "y", None], "C": [None, None, None]}
    )
    df.to_csv(file_path, index=False)

    plugin = CSVPlugin()
    data_info = plugin.load(file_path)

    # Check missing value detection
    assert data_info.metadata["missing_values"]["A"] == 1
    assert data_info.metadata["missing_values"]["B"] == 1
    assert data_info.metadata["missing_values"]["C"] == 3


def test_load_csv_error_handling(tmp_path):
    """
    Test error handling during CSV loading.

    This test verifies:
    1. Appropriate error for non-existent file
    2. Appropriate error for invalid CSV format
    """
    plugin = CSVPlugin()

    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        plugin.load(tmp_path / "nonexistent.csv")

    # Test with invalid CSV (unmatched quotes will reliably cause a ParserError)
    invalid_file = tmp_path / "invalid.csv"
    invalid_file.write_text('a,b\n"unclosed quote,2\n4,5', encoding="utf-8")

    with pytest.raises(pd.errors.ParserError):
        plugin.load(invalid_file)
