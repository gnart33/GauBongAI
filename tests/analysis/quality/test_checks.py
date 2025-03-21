import pytest
import pandas as pd
from gaubongai.analysis.quality import CompletenessCheck, QualityCheckCategory


def test_completeness_check_initialization():
    """Test CompletenessCheck initialization and attributes."""
    check = CompletenessCheck()
    assert check.name == "completeness_check"
    assert check.category == QualityCheckCategory.COMPLETENESS
    assert isinstance(check.description, str)


def test_can_handle_valid_data(complete_data_info):
    """Test can_handle method with valid tabular data."""
    check = CompletenessCheck()
    assert check.can_handle(complete_data_info) is True


def test_can_handle_invalid_data(invalid_data_info):
    """Test can_handle method with invalid data."""
    check = CompletenessCheck()
    assert check.can_handle(invalid_data_info) is False


def test_check_complete_data(complete_data_info):
    """Test check method with complete data."""
    check = CompletenessCheck()
    result = check.check(complete_data_info)

    assert result.status is True  # Should pass as no missing values
    assert result.check_name == check.name
    assert result.category == check.category
    assert isinstance(result.summary, str)

    # Verify details
    assert result.details["total_missing"] == 0
    assert result.details["overall_completeness"] == 100.0
    assert len(result.details["column_stats"]) == len(complete_data_info.data.columns)

    # Check column stats
    for col_stats in result.details["column_stats"].values():
        assert col_stats["missing_count"] == 0
        assert col_stats["missing_percentage"] == 0.0
        assert "dtype" in col_stats


def test_check_incomplete_data(incomplete_data_info):
    """Test check method with incomplete data."""
    check = CompletenessCheck()
    result = check.check(incomplete_data_info)

    # With 30 missing values out of 500 cells (100 rows × 5 columns)
    # Overall completeness should be 94%
    assert result.status is False  # Below 95% threshold

    # Verify missing counts
    details = result.details
    assert details["total_missing"] == 30
    assert 93.9 < details["overall_completeness"] < 94.1

    # Check specific columns with known missing values
    col_stats = details["column_stats"]
    assert col_stats["name"]["missing_count"] == 10
    assert col_stats["value"]["missing_count"] == 10
    assert col_stats["category"]["missing_count"] == 10

    # Columns that should have no missing values
    assert col_stats["id"]["missing_count"] == 0
    assert col_stats["date"]["missing_count"] == 0


def test_check_invalid_data(invalid_data_info):
    """Test check method with invalid data."""
    check = CompletenessCheck()
    with pytest.raises(ValueError, match="Data must be tabular"):
        check.check(invalid_data_info)


def test_check_empty_dataframe():
    """Test check method with empty DataFrame."""
    empty_df = pd.DataFrame()
    data_info = DataInfo(
        data=empty_df, metadata={"source": "test"}, category=DataCategory.TABULAR
    )

    check = CompletenessCheck()
    result = check.check(data_info)

    assert result.status is True  # An empty DataFrame is technically complete
    assert result.details["total_cells"] == 0
    assert result.details["total_missing"] == 0
    assert result.details["total_columns"] == 0
    assert result.details["total_rows"] == 0
