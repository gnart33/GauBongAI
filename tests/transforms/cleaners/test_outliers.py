import pytest
import pandas as pd
import numpy as np
from gauai.transforms.cleaners import OutlierCleaner
from gauai.core.data_management.interfaces import DataInfo, DataCategory


@pytest.fixture
def sample_data_with_outliers():
    """Create sample data with outliers."""
    np.random.seed(42)
    normal_data = np.random.normal(loc=0, scale=1, size=95)
    outliers = np.array([10, 12, -8, -10, 15])  # Add some obvious outliers
    data = np.concatenate([normal_data, outliers])
    return pd.DataFrame(
        {
            "values": data,
            "normal": normal_data,
            "categorical": ["A"] * 90 + ["B"] * 10,
            "other_numeric": np.random.normal(loc=5, scale=2, size=100),
        }
    )


@pytest.fixture
def data_info_with_outliers(sample_data_with_outliers):
    """Create DataInfo object with outlier data."""
    return DataInfo(
        data=sample_data_with_outliers,
        metadata={},
        category=DataCategory.TABULAR,
        source_path="test.csv",
    )


def test_outlier_cleaner_initialization():
    """Test OutlierCleaner initialization."""
    # Test valid initialization
    cleaner = OutlierCleaner(method="zscore", threshold=3.0)
    assert cleaner.method == "zscore"
    assert cleaner.threshold == 3.0

    # Test invalid method
    with pytest.raises(ValueError):
        OutlierCleaner(method="invalid")

    # Test invalid strategy
    with pytest.raises(ValueError):
        OutlierCleaner(strategy="invalid")

    # Test invalid threshold
    with pytest.raises(ValueError):
        OutlierCleaner(threshold=-1)


def test_can_transform(data_info_with_outliers):
    """Test can_transform method."""
    cleaner = OutlierCleaner()

    # Test with valid data
    assert cleaner.can_transform(data_info_with_outliers)

    # Test with invalid data type
    invalid_data = DataInfo(
        data="not a dataframe",
        metadata={},
        category=DataCategory.TABULAR,
        source_path="test.txt",
    )
    assert not cleaner.can_transform(invalid_data)


def test_zscore_detection(data_info_with_outliers):
    """Test outlier detection using z-score method."""
    cleaner = OutlierCleaner(method="zscore", threshold=3.0)
    result = cleaner.transform(data_info_with_outliers)

    # Check that extreme values were handled
    assert result.data["values"].max() < 10
    assert result.data["values"].min() > -8

    # Check metadata
    changes = result.metadata["transforms"][0]["changes"]
    assert "values" in changes
    assert changes["values"]["outliers_detected"] > 0


def test_iqr_detection(data_info_with_outliers):
    """Test outlier detection using IQR method."""
    cleaner = OutlierCleaner(method="iqr", threshold=1.5)
    result = cleaner.transform(data_info_with_outliers)

    # Check that extreme values were handled
    assert result.data["values"].max() < 10
    assert result.data["values"].min() > -8

    # Check metadata
    changes = result.metadata["transforms"][0]["changes"]
    assert "values" in changes
    assert changes["values"]["outliers_detected"] > 0


@pytest.mark.skipif(ImportError, reason="scikit-learn not installed")
def test_isolation_forest_detection(data_info_with_outliers):
    """Test outlier detection using Isolation Forest method."""
    try:
        cleaner = OutlierCleaner(method="isolation_forest")
        result = cleaner.transform(data_info_with_outliers)

        # Check metadata
        changes = result.metadata["transforms"][0]["changes"]
        assert "values" in changes
        assert changes["values"]["outliers_detected"] > 0
    except ImportError:
        pytest.skip("scikit-learn not installed")


def test_clip_strategy(data_info_with_outliers):
    """Test clip strategy for handling outliers."""
    cleaner = OutlierCleaner(method="zscore", strategy="clip")
    result = cleaner.transform(data_info_with_outliers)

    original_range = data_info_with_outliers.data["values"].agg(["min", "max"])
    new_range = result.data["values"].agg(["min", "max"])

    # Check that range was reduced
    assert new_range["min"] > original_range["min"]
    assert new_range["max"] < original_range["max"]


def test_remove_strategy(data_info_with_outliers):
    """Test remove strategy for handling outliers."""
    cleaner = OutlierCleaner(method="zscore", strategy="remove")
    result = cleaner.transform(data_info_with_outliers)

    # Check that rows were removed
    assert len(result.data) < len(data_info_with_outliers.data)


def test_winsorize_strategy(data_info_with_outliers):
    """Test winsorize strategy for handling outliers."""
    cleaner = OutlierCleaner(method="zscore", strategy="winsorize")
    result = cleaner.transform(data_info_with_outliers)

    # Check that extreme values were replaced but count remains same
    assert len(result.data) == len(data_info_with_outliers.data)
    assert result.data["values"].max() < data_info_with_outliers.data["values"].max()
    assert result.data["values"].min() > data_info_with_outliers.data["values"].min()


def test_specific_columns(data_info_with_outliers):
    """Test handling outliers in specific columns only."""
    cleaner = OutlierCleaner(method="zscore", columns=["values"], strategy="clip")
    result = cleaner.transform(data_info_with_outliers)

    # Check that only specified column was modified
    assert not np.array_equal(
        result.data["values"], data_info_with_outliers.data["values"]
    )
    assert np.array_equal(
        result.data["other_numeric"], data_info_with_outliers.data["other_numeric"]
    )


def test_metadata_updates(data_info_with_outliers):
    """Test that metadata is properly updated after transformation."""
    cleaner = OutlierCleaner(method="zscore", threshold=3.0)
    result = cleaner.transform(data_info_with_outliers)

    # Check transform was recorded in metadata
    assert "transforms" in result.metadata
    transform_record = result.metadata["transforms"][0]

    assert transform_record["name"] == "outlier_cleaner"
    assert transform_record["type"] == "cleaner"
    assert transform_record["parameters"] == {
        "method": "zscore",
        "threshold": 3.0,
        "columns": None,
        "strategy": "clip",
    }
    assert "changes" in transform_record


def test_get_parameters():
    """Test get_parameters method."""
    params = {
        "method": "iqr",
        "threshold": 2.0,
        "columns": ["col1", "col2"],
        "strategy": "winsorize",
    }
    cleaner = OutlierCleaner(**params)
    assert cleaner.get_parameters() == params
