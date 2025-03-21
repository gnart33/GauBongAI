"""Tests for data management plugins."""

import pytest
from pathlib import Path
from typing import List, Type

from gaubongai.data_management.interfaces import (
    DataPlugin,
    DataInfo,
    DataCategory,
    PluginVariant,
)
from gaubongai.data_management.plugins.csv_plugin import (
    PandasCSVPlugin,
    PolarsCSVPlugin,
)


# ============================================================================
# Plugin Base Tests
# ============================================================================


def test_plugin_registration(mock_plugins: List[Type[DataPlugin]]):
    """Test that plugins can be registered and retrieved correctly."""
    assert len(mock_plugins) == 3
    assert all(issubclass(plugin, DataPlugin) for plugin in mock_plugins)
    assert all(hasattr(plugin, "name") for plugin in mock_plugins)
    assert all(hasattr(plugin, "variant") for plugin in mock_plugins)
    assert all(hasattr(plugin, "priority") for plugin in mock_plugins)


def test_plugin_variants(mock_plugins: List[Type[DataPlugin]]):
    """Test that plugin variants are correctly defined."""
    variants = [plugin.variant for plugin in mock_plugins]
    assert PluginVariant.DEFAULT in variants
    assert PluginVariant.MEMORY_EFFICIENT in variants
    assert PluginVariant.PERFORMANCE in variants


def test_plugin_priorities(mock_plugins: List[Type[DataPlugin]]):
    """Test that plugin priorities are correctly ordered."""
    priorities = [plugin.priority for plugin in mock_plugins]
    assert priorities == sorted(priorities)  # Ensure priorities are in ascending order
    assert len(set(priorities)) == len(priorities)  # Ensure unique priorities


# ============================================================================
# Mock Plugin Tests
# ============================================================================


def test_mock_plugin_can_handle(mock_plugin: DataPlugin, mock_file_path: Path):
    """Test that mock plugin correctly identifies files it can handle."""
    assert mock_plugin.can_handle(mock_file_path)
    assert not mock_plugin.can_handle(Path("test.csv"))


def test_mock_plugin_load(mock_plugin: DataPlugin, mock_file_path: Path):
    """Test that mock plugin correctly loads data."""
    data_info = mock_plugin.load(mock_file_path)
    assert isinstance(data_info, DataInfo)
    assert data_info.category == DataCategory.TABULAR
    assert "test" in data_info.data
    assert data_info.metadata["implementation"] == "mock"


def test_mock_memory_efficient_plugin(
    mock_memory_efficient_plugin: DataPlugin, mock_file_path: Path
):
    """Test that memory efficient plugin adds memory usage metadata."""
    data_info = mock_memory_efficient_plugin.load(mock_file_path)
    assert data_info.metadata["implementation"] == "mock_memory_efficient"
    assert "memory_usage" in data_info.metadata
    assert data_info.metadata["memory_usage"] == 100


def test_mock_performance_plugin(
    mock_performance_plugin: DataPlugin, mock_file_path: Path
):
    """Test that performance plugin adds processing time metadata."""
    data_info = mock_performance_plugin.load(mock_file_path)
    assert data_info.metadata["implementation"] == "mock_performance"
    assert "processing_time" in data_info.metadata
    assert isinstance(data_info.metadata["processing_time"], float)


# ============================================================================
# CSV Plugin Tests
# ============================================================================


def test_pandas_csv_plugin_can_handle():
    """Test that pandas CSV plugin correctly identifies CSV files."""
    plugin = PandasCSVPlugin()
    assert plugin.can_handle(Path("test.csv"))
    assert not plugin.can_handle(Path("test.txt"))


def test_polars_csv_plugin_can_handle():
    """Test that polars CSV plugin correctly identifies CSV files."""
    plugin = PolarsCSVPlugin()
    assert plugin.can_handle(Path("test.csv"))
    assert not plugin.can_handle(Path("test.txt"))


def test_pandas_csv_plugin_load(sample_csv_file: Path):
    """Test that pandas CSV plugin correctly loads data."""
    plugin = PandasCSVPlugin()
    data_info = plugin.load(sample_csv_file)
    assert isinstance(data_info, DataInfo)
    assert data_info.category == DataCategory.TABULAR
    assert data_info.metadata["implementation"] == "pandas"
    assert len(data_info.data) == 5  # Sample data has 5 rows
    assert all(
        col in data_info.data.columns
        for col in ["id", "name", "age", "score", "active"]
    )


def test_polars_csv_plugin_load(sample_csv_file: Path):
    """Test that polars CSV plugin correctly loads data."""
    plugin = PolarsCSVPlugin()
    data_info = plugin.load(sample_csv_file)
    assert isinstance(data_info, DataInfo)
    assert data_info.category == DataCategory.TABULAR
    assert data_info.metadata["implementation"] == "polars"
    assert "memory_usage" in data_info.metadata
    assert data_info.data.height == 5  # Sample data has 5 rows
    assert all(
        col in data_info.data.columns
        for col in ["id", "name", "age", "score", "active"]
    )


def test_csv_plugin_variants():
    """Test that CSV plugins have correct variants."""
    pandas_plugin = PandasCSVPlugin()
    polars_plugin = PolarsCSVPlugin()

    assert pandas_plugin.variant == PluginVariant.DEFAULT
    assert polars_plugin.variant == PluginVariant.MEMORY_EFFICIENT
    assert pandas_plugin.priority < polars_plugin.priority


def test_csv_plugin_metadata(sample_csv_file: Path):
    """Test that CSV plugins provide correct metadata."""
    pandas_plugin = PandasCSVPlugin()
    polars_plugin = PolarsCSVPlugin()

    pandas_info = pandas_plugin.load(sample_csv_file)
    polars_info = polars_plugin.load(sample_csv_file)

    # Check common metadata
    assert pandas_info.metadata["rows"] == polars_info.metadata["rows"]
    assert pandas_info.metadata["columns"] == polars_info.metadata["columns"]

    # Check implementation-specific metadata
    assert pandas_info.metadata["implementation"] == "pandas"
    assert polars_info.metadata["implementation"] == "polars"
    assert "memory_usage" in polars_info.metadata


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_plugin_load_nonexistent_file(mock_plugin: DataPlugin):
    """Test that plugin handles nonexistent file correctly."""
    with pytest.raises(FileNotFoundError):
        mock_plugin.load(Path("nonexistent.mock"))


def test_plugin_load_invalid_file(mock_plugin: DataPlugin, tmp_path: Path):
    """Test that plugin handles invalid file content correctly."""
    invalid_file = tmp_path / "invalid.mock"
    invalid_file.write_text("invalid content")

    with pytest.raises(Exception):
        mock_plugin.load(invalid_file)


def test_plugin_load_wrong_extension(mock_plugin: DataPlugin, sample_csv_file: Path):
    """Test that plugin handles wrong file extension correctly."""
    assert not mock_plugin.can_handle(sample_csv_file)

    with pytest.raises(ValueError):
        mock_plugin.load(sample_csv_file)
