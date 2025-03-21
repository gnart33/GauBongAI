"""Tests for data management interfaces."""

import pytest
from pathlib import Path
import pandas as pd
import polars as pl
from typing import Dict, Any
from gaubongai.data_management.interfaces import (
    DataCategory,
    DataInfo,
    DataPlugin,
    DataTransformation,
    Pipeline,
    PluginVariant,
)


# ============================================================================
# DataInfo Tests
# ============================================================================


def test_data_info_creation(sample_dataframe):
    """Test DataInfo object creation."""
    metadata = {
        "rows": len(sample_dataframe),
        "columns": list(sample_dataframe.columns),
        "implementation": "pandas",
    }
    data_info = DataInfo(
        data=sample_dataframe,
        metadata=metadata,
        category=DataCategory.TABULAR,
    )

    assert data_info.data is sample_dataframe
    assert data_info.metadata == metadata
    assert data_info.category == DataCategory.TABULAR


def test_data_info_with_polars(sample_polars_df):
    """Test DataInfo object with polars DataFrame."""
    metadata = {
        "rows": sample_polars_df.height,
        "columns": sample_polars_df.columns,
        "implementation": "polars",
        "memory_usage": sample_polars_df.estimated_size(),
    }
    data_info = DataInfo(
        data=sample_polars_df,
        metadata=metadata,
        category=DataCategory.TABULAR,
    )

    assert isinstance(data_info.data, pl.DataFrame)
    assert data_info.metadata["implementation"] == "polars"
    assert "memory_usage" in data_info.metadata


def test_data_info_with_text(sample_text_data):
    """Test DataInfo object with text data."""
    metadata = {
        "type": "text",
        "length": len(sample_text_data),
        "lines": len(sample_text_data.splitlines()),
    }
    data_info = DataInfo(
        data=sample_text_data,
        metadata=metadata,
        category=DataCategory.TEXT,
    )

    assert isinstance(data_info.data, str)
    assert data_info.metadata["type"] == "text"
    assert data_info.category == DataCategory.TEXT


def test_data_info_with_mixed_data(sample_mixed_data):
    """Test DataInfo object with mixed data types."""
    metadata = {
        "rows": len(sample_mixed_data),
        "columns": list(sample_mixed_data.columns),
        "dtypes": sample_mixed_data.dtypes.astype(str).to_dict(),
        "missing_values": sample_mixed_data.isna().sum().to_dict(),
    }
    data_info = DataInfo(
        data=sample_mixed_data,
        metadata=metadata,
        category=DataCategory.MIXED,
    )

    assert isinstance(data_info.data, pd.DataFrame)
    assert "missing_values" in data_info.metadata
    assert data_info.category == DataCategory.MIXED


# ============================================================================
# Plugin Tests
# ============================================================================


def test_plugin_interface():
    """Test plugin interface attributes."""

    class TestPlugin(DataPlugin):
        name = "test"
        variant = PluginVariant.DEFAULT
        supported_extensions = [".test"]
        data_category = DataCategory.TABULAR
        priority = 1

        @classmethod
        def can_handle(cls, file_path: Path) -> bool:
            return file_path.suffix in cls.supported_extensions

        def load(self, file_path: Path, **kwargs) -> DataInfo:
            return DataInfo(
                data=pd.DataFrame(),
                metadata={},
                category=self.data_category,
            )

    plugin = TestPlugin()
    assert hasattr(plugin, "name")
    assert hasattr(plugin, "variant")
    assert hasattr(plugin, "supported_extensions")
    assert hasattr(plugin, "data_category")
    assert hasattr(plugin, "priority")
    assert callable(plugin.can_handle)
    assert callable(plugin.load)


def test_plugin_variants():
    """Test plugin variants."""

    class DefaultPlugin(DataPlugin):
        name = "default"
        variant = PluginVariant.DEFAULT
        supported_extensions = [".test"]
        data_category = DataCategory.TABULAR
        priority = 1

        @classmethod
        def can_handle(cls, file_path: Path) -> bool:
            return file_path.suffix in cls.supported_extensions

        def load(self, file_path: Path, **kwargs) -> DataInfo:
            return DataInfo(
                data=pd.DataFrame(),
                metadata={"implementation": "default"},
                category=self.data_category,
            )

    class MemoryEfficientPlugin(DefaultPlugin):
        name = "memory_efficient"
        variant = PluginVariant.MEMORY_EFFICIENT
        priority = 2

        def load(self, file_path: Path, **kwargs) -> DataInfo:
            data_info = super().load(file_path, **kwargs)
            data_info.metadata["implementation"] = "memory_efficient"
            data_info.metadata["memory_usage"] = 100
            return data_info

    class PerformancePlugin(DefaultPlugin):
        name = "performance"
        variant = PluginVariant.PERFORMANCE
        priority = 3

        def load(self, file_path: Path, **kwargs) -> DataInfo:
            data_info = super().load(file_path, **kwargs)
            data_info.metadata["implementation"] = "performance"
            data_info.metadata["processing_time"] = 0.001
            return data_info

    # Test variant attributes
    default = DefaultPlugin()
    memory = MemoryEfficientPlugin()
    performance = PerformancePlugin()

    assert default.variant == PluginVariant.DEFAULT
    assert memory.variant == PluginVariant.MEMORY_EFFICIENT
    assert performance.variant == PluginVariant.PERFORMANCE

    # Test priorities
    assert default.priority < memory.priority < performance.priority

    # Test metadata
    test_file = Path("test.test")
    default_info = default.load(test_file)
    memory_info = memory.load(test_file)
    performance_info = performance.load(test_file)

    assert default_info.metadata["implementation"] == "default"
    assert memory_info.metadata["implementation"] == "memory_efficient"
    assert performance_info.metadata["implementation"] == "performance"

    assert "memory_usage" in memory_info.metadata
    assert "processing_time" in performance_info.metadata


# ============================================================================
# Transformation Tests
# ============================================================================


def test_transformation_interface():
    """Test transformation interface."""

    class TestTransformation(DataTransformation):
        name = "test"
        supported_categories = [DataCategory.TABULAR]

        def can_transform(self, data_info: DataInfo) -> bool:
            return data_info.category in self.supported_categories

        def transform(self, data_info: DataInfo) -> DataInfo:
            return data_info

    transform = TestTransformation()
    assert hasattr(transform, "name")
    assert hasattr(transform, "supported_categories")
    assert callable(transform.can_transform)
    assert callable(transform.transform)


def test_transformation_with_metadata(sample_data_info):
    """Test transformation with metadata handling."""

    class MetadataTransformation(DataTransformation):
        name = "metadata_transform"
        supported_categories = [DataCategory.TABULAR]

        def can_transform(self, data_info: DataInfo) -> bool:
            return data_info.category in self.supported_categories

        def transform(self, data_info: DataInfo) -> DataInfo:
            data = data_info.data.copy()
            data["new_col"] = 1
            metadata = {
                **data_info.metadata,
                "columns": list(data.columns),
                "transformed": True,
            }
            return DataInfo(
                data=data,
                metadata=metadata,
                category=data_info.category,
            )

    transform = MetadataTransformation()
    result = transform.transform(sample_data_info)

    assert "new_col" in result.data.columns
    assert "transformed" in result.metadata
    assert result.metadata["transformed"]


# ============================================================================
# Pipeline Tests
# ============================================================================


def test_pipeline_interface():
    """Test pipeline interface."""

    class TestPipeline(Pipeline):
        name = "test"
        steps = ["step1", "step2"]

        def execute(self, data_info: DataInfo) -> DataInfo:
            return data_info

    pipeline = TestPipeline()
    assert hasattr(pipeline, "name")
    assert hasattr(pipeline, "steps")
    assert callable(pipeline.execute)


def test_pipeline_with_metadata(sample_data_info):
    """Test pipeline with metadata handling."""

    class MetadataPipeline(Pipeline):
        name = "metadata_pipeline"
        steps = ["transform1", "transform2"]

        def execute(self, data_info: DataInfo) -> DataInfo:
            metadata = {
                **data_info.metadata,
                "pipeline": self.name,
                "steps": self.steps,
            }
            return DataInfo(
                data=data_info.data,
                metadata=metadata,
                category=data_info.category,
            )

    pipeline = MetadataPipeline()
    result = pipeline.execute(sample_data_info)

    assert "pipeline" in result.metadata
    assert result.metadata["pipeline"] == pipeline.name
    assert "steps" in result.metadata
    assert result.metadata["steps"] == pipeline.steps
