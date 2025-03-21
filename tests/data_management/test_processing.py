import pytest
from pathlib import Path
import pandas as pd
import polars as pl
from typing import Dict, Any
from unittest.mock import Mock, patch
from gaubongai.data_management.processing import DataProcessingManager
from gaubongai.data_management.interfaces import (
    DataCategory,
    DataInfo,
    DataPlugin,
    DataTransformation,
    Pipeline,
    PluginVariant,
)


class MockPlugin:
    name = "mock_plugin"
    supported_extensions = [".mock"]
    data_category = DataCategory.TABULAR

    @classmethod
    def can_handle(cls, file_path: Path) -> bool:
        return file_path.suffix in cls.supported_extensions

    def load(self, file_path: Path, **kwargs) -> DataInfo:
        data = pd.DataFrame({"A": [1, 2, 3]})
        metadata = {"rows": 3, "columns": ["A"]}
        return DataInfo(data=data, metadata=metadata, category=self.data_category)


class MockTransformation:
    name = "mock_transform"
    supported_categories = [DataCategory.TABULAR]

    def can_transform(self, data_info: DataInfo) -> bool:
        return data_info.category in self.supported_categories

    def transform(self, data_info: DataInfo) -> DataInfo:
        # Add a new column
        data = data_info.data.copy()
        data["B"] = data["A"] * 2
        metadata = {**data_info.metadata, "columns": ["A", "B"]}
        return DataInfo(data=data, metadata=metadata, category=data_info.category)


class MockPipeline:
    name = "mock_pipeline"
    steps = ["mock_transform"]

    def execute(self, data_info: DataInfo) -> DataInfo:
        transform = MockTransformation()
        return transform.transform(data_info)


@pytest.fixture
def processing_manager():
    """Create a DataProcessingManager instance with mocked dependencies."""
    manager = DataProcessingManager()

    # Create a DataInfo instance for reuse
    data_info = DataInfo(
        data=pd.DataFrame({"A": [1, 2, 3]}),
        metadata={"rows": 3, "columns": ["A"]},
        category=DataCategory.TABULAR,
    )

    # Mock plugin manager
    mock_plugin = Mock()
    mock_plugin.load = Mock(return_value=data_info)
    manager.plugin_manager = Mock()
    manager.plugin_manager.get_plugin = Mock(return_value=mock_plugin)
    manager.plugin_manager.list_plugins = Mock(return_value=["mock_plugin"])

    # Mock pipeline manager
    mock_pipeline = Mock()
    mock_pipeline.execute = Mock(
        return_value=DataInfo(
            data=pd.DataFrame({"A": [1, 2, 3], "B": [2, 4, 6]}),
            metadata={"rows": 3, "columns": ["A", "B"]},
            category=DataCategory.TABULAR,
        )
    )
    manager.pipeline_manager = Mock()
    manager.pipeline_manager.get_pipeline = Mock(return_value=mock_pipeline)
    manager.pipeline_manager.list_pipelines = Mock(return_value=["mock_pipeline"])
    manager.pipeline_manager.create_pipeline = Mock()
    manager.pipeline_manager.register_processor = Mock()

    # Mock data storage
    manager.data_storage = Mock()
    manager.data_storage.store = Mock()
    manager.data_storage.get = Mock(return_value=data_info)
    manager.data_storage.get_metadata = Mock(return_value={"rows": 3, "columns": ["A"]})
    manager.data_storage.list_by_category = Mock(return_value=["data1", "data2"])
    manager.data_storage.list_all = Mock(
        return_value={DataCategory.TABULAR: ["data1", "data2"]}
    )

    return manager


def test_process_file(processing_manager, tmp_path):
    """Test processing a file with a plugin."""
    # Create a mock file
    file_path = tmp_path / "test.mock"
    file_path.touch()

    # Create expected data
    expected_data = DataInfo(
        data=pd.DataFrame({"A": [1, 2, 3]}),
        metadata={"rows": 3, "columns": ["A"]},
        category=DataCategory.TABULAR,
    )

    # Setup mock plugin
    mock_plugin_cls = Mock()
    mock_plugin_instance = Mock()
    mock_plugin_instance.load.return_value = expected_data
    mock_plugin_cls.return_value = mock_plugin_instance
    processing_manager.plugin_manager.get_plugin.return_value = mock_plugin_cls

    # Process the file
    result = processing_manager.process_file(file_path)

    # Verify the result
    assert isinstance(result, DataInfo)
    assert isinstance(result.data, pd.DataFrame)
    assert "A" in result.data.columns
    assert result.category == DataCategory.TABULAR


def test_process_file_with_pipeline(processing_manager, tmp_path):
    """Test processing a file with a plugin and pipeline."""
    # Create a mock file
    file_path = tmp_path / "test.mock"
    file_path.touch()

    # Process the file with pipeline
    result = processing_manager.process_file(file_path, pipeline_name="mock_pipeline")

    # Verify the result
    assert isinstance(result, DataInfo)
    assert isinstance(result.data, pd.DataFrame)
    assert set(result.data.columns) == {"A", "B"}  # Pipeline adds column B
    assert result.category == DataCategory.TABULAR


def test_process_file_unsupported_extension(processing_manager, tmp_path):
    """Test processing a file with unsupported extension."""
    file_path = tmp_path / "test.unsupported"
    file_path.touch()

    processing_manager.plugin_manager.get_plugin.return_value = None

    with pytest.raises(ValueError, match="No plugin found"):
        processing_manager.process_file(file_path)


def test_process_file_nonexistent_pipeline(processing_manager, tmp_path):
    """Test processing a file with a nonexistent pipeline."""
    file_path = tmp_path / "test.mock"
    file_path.touch()

    processing_manager.pipeline_manager.get_pipeline.return_value = None

    with pytest.raises(ValueError, match="Pipeline not found: nonexistent"):
        processing_manager.process_file(file_path, pipeline_name="nonexistent")


def test_get_data(processing_manager):
    """Test retrieving data by name and category."""
    data = pd.DataFrame({"A": [1, 2, 3]})
    data_info = DataInfo(data=data, metadata={}, category=DataCategory.TABULAR)
    processing_manager.data_storage.get.return_value = data_info

    result = processing_manager.get_data("test_data", DataCategory.TABULAR)
    assert result == data_info


def test_get_metadata(processing_manager):
    """Test retrieving metadata for stored data."""
    metadata = {"rows": 3, "columns": ["A"]}
    processing_manager.data_storage.get_metadata.return_value = metadata

    result = processing_manager.get_metadata("test_data", DataCategory.TABULAR)
    assert result == metadata


def test_list_data(processing_manager):
    """Test listing available data."""
    data_list = {DataCategory.TABULAR: ["data1", "data2"]}
    processing_manager.data_storage.list_all.return_value = data_list

    result = processing_manager.list_data()
    assert result == data_list

    # Test listing by category
    category_list = ["data1", "data2"]
    processing_manager.data_storage.list_by_category.return_value = category_list
    result = processing_manager.list_data(category=DataCategory.TABULAR)
    assert result == category_list


def test_list_plugins(processing_manager):
    """Test listing available plugins."""
    plugins = ["mock_plugin"]
    processing_manager.plugin_manager.list_plugins.return_value = plugins

    result = processing_manager.list_plugins()
    assert result == plugins


def test_list_pipelines(processing_manager):
    """Test listing available pipelines."""
    pipelines = ["mock_pipeline"]
    processing_manager.pipeline_manager.list_pipelines.return_value = pipelines

    result = processing_manager.list_pipelines()
    assert result == pipelines


def test_create_pipeline(processing_manager):
    """Test creating a new pipeline."""
    name = "new_pipeline"
    steps = ["mock_transform"]

    processing_manager.create_pipeline(name, steps)
    processing_manager.pipeline_manager.create_pipeline.assert_called_once_with(
        "new_pipeline", ["mock_transform"]
    )


def test_register_processor(processing_manager):
    """Test registering a new processor."""
    processor = Mock(name="mock_transform")
    processing_manager.register_processor(processor)
    processing_manager.pipeline_manager.register_processor.assert_called_once_with(
        processor
    )


@patch("gaubongai.data_management.plugins.csv_plugin.CSVPlugin")
def test_load_csv_backward_compatibility(
    mock_csv_plugin_cls, processing_manager, tmp_path
):
    """Test the backward compatibility CSV loading method."""
    # Create test data
    test_data = pd.DataFrame({"A": [1, 2, 3]})

    # Setup mock CSV plugin
    mock_plugin_instance = Mock()
    mock_csv_plugin_cls.return_value = mock_plugin_instance
    data_info = DataInfo(
        data=test_data,
        metadata={"rows": 3, "columns": ["A"]},
        category=DataCategory.TABULAR,
    )
    mock_plugin_instance.load.return_value = data_info
    processing_manager.plugin_manager.get_plugin.return_value = mock_csv_plugin_cls

    # Create a test CSV file
    file_path = tmp_path / "test.csv"
    file_path.touch()

    # Test loading CSV
    result = processing_manager.load_csv(file_path)

    # Verify the result
    assert isinstance(result, pd.DataFrame)
    assert "A" in result.columns
    pd.testing.assert_frame_equal(result, test_data)
    mock_plugin_instance.load.assert_called_once_with(file_path)


def test_processing_manager_init():
    """Test processing manager initialization."""
    manager = DataProcessingManager()
    assert manager.plugin_manager is not None
    assert manager.pipeline_manager is not None
    assert manager.data_storage is not None


def test_process_file_with_default_variant(mock_plugins, mock_file_path):
    """Test processing file with default variant."""
    manager = DataProcessingManager()
    for plugin in mock_plugins:
        manager.plugin_manager.register_plugin(plugin)

    data_info = manager.process_file(mock_file_path, name="test")
    assert data_info is not None
    assert data_info.category == DataCategory.TABULAR
    assert "test" in data_info.metadata.get("columns", [])


def test_process_file_with_memory_efficient_variant(mock_plugins, mock_file_path):
    """Test processing file with memory efficient variant."""
    manager = DataProcessingManager()
    for plugin in mock_plugins:
        manager.plugin_manager.register_plugin(plugin)

    data_info = manager.process_file(
        mock_file_path, name="test", plugin_variant=PluginVariant.MEMORY_EFFICIENT
    )
    assert data_info is not None
    assert data_info.metadata.get("implementation") == "memory_efficient"
    assert "memory_usage" in data_info.metadata


def test_process_file_with_pipeline(mock_plugins, mock_transformation, mock_file_path):
    """Test processing file with pipeline."""
    manager = DataProcessingManager()
    for plugin in mock_plugins:
        manager.plugin_manager.register_plugin(plugin)

    manager.pipeline_manager.register_processor(mock_transformation)
    manager.pipeline_manager.create_pipeline(
        "test_pipeline", [mock_transformation.name]
    )

    data_info = manager.process_file(
        mock_file_path, name="test", pipeline_name="test_pipeline"
    )
    assert data_info is not None


def test_process_file_nonexistent_plugin():
    """Test processing file with no available plugin."""
    manager = DataProcessingManager()
    with pytest.raises(ValueError, match="No plugin found for file"):
        manager.process_file(Path("test.unsupported"))


def test_process_file_nonexistent_pipeline(mock_plugins, mock_file_path):
    """Test processing file with nonexistent pipeline."""
    manager = DataProcessingManager()
    for plugin in mock_plugins:
        manager.plugin_manager.register_plugin(plugin)

    with pytest.raises(ValueError, match="Pipeline not found"):
        manager.process_file(mock_file_path, name="test", pipeline_name="nonexistent")


def test_list_available_variants(mock_plugins, mock_file_path):
    """Test listing available variants for a file."""
    manager = DataProcessingManager()
    for plugin in mock_plugins:
        manager.plugin_manager.register_plugin(plugin)

    variants_info = manager.list_available_variants(mock_file_path)
    assert variants_info["file_type"] == mock_file_path.suffix
    assert len(variants_info["variants"]) == len(mock_plugins)

    # Check variant information
    for variant_info in variants_info["variants"]:
        assert "name" in variant_info
        assert "variant" in variant_info
        assert "priority" in variant_info


def test_process_csv_file_pandas(sample_csv_file):
    """Test processing CSV file with pandas plugin."""
    manager = DataProcessingManager()
    data_info = manager.process_file(
        sample_csv_file, name="test_csv", plugin_variant=PluginVariant.DEFAULT
    )
    assert data_info is not None
    assert isinstance(data_info.data, pd.DataFrame)
    assert data_info.metadata.get("implementation") == "pandas"


def test_process_csv_file_polars(sample_csv_file):
    """Test processing CSV file with polars plugin."""
    manager = DataProcessingManager()
    data_info = manager.process_file(
        sample_csv_file, name="test_csv", plugin_variant=PluginVariant.MEMORY_EFFICIENT
    )
    assert data_info is not None
    assert isinstance(data_info.data, pl.DataFrame)
    assert data_info.metadata.get("implementation") == "polars"
    assert "memory_usage" in data_info.metadata


def test_get_data(mock_plugins, mock_file_path):
    """Test retrieving stored data."""
    manager = DataProcessingManager()
    for plugin in mock_plugins:
        manager.plugin_manager.register_plugin(plugin)

    # Process and store data
    manager.process_file(mock_file_path, name="test")

    # Retrieve data
    data_info = manager.get_data("test")
    assert data_info is not None
    assert data_info.category == DataCategory.TABULAR


def test_get_metadata(mock_plugins, mock_file_path):
    """Test retrieving metadata."""
    manager = DataProcessingManager()
    for plugin in mock_plugins:
        manager.plugin_manager.register_plugin(plugin)

    # Process and store data
    manager.process_file(mock_file_path, name="test")

    # Retrieve metadata
    metadata = manager.get_metadata("test")
    assert metadata is not None
    assert "columns" in metadata


def test_list_data(mock_plugins, mock_file_path):
    """Test listing available data."""
    manager = DataProcessingManager()
    for plugin in mock_plugins:
        manager.plugin_manager.register_plugin(plugin)

    # Process and store data
    manager.process_file(mock_file_path, name="test")

    # List all data
    data_list = manager.list_data()
    assert isinstance(data_list, dict)
    assert DataCategory.TABULAR in data_list
    assert "test" in data_list[DataCategory.TABULAR]

    # List data by category
    category_list = manager.list_data(DataCategory.TABULAR)
    assert isinstance(category_list, list)
    assert "test" in category_list
