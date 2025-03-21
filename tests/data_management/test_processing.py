import pytest
from pathlib import Path
import pandas as pd
from unittest.mock import Mock, patch
from gaubongai.data_management.processing import DataProcessingManager
from gaubongai.data_management.interfaces import (
    DataCategory,
    DataInfo,
    DataPlugin,
    DataTransformation,
    Pipeline,
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
