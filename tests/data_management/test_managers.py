import pytest
from pathlib import Path
import pandas as pd
from unittest.mock import Mock, patch
from gaubongai.data_management.managers import PluginManager, PipelineManager
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


# Plugin Manager Tests
@pytest.fixture
def plugin_manager():
    """Create a PluginManager instance."""
    return PluginManager()


def test_plugin_registration(plugin_manager):
    """Test registering a plugin."""
    plugin = MockPlugin()
    plugin_manager.plugins[plugin.name] = plugin

    assert plugin_manager.get_plugin(Path("test.mock")) == plugin
    assert plugin_manager.list_plugins() == [plugin.name]


def test_get_nonexistent_plugin(plugin_manager):
    """Test getting a plugin for an unsupported file type."""
    assert plugin_manager.get_plugin(Path("test.unsupported")) is None


def test_get_plugins_for_category(plugin_manager):
    """Test getting plugins for a specific category."""
    plugin = MockPlugin()
    plugin_manager.plugins[plugin.name] = plugin

    plugins = plugin_manager.get_plugins_for_category(DataCategory.TABULAR)
    assert plugins == [plugin]

    # Test empty category
    assert plugin_manager.get_plugins_for_category(DataCategory.TEXT) == []


# Pipeline Manager Tests
@pytest.fixture
def pipeline_manager():
    """Create a PipelineManager instance."""
    return PipelineManager()


def test_register_processor(pipeline_manager):
    """Test registering a processor."""
    processor = MockTransformation()
    pipeline_manager.register_processor(processor)

    assert pipeline_manager._processors[processor.name] == processor
    assert pipeline_manager.list_processors() == [processor.name]


def test_register_duplicate_processor(pipeline_manager):
    """Test registering a processor with a duplicate name."""
    processor = MockTransformation()
    pipeline_manager.register_processor(processor)

    with pytest.raises(
        ValueError, match=f"Processor '{processor.name}' already exists"
    ):
        pipeline_manager.register_processor(processor)


def test_create_pipeline(pipeline_manager):
    """Test creating a new pipeline."""
    # Register required processor first
    processor = MockTransformation()
    pipeline_manager.register_processor(processor)

    # Create pipeline
    pipeline_name = "test_pipeline"
    steps = [processor.name]
    pipeline_manager.create_pipeline(pipeline_name, steps)

    assert pipeline_name in pipeline_manager._pipelines
    assert pipeline_manager.list_pipelines() == [pipeline_name]


def test_create_pipeline_with_invalid_step(pipeline_manager):
    """Test creating a pipeline with an invalid step."""
    pipeline_name = "test_pipeline"
    steps = ["nonexistent_processor"]

    with pytest.raises(ValueError, match="Invalid processor"):
        pipeline_manager.create_pipeline(pipeline_name, steps)


def test_create_duplicate_pipeline(pipeline_manager):
    """Test creating a pipeline with a duplicate name."""
    processor = MockTransformation()
    pipeline_manager.register_processor(processor)

    pipeline_name = "test_pipeline"
    steps = [processor.name]
    pipeline_manager.create_pipeline(pipeline_name, steps)

    with pytest.raises(ValueError, match=f"Pipeline '{pipeline_name}' already exists"):
        pipeline_manager.create_pipeline(pipeline_name, steps)


def test_get_pipeline(pipeline_manager):
    """Test getting a pipeline."""
    # Register processor and pipeline
    processor = MockTransformation()
    pipeline_manager.register_processor(processor)

    pipeline_name = "test_pipeline"
    steps = [processor.name]
    pipeline_manager.create_pipeline(pipeline_name, steps)

    # Get pipeline
    pipeline = pipeline_manager.get_pipeline(pipeline_name)
    assert isinstance(pipeline, Pipeline)
    assert pipeline.name == pipeline_name
    assert pipeline.steps == steps


def test_get_nonexistent_pipeline(pipeline_manager):
    """Test getting a nonexistent pipeline."""
    assert pipeline_manager.get_pipeline("nonexistent") is None


def test_pipeline_execution(pipeline_manager):
    """Test executing a pipeline."""
    # Register processor and pipeline
    processor = MockTransformation()
    pipeline_manager.register_processor(processor)

    pipeline_name = "test_pipeline"
    steps = [processor.name]
    pipeline_manager.create_pipeline(pipeline_name, steps)

    # Create test data
    data = pd.DataFrame({"A": [1, 2, 3]})
    data_info = DataInfo(
        data=data, metadata={"rows": 3, "columns": ["A"]}, category=DataCategory.TABULAR
    )

    # Execute pipeline
    pipeline = pipeline_manager.get_pipeline(pipeline_name)
    result = pipeline.execute(data_info)

    # Verify transformation
    assert set(result.data.columns) == {"A", "B"}
    assert (result.data["B"] == data["A"] * 2).all()


def test_pipeline_with_multiple_steps(pipeline_manager):
    """Test pipeline with multiple transformation steps."""
    # Create and register two processors
    processor1 = MockTransformation()
    processor2 = type(
        "MockTransformation2", (MockTransformation,), {"name": "mock_transform2"}
    )()

    pipeline_manager.register_processor(processor1)
    pipeline_manager.register_processor(processor2)

    # Create pipeline with both processors
    pipeline_name = "multi_step_pipeline"
    steps = [processor1.name, processor2.name]
    pipeline_manager.create_pipeline(pipeline_name, steps)

    # Create test data
    data = pd.DataFrame({"A": [1, 2, 3]})
    data_info = DataInfo(
        data=data, metadata={"rows": 3, "columns": ["A"]}, category=DataCategory.TABULAR
    )

    # Execute pipeline
    pipeline = pipeline_manager.get_pipeline(pipeline_name)
    result = pipeline.execute(data_info)

    # Verify both transformations were applied
    assert set(result.data.columns) == {"A", "B"}
    assert result.metadata["columns"] == ["A", "B"]
