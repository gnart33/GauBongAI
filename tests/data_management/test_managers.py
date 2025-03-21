import pytest
from pathlib import Path
import pandas as pd
from unittest.mock import Mock, patch
from gaubongai.data_management.managers import (
    PluginManager,
    PipelineManager,
    DataProcessingManager,
    QualityManager,
)
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
    variant = PluginVariant.DEFAULT
    supported_extensions = [".mock"]
    data_category = DataCategory.TABULAR
    priority = 1

    @classmethod
    def can_handle(cls, file_path: Path) -> bool:
        return file_path.suffix in cls.supported_extensions

    def load(self, file_path: Path, **kwargs) -> DataInfo:
        data = pd.DataFrame({"A": [1, 2, 3]})
        metadata = {
            "rows": 3,
            "columns": ["A"],
            "implementation": "mock",
        }
        return DataInfo(data=data, metadata=metadata, category=self.data_category)


class MockMemoryEfficientPlugin(MockPlugin):
    name = "mock_memory_efficient"
    variant = PluginVariant.MEMORY_EFFICIENT
    priority = 2

    def load(self, file_path: Path, **kwargs) -> DataInfo:
        data_info = super().load(file_path, **kwargs)
        data_info.metadata["implementation"] = "mock_memory_efficient"
        data_info.metadata["memory_usage"] = 100
        return data_info


class MockPerformancePlugin(MockPlugin):
    name = "mock_performance"
    variant = PluginVariant.PERFORMANCE
    priority = 3

    def load(self, file_path: Path, **kwargs) -> DataInfo:
        data_info = super().load(file_path, **kwargs)
        data_info.metadata["implementation"] = "mock_performance"
        data_info.metadata["processing_time"] = 0.001
        return data_info


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


# ============================================================================
# Plugin Manager Tests
# ============================================================================


@pytest.fixture
def plugin_manager():
    """Create a PluginManager instance."""
    return PluginManager()


def test_plugin_manager_init():
    """Test plugin manager initialization."""
    manager = PluginManager()
    assert isinstance(manager.plugins, dict)
    assert manager.plugin_package == "gaubongai.data_management.plugins"


def test_plugin_registration(mock_plugin):
    """Test plugin registration."""
    manager = PluginManager()
    manager.register_plugin(type(mock_plugin))
    assert ".mock" in manager.plugins
    assert len(manager.plugins[".mock"]) == 1


def test_duplicate_plugin_registration(mock_plugin):
    """Test registering duplicate plugin."""
    manager = PluginManager()
    manager.register_plugin(type(mock_plugin))
    with pytest.raises(ValueError):
        manager.register_plugin(type(mock_plugin))


def test_get_plugin_by_extension(mock_plugin, mock_file_path):
    """Test getting plugin by file extension."""
    manager = PluginManager()
    manager.register_plugin(type(mock_plugin))
    plugin = manager.get_plugin(mock_file_path)
    assert plugin is not None
    assert plugin.name == mock_plugin.name


def test_get_plugin_with_variant(mock_plugins, mock_file_path):
    """Test getting plugin with specific variant."""
    manager = PluginManager()
    for plugin in mock_plugins:
        manager.register_plugin(plugin)

    # Test getting default variant
    default_plugin = manager.get_plugin(mock_file_path, variant=PluginVariant.DEFAULT)
    assert default_plugin is not None
    assert default_plugin.variant == PluginVariant.DEFAULT

    # Test getting memory efficient variant
    memory_plugin = manager.get_plugin(
        mock_file_path, variant=PluginVariant.MEMORY_EFFICIENT
    )
    assert memory_plugin is not None
    assert memory_plugin.variant == PluginVariant.MEMORY_EFFICIENT

    # Test getting performance variant
    performance_plugin = manager.get_plugin(
        mock_file_path, variant=PluginVariant.PERFORMANCE
    )
    assert performance_plugin is not None
    assert performance_plugin.variant == PluginVariant.PERFORMANCE


def test_get_plugin_by_name(mock_plugins, mock_file_path):
    """Test getting plugin by name."""
    manager = PluginManager()
    for plugin in mock_plugins:
        manager.register_plugin(plugin)

    # Test getting plugin by name
    plugin = manager.get_plugin(mock_file_path, plugin_name="mock_memory_efficient")
    assert plugin is not None
    assert plugin.name == "mock_memory_efficient"
    assert plugin.variant == PluginVariant.MEMORY_EFFICIENT


def test_get_plugin_priority(mock_plugins, mock_file_path):
    """Test plugin selection based on priority."""
    manager = PluginManager()
    for plugin in mock_plugins:
        manager.register_plugin(plugin)

    # Without specifying variant or name, should get highest priority plugin
    plugin = manager.get_plugin(mock_file_path)
    assert plugin is not None
    assert plugin.priority == max(p.priority for p in mock_plugins)


def test_get_variants_for_file(mock_plugins, mock_file_path):
    """Test getting available variants for a file."""
    manager = PluginManager()
    for plugin in mock_plugins:
        manager.register_plugin(plugin)

    variants = manager.get_variants_for_file(mock_file_path)
    assert len(variants) == len(mock_plugins)

    # Check variants are sorted by priority
    priorities = [priority for _, _, priority in variants]
    assert priorities == sorted(priorities, reverse=True)


def test_get_nonexistent_plugin(mock_file_path):
    """Test getting plugin for unsupported file type."""
    manager = PluginManager()
    plugin = manager.get_plugin(mock_file_path)
    assert plugin is None


def test_list_plugins(mock_plugins):
    """Test listing available plugins."""
    manager = PluginManager()
    for plugin in mock_plugins:
        manager.register_plugin(plugin)
    plugins = manager.list_plugins()
    assert len(plugins) == len(mock_plugins)


def test_get_plugins_for_category(mock_plugins):
    """Test getting plugins by category."""
    manager = PluginManager()
    for plugin in mock_plugins:
        manager.register_plugin(plugin)
    plugins = manager.get_plugins_for_category(DataCategory.TABULAR)
    assert len(plugins) == len(mock_plugins)


def test_plugin_discovery():
    """Test automatic plugin discovery."""
    manager = PluginManager()
    manager._load_plugins()

    # Check CSV plugins are discovered
    assert ".csv" in manager.plugins
    assert len(manager.plugins[".csv"]) == 2  # pandas and polars plugins

    # Check variants
    csv_plugins = manager.plugins[".csv"]
    variants = [p.variant for p in csv_plugins]
    assert PluginVariant.DEFAULT in variants
    assert PluginVariant.MEMORY_EFFICIENT in variants


# ============================================================================
# Data Processing Manager Tests
# ============================================================================


@pytest.fixture
def data_processing_manager():
    """Create a DataProcessingManager instance."""
    return DataProcessingManager()


def test_data_processing_manager_init():
    """Test data processing manager initialization."""
    manager = DataProcessingManager()
    assert isinstance(manager.plugin_manager, PluginManager)


def test_process_file_with_variant(data_processing_manager, sample_csv_file):
    """Test processing file with specific variant."""
    data_info = data_processing_manager.process_file(
        sample_csv_file, variant=PluginVariant.MEMORY_EFFICIENT
    )
    assert data_info.metadata["implementation"] == "polars"
    assert "memory_usage" in data_info.metadata


def test_process_file_with_plugin_name(data_processing_manager, sample_csv_file):
    """Test processing file with specific plugin name."""
    data_info = data_processing_manager.process_file(
        sample_csv_file, plugin_name="csv_default"
    )
    assert data_info.metadata["implementation"] == "pandas"


def test_process_file_nonexistent(data_processing_manager):
    """Test processing nonexistent file."""
    with pytest.raises(FileNotFoundError):
        data_processing_manager.process_file(Path("nonexistent.csv"))


def test_process_file_unsupported(data_processing_manager):
    """Test processing unsupported file type."""
    with pytest.raises(ValueError):
        data_processing_manager.process_file(Path("test.unsupported"))


# ============================================================================
# Pipeline Manager Tests
# ============================================================================


@pytest.fixture
def pipeline_manager():
    """Create a PipelineManager instance."""
    return PipelineManager()


def test_pipeline_manager_init():
    """Test pipeline manager initialization."""
    manager = PipelineManager()
    assert isinstance(manager._processors, dict)
    assert isinstance(manager._pipelines, dict)


def test_register_processor(mock_transformation):
    """Test processor registration."""
    manager = PipelineManager()
    manager.register_processor(mock_transformation)
    assert mock_transformation.name in manager._processors


def test_duplicate_processor_registration(mock_transformation):
    """Test registering duplicate processor."""
    manager = PipelineManager()
    manager.register_processor(mock_transformation)
    with pytest.raises(ValueError):
        manager.register_processor(mock_transformation)


def test_create_pipeline(mock_transformation):
    """Test pipeline creation."""
    manager = PipelineManager()
    manager.register_processor(mock_transformation)
    manager.create_pipeline("test", [mock_transformation.name])
    assert "test" in manager._pipelines


def test_create_pipeline_invalid_processor():
    """Test creating pipeline with invalid processor."""
    manager = PipelineManager()
    with pytest.raises(ValueError):
        manager.create_pipeline("test", ["invalid"])


def test_get_pipeline(mock_transformation):
    """Test getting pipeline."""
    manager = PipelineManager()
    manager.register_processor(mock_transformation)
    manager.create_pipeline("test", [mock_transformation.name])
    pipeline = manager.get_pipeline("test")
    assert pipeline is not None
    assert pipeline.name == "test"


def test_list_pipelines(mock_transformation):
    """Test listing pipelines."""
    manager = PipelineManager()
    manager.register_processor(mock_transformation)
    manager.create_pipeline("test", [mock_transformation.name])
    pipelines = manager.list_pipelines()
    assert "test" in pipelines


def test_list_processors(mock_transformation):
    """Test listing processors."""
    manager = PipelineManager()
    manager.register_processor(mock_transformation)
    processors = manager.list_processors()
    assert mock_transformation.name in processors


def test_create_duplicate_pipeline(pipeline_manager, mock_transformation):
    """Test creating a duplicate pipeline."""
    pipeline_manager.register_processor(mock_transformation)
    pipeline_manager.create_pipeline("test", [mock_transformation.name])

    with pytest.raises(ValueError):
        pipeline_manager.create_pipeline("test", [mock_transformation.name])


def test_get_nonexistent_pipeline(pipeline_manager):
    """Test getting nonexistent pipeline."""
    pipeline = pipeline_manager.get_pipeline("nonexistent")
    assert pipeline is None


def test_pipeline_execution(pipeline_manager, mock_transformation, sample_data_info):
    """Test pipeline execution."""
    pipeline_manager.register_processor(mock_transformation)
    pipeline_manager.create_pipeline("test", [mock_transformation.name])

    pipeline = pipeline_manager.get_pipeline("test")
    result = pipeline.execute(sample_data_info)

    assert isinstance(result, DataInfo)
    assert "B" in result.data.columns
    assert all(result.data["B"] == result.data["A"] * 2)


def test_pipeline_with_multiple_steps(
    pipeline_manager, mock_transformation, sample_data_info
):
    """Test pipeline with multiple steps."""

    # Create a second transformation
    class SecondTransformation(MockTransformation):
        name = "second_transform"

        def transform(self, data_info: DataInfo) -> DataInfo:
            data = data_info.data.copy()
            data["C"] = data["B"] + 1
            metadata = {**data_info.metadata, "columns": ["A", "B", "C"]}
            return DataInfo(data=data, metadata=metadata, category=data_info.category)

    pipeline_manager.register_processor(mock_transformation)
    pipeline_manager.register_processor(SecondTransformation())
    pipeline_manager.create_pipeline(
        "test", [mock_transformation.name, "second_transform"]
    )

    pipeline = pipeline_manager.get_pipeline("test")
    result = pipeline.execute(sample_data_info)

    assert isinstance(result, DataInfo)
    assert "C" in result.data.columns
    assert all(result.data["C"] == result.data["B"] + 1)


# ============================================================================
# Quality Manager Tests
# ============================================================================


@pytest.fixture
def quality_manager():
    """Create a QualityManager instance."""
    return QualityManager()


def test_quality_manager_init():
    """Test quality manager initialization."""
    manager = QualityManager()
    assert manager.checks == []


def test_add_check(quality_manager):
    """Test adding a check."""

    def mock_check(data_info: DataInfo) -> bool:
        return True

    quality_manager.add_check(mock_check)
    assert len(quality_manager.checks) == 1


def test_run_checks(quality_manager, sample_data_info):
    """Test running checks."""

    def mock_check(data_info: DataInfo) -> bool:
        return True

    quality_manager.add_check(mock_check)
    results = quality_manager.run_checks(sample_data_info)

    assert len(results) == 1
    assert results[0]["passed"]


def test_run_checks_with_metadata(quality_manager, sample_data_info):
    """Test running checks with metadata."""

    def mock_check(data_info: DataInfo) -> bool:
        return len(data_info.metadata["columns"]) > 0

    quality_manager.add_check(mock_check)
    results = quality_manager.run_checks(sample_data_info)

    assert len(results) == 1
    assert results[0]["passed"]


def test_run_checks_with_failure(quality_manager, sample_data_info):
    """Test running checks with failure."""

    def mock_check(data_info: DataInfo) -> bool:
        return False

    quality_manager.add_check(mock_check)
    results = quality_manager.run_checks(sample_data_info)

    assert len(results) == 1
    assert not results[0]["passed"]
