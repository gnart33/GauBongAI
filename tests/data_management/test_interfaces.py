import pytest
from pathlib import Path
from gaubongai.data_management.interfaces import (
    DataCategory,
    DataInfo,
    DataPlugin,
    DataTransformation,
    DataValidation,
    Pipeline,
)


def test_data_category_values():
    """Test DataCategory enum values."""
    assert set(DataCategory) == {
        DataCategory.TABULAR,
        DataCategory.TEXT,
        DataCategory.DOCUMENT,
        DataCategory.IMAGE,
        DataCategory.MIXED,
    }


def test_data_info_creation():
    """Test DataInfo creation and attributes."""
    data = {"test": "data"}
    metadata = {"rows": 1, "columns": ["test"]}
    category = DataCategory.TABULAR
    source_path = Path("test.csv")

    data_info = DataInfo(
        data=data,
        metadata=metadata,
        category=category,
        source_path=source_path,
    )

    assert data_info.data == data
    assert data_info.metadata == metadata
    assert data_info.category == category
    assert data_info.source_path == source_path


def test_data_info_optional_source_path():
    """Test DataInfo creation without source_path."""
    data_info = DataInfo(
        data="test",
        metadata={},
        category=DataCategory.TEXT,
    )

    assert data_info.source_path is None


def test_data_info_immutability():
    """Test that DataInfo attributes are not modified after creation."""
    metadata = {"key": "value"}
    data = ["item1", "item2"]

    data_info = DataInfo(
        data=data,
        metadata=metadata,
        category=DataCategory.TEXT,
    )

    # Modify the original data and metadata
    metadata["new_key"] = "new_value"
    data.append("item3")

    # Check that DataInfo's copies are unchanged
    assert "new_key" not in data_info.metadata
    assert len(data_info.data) == 2


# Example implementations for protocol testing
class MockPlugin:
    name = "mock_plugin"
    supported_extensions = [".mock"]
    data_category = DataCategory.TEXT

    @classmethod
    def can_handle(cls, file_path: Path) -> bool:
        return file_path.suffix in cls.supported_extensions

    def load(self, file_path: Path, **kwargs) -> DataInfo:
        return DataInfo(data="mock_data", metadata={}, category=self.data_category)


class MockTransformation:
    name = "mock_transform"
    supported_categories = [DataCategory.TEXT]

    def can_transform(self, data_info: DataInfo) -> bool:
        return data_info.category in self.supported_categories

    def transform(self, data_info: DataInfo) -> DataInfo:
        return data_info


class MockValidation:
    name = "mock_validation"
    supported_categories = [DataCategory.TEXT]

    def validate(self, data_info: DataInfo) -> dict:
        return {"is_valid": True}


class MockPipeline:
    name = "mock_pipeline"
    steps = ["step1", "step2"]

    def execute(self, data_info: DataInfo) -> DataInfo:
        return data_info


def test_data_plugin_protocol():
    """Test that MockPlugin correctly implements DataPlugin protocol."""
    plugin = MockPlugin()
    assert isinstance(plugin, DataPlugin)
    assert hasattr(plugin, "name")
    assert hasattr(plugin, "supported_extensions")
    assert hasattr(plugin, "data_category")
    assert hasattr(plugin, "can_handle")
    assert hasattr(plugin, "load")


def test_data_transformation_protocol():
    """Test that MockTransformation correctly implements DataTransformation protocol."""
    transform = MockTransformation()
    assert isinstance(transform, DataTransformation)
    assert hasattr(transform, "name")
    assert hasattr(transform, "supported_categories")
    assert hasattr(transform, "can_transform")
    assert hasattr(transform, "transform")


def test_data_validation_protocol():
    """Test that MockValidation correctly implements DataValidation protocol."""
    validation = MockValidation()
    assert isinstance(validation, DataValidation)
    assert hasattr(validation, "name")
    assert hasattr(validation, "supported_categories")
    assert hasattr(validation, "validate")


def test_pipeline_protocol():
    """Test that MockPipeline correctly implements Pipeline protocol."""
    pipeline = MockPipeline()
    assert isinstance(pipeline, Pipeline)
    assert hasattr(pipeline, "name")
    assert hasattr(pipeline, "steps")
    assert hasattr(pipeline, "execute")
