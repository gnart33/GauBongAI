from typing import Protocol, Dict, Any, List, Union, Optional
from pathlib import Path
from enum import Enum, auto
from dataclasses import dataclass


class DataCategory(Enum):
    """Categories of data that can be handled."""

    TABULAR = auto()
    TEXT = auto()
    DOCUMENT = auto()
    IMAGE = auto()
    MIXED = auto()


@dataclass
class DataInfo:
    """Container for data and its metadata."""

    data: Any
    metadata: Dict[str, Any]
    category: DataCategory
    source_path: Optional[Path] = None


class DataPlugin(Protocol):
    """Protocol for data source plugins."""

    name: str
    supported_extensions: List[str]
    data_category: DataCategory

    @classmethod
    def can_handle(cls, file_path: Path) -> bool:
        """Check if plugin can handle the given file."""
        ...

    def load(self, file_path: Path, **kwargs) -> DataInfo:
        """Load data from the file."""
        ...


class DataTransform(Protocol):
    """Protocol for data transformations in the pipeline."""

    name: str
    supported_categories: List[DataCategory]

    def can_transform(self, data_info: DataInfo) -> bool:
        """Check if transform can handle the data."""
        ...

    def transform(self, data_info: DataInfo) -> DataInfo:
        """Transform the data."""
        ...


class DataValidator(Protocol):
    """Protocol for data validation."""

    name: str
    supported_categories: List[DataCategory]

    def validate(self, data_info: DataInfo) -> Dict[str, Any]:
        """Validate the data and return validation results."""
        ...


class Pipeline(Protocol):
    """Protocol for data processing pipelines."""

    name: str
    steps: List[str]

    def execute(self, data_info: DataInfo) -> DataInfo:
        """Execute pipeline on data."""
        ...
