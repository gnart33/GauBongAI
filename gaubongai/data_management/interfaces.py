from typing import Protocol, Dict, Any, List, Union, Optional, runtime_checkable
from pathlib import Path
from enum import Enum, auto
from dataclasses import dataclass, field
import copy


class DataCategory(Enum):
    """Categories of data that can be handled."""

    TABULAR = auto()
    TEXT = auto()
    DOCUMENT = auto()
    IMAGE = auto()
    MIXED = auto()


@dataclass(frozen=True)
class DataInfo:
    data: Any  # The actual data (e.g., DataFrame, text, image)
    metadata: Dict[str, Any]  # Associated metadata
    category: DataCategory  # Type of data (TABULAR, TEXT, etc.)
    source_path: Optional[Path] = None  # Original file path

    def __post_init__(self):
        """Make deep copies of mutable attributes."""
        object.__setattr__(self, "data", copy.deepcopy(self.data))
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))


@runtime_checkable
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


@runtime_checkable
class DataTransformation(Protocol):
    """Protocol for data transformations in the pipeline."""

    name: str
    supported_categories: List[DataCategory]

    def can_transform(self, data_info: DataInfo) -> bool:
        """Check if transform can handle the data."""
        ...

    def transform(self, data_info: DataInfo) -> DataInfo:
        """Transform the data."""
        ...


@runtime_checkable
class DataValidation(Protocol):
    """Protocol for data validation."""

    name: str
    supported_categories: List[DataCategory]

    def validate(self, data_info: DataInfo) -> Dict[str, Any]:
        """Validate the data and return validation results."""
        ...


@runtime_checkable
class Pipeline(Protocol):
    """Protocol for data processing pipelines."""

    name: str
    steps: List[str]

    def execute(self, data_info: DataInfo) -> DataInfo:
        """Execute pipeline on data."""
        ...
