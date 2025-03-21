"""Data management interfaces."""

from typing import Protocol, Dict, Any, List, Union, Optional, runtime_checkable
from pathlib import Path
from enum import Enum, auto
from dataclasses import dataclass, field
import copy


class DataCategory(Enum):
    """Data category enumeration."""

    TABULAR = "tabular"
    TEXT = "text"
    DOCUMENT = "document"
    IMAGE = "image"
    MIXED = "mixed"


@dataclass(frozen=True)
class DataInfo:
    """Data information container."""

    data: Any
    metadata: Dict[str, Any]
    category: DataCategory
    source_path: Optional[Path] = None  # Original file path

    def __post_init__(self):
        """Make deep copies of mutable attributes."""
        object.__setattr__(self, "data", copy.deepcopy(self.data))
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))


class DataPlugin:
    """Base class for data plugins."""

    name: str
    supported_extensions: List[str]
    data_category: DataCategory
    priority: int = 1

    @classmethod
    def can_handle(cls, file_path: Path) -> bool:
        """Check if plugin can handle the given file."""
        return file_path.suffix in cls.supported_extensions

    def load(self, file_path: Path, **kwargs) -> DataInfo:
        """Load data from file."""
        raise NotImplementedError


@runtime_checkable
class DataTransformation(Protocol):
    """Protocol for data transformations in the pipeline."""

    name: str
    supported_categories: List[DataCategory]

    def can_transform(self, data_info: DataInfo) -> bool:
        """Check if transformation can be applied to data."""
        return data_info.category in self.supported_categories

    def transform(self, data_info: DataInfo) -> DataInfo:
        """Transform data."""
        raise NotImplementedError


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
        raise NotImplementedError
