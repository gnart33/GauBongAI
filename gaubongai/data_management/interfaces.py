"""Data management interfaces."""

from typing import Protocol, Dict, Any, List, Union, Optional, runtime_checkable
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
import copy
import pandas as pd


class DataCategory(Enum):
    """Data category enumeration."""

    TABULAR = "tabular"
    TEXT = "text"
    DOCUMENT = "document"
    IMAGE = "image"
    MIXED = "mixed"


@dataclass(frozen=True)
class DataContainer:
    """Container for data and its associated metadata, category, notes, and source information."""

    data: Any
    metadata: Dict[str, Any]
    category: DataCategory
    source_path: Optional[Path] = None  # Original file path
    notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Make deep copies of mutable attributes."""
        object.__setattr__(self, "data", copy.deepcopy(self.data))
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))
        object.__setattr__(self, "notes", copy.deepcopy(self.notes))


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

    def load(self, file_path: Path, **kwargs) -> DataContainer:
        """Load data from file."""
        raise NotImplementedError


@runtime_checkable
class DataTransformer(Protocol):
    """Protocol for data transformations in the pipeline."""

    name: str
    supported_categories: List[DataCategory]

    def can_transform(self, data_info: DataContainer) -> bool:
        """Check if transformation can be applied to data."""
        return data_info.category in self.supported_categories

    def transform(self, data_info: DataContainer) -> DataContainer:
        """Transform data."""
        raise NotImplementedError


@runtime_checkable
class DataValidator(Protocol):
    """Protocol for data validation."""

    name: str
    supported_categories: List[DataCategory]

    def validate(self, data_info: DataContainer) -> Dict[str, Any]:
        """Validate the data and return validation results."""
        ...


@dataclass(frozen=True)
class AnalysisContext(DataContainer):
    """Specialized container for analysis contexts."""

    def __init__(
        self, data: pd.DataFrame, metadata: Dict[str, Any], notes: List[str] = None
    ):
        super().__init__(
            data=data,
            metadata=metadata,
            category=DataCategory.TABULAR,
            notes=notes or [],
        )

    @property
    def dataframe(self) -> pd.DataFrame:
        """Type-safe access to the underlying DataFrame."""
        return self.data


@runtime_checkable
class Pipeline(Protocol):
    """Protocol for data processing pipelines."""

    name: str
    steps: List[str]

    def execute(self, data_info: DataContainer) -> DataContainer:
        """Execute pipeline on data."""
        raise NotImplementedError
