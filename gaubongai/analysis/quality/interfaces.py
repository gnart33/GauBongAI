from typing import Protocol, Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum, auto
import pandas as pd
import numpy as np

from gaubongai.data_management.interfaces import DataInfo, DataCategory


class QualityCheckCategory(Enum):
    """Categories of quality checks that can be performed."""

    COMPLETENESS = auto()  # Missing values, empty fields
    CONSISTENCY = auto()  # Data format, units
    ACCURACY = auto()  # Outliers, impossible values
    TEMPORAL = auto()  # Time series specific checks
    DISTRIBUTION = auto()  # Statistical distribution checks


@dataclass
class QualityCheckResult:
    """Results from a quality check."""

    check_name: str
    category: QualityCheckCategory
    status: bool  # True if passed, False if failed
    details: Dict[str, Any]  # Detailed results
    summary: str  # Human readable summary
    visualization: Optional[Any] = None  # Plot or visualization if applicable


class QualityCheck(Protocol):
    """Protocol for quality check plugins."""

    name: str
    category: QualityCheckCategory
    description: str

    def check(self, data: DataInfo) -> QualityCheckResult:
        """Perform the quality check on the data."""
        ...

    def can_handle(self, data: DataInfo) -> bool:
        """Check if this quality check can handle the given data."""
        ...


@dataclass
class QualityReport:
    """Comprehensive quality report for a dataset."""

    data_info: DataInfo
    checks_performed: List[QualityCheckResult]
    timestamp: str
    metadata: Dict[str, Any]

    @property
    def passed_checks(self) -> List[QualityCheckResult]:
        """Get all passed checks."""
        return [check for check in self.checks_performed if check.status]

    @property
    def failed_checks(self) -> List[QualityCheckResult]:
        """Get all failed checks."""
        return [check for check in self.checks_performed if not check.status]

    def get_checks_by_category(
        self, category: QualityCheckCategory
    ) -> List[QualityCheckResult]:
        """Get all checks for a specific category."""
        return [check for check in self.checks_performed if check.category == category]
