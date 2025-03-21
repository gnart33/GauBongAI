from typing import Dict, List, Type, Optional
from datetime import datetime
import importlib
import pkgutil

from gaubongai.data_management.interfaces import DataInfo
from .interfaces import QualityCheck, QualityReport, QualityCheckCategory
from .checks import CompletenessCheck


class QualityManager:
    """Manages data quality assessment checks."""

    def __init__(self):
        self.checks: Dict[str, QualityCheck] = {}
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """Register built-in quality checks."""
        self.register_check(CompletenessCheck())

    def register_check(self, check: QualityCheck) -> None:
        """Register a new quality check."""
        if check.name in self.checks:
            raise ValueError(f"Check '{check.name}' already exists")
        self.checks[check.name] = check

    def get_available_checks(self) -> List[str]:
        """Get names of all registered checks."""
        return list(self.checks.keys())

    def get_checks_by_category(
        self, category: QualityCheckCategory
    ) -> List[QualityCheck]:
        """Get all checks for a specific category."""
        return [check for check in self.checks.values() if check.category == category]

    def run_check(self, data: DataInfo, check_name: str) -> QualityReport:
        """Run a specific quality check on the data."""
        if check_name not in self.checks:
            raise ValueError(f"Check '{check_name}' not found")

        check = self.checks[check_name]
        if not check.can_handle(data):
            raise ValueError(f"Check '{check_name}' cannot handle this data type")

        result = check.check(data)

        return QualityReport(
            data_info=data,
            checks_performed=[result],
            timestamp=datetime.now().isoformat(),
            metadata={"check_name": check_name, "data_category": data.category.name},
        )

    def run_all_checks(self, data: DataInfo) -> QualityReport:
        """Run all applicable quality checks on the data."""
        results = []

        for check in self.checks.values():
            if check.can_handle(data):
                try:
                    result = check.check(data)
                    results.append(result)
                except Exception as e:
                    # Log error and continue with other checks
                    print(f"Error running check {check.name}: {str(e)}")
                    continue

        return QualityReport(
            data_info=data,
            checks_performed=results,
            timestamp=datetime.now().isoformat(),
            metadata={
                "total_checks": len(results),
                "data_category": data.category.name,
            },
        )
