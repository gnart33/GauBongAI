import pytest
from datetime import datetime

from gaubongai.analysis.quality import (
    QualityManager,
    CompletenessCheck,
    QualityCheckCategory,
    QualityCheck,
    QualityCheckResult,
)


class MockCheck:
    """Mock quality check for testing."""

    name = "mock_check"
    category = QualityCheckCategory.ACCURACY
    description = "Mock check for testing"

    def can_handle(self, data_info):
        return True

    def check(self, data_info):
        return QualityCheckResult(
            check_name=self.name,
            category=self.category,
            status=True,
            details={},
            summary="Mock check passed",
        )


def test_quality_manager_initialization():
    """Test QualityManager initialization."""
    manager = QualityManager()
    assert isinstance(manager.checks, dict)
    assert "completeness_check" in manager.checks  # Default check should be registered


def test_register_check():
    """Test registering a new check."""
    manager = QualityManager()
    mock_check = MockCheck()

    # Register new check
    manager.register_check(mock_check)
    assert mock_check.name in manager.checks
    assert len(manager.checks) > 1  # Should have default check + new check

    # Try registering same check again
    with pytest.raises(ValueError, match="already exists"):
        manager.register_check(mock_check)


def test_get_available_checks():
    """Test getting list of available checks."""
    manager = QualityManager()
    mock_check = MockCheck()
    manager.register_check(mock_check)

    available_checks = manager.get_available_checks()
    assert isinstance(available_checks, list)
    assert "completeness_check" in available_checks
    assert "mock_check" in available_checks


def test_get_checks_by_category():
    """Test getting checks by category."""
    manager = QualityManager()
    mock_check = MockCheck()
    manager.register_check(mock_check)

    # Get completeness checks
    completeness_checks = manager.get_checks_by_category(
        QualityCheckCategory.COMPLETENESS
    )
    assert len(completeness_checks) == 1
    assert isinstance(completeness_checks[0], CompletenessCheck)

    # Get accuracy checks
    accuracy_checks = manager.get_checks_by_category(QualityCheckCategory.ACCURACY)
    assert len(accuracy_checks) == 1
    assert isinstance(accuracy_checks[0], MockCheck)


def test_run_check(complete_data_info):
    """Test running a specific check."""
    manager = QualityManager()

    # Run completeness check
    report = manager.run_check(complete_data_info, "completeness_check")
    assert len(report.checks_performed) == 1
    assert report.checks_performed[0].check_name == "completeness_check"
    assert isinstance(report.timestamp, str)

    # Try running non-existent check
    with pytest.raises(ValueError, match="not found"):
        manager.run_check(complete_data_info, "non_existent_check")


def test_run_all_checks(complete_data_info, incomplete_data_info):
    """Test running all checks."""
    manager = QualityManager()
    mock_check = MockCheck()
    manager.register_check(mock_check)

    # Test with complete data
    report = manager.run_all_checks(complete_data_info)
    assert len(report.checks_performed) == 2  # Default check + mock check
    assert all(check.status for check in report.checks_performed)  # All should pass

    # Test with incomplete data
    report = manager.run_all_checks(incomplete_data_info)
    assert len(report.checks_performed) == 2
    # Mock check passes but completeness check fails
    assert any(not check.status for check in report.checks_performed)


def test_run_check_with_invalid_data(invalid_data_info):
    """Test running checks with invalid data."""
    manager = QualityManager()

    # Running specific check
    with pytest.raises(ValueError):
        manager.run_check(invalid_data_info, "completeness_check")

    # Running all checks
    report = manager.run_all_checks(invalid_data_info)
    assert len(report.checks_performed) == 0  # No checks should be able to run
