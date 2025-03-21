import pytest
from datetime import datetime

from gaubongai.analysis.quality import (
    QualityCheckCategory,
    QualityCheckResult,
    QualityReport,
)
from gaubongai.data_management.interfaces import DataInfo, DataCategory


def test_quality_check_category():
    """Test QualityCheckCategory enum."""
    # Test all categories exist
    assert hasattr(QualityCheckCategory, "COMPLETENESS")
    assert hasattr(QualityCheckCategory, "CONSISTENCY")
    assert hasattr(QualityCheckCategory, "ACCURACY")
    assert hasattr(QualityCheckCategory, "TEMPORAL")
    assert hasattr(QualityCheckCategory, "DISTRIBUTION")

    # Test categories are unique
    categories = [cat for cat in QualityCheckCategory]
    assert len(categories) == len(set(categories))


def test_quality_check_result():
    """Test QualityCheckResult data class."""
    result = QualityCheckResult(
        check_name="test_check",
        category=QualityCheckCategory.COMPLETENESS,
        status=True,
        details={"key": "value"},
        summary="Test summary",
    )

    assert result.check_name == "test_check"
    assert result.category == QualityCheckCategory.COMPLETENESS
    assert result.status is True
    assert result.details == {"key": "value"}
    assert result.summary == "Test summary"
    assert result.visualization is None


def test_quality_report(complete_data_info):
    """Test QualityReport data class."""
    # Create sample check results
    results = [
        QualityCheckResult(
            check_name="check1",
            category=QualityCheckCategory.COMPLETENESS,
            status=True,
            details={},
            summary="Check 1 passed",
        ),
        QualityCheckResult(
            check_name="check2",
            category=QualityCheckCategory.ACCURACY,
            status=False,
            details={},
            summary="Check 2 failed",
        ),
    ]

    # Create report
    report = QualityReport(
        data_info=complete_data_info,
        checks_performed=results,
        timestamp=datetime.now().isoformat(),
        metadata={"test": "metadata"},
    )

    # Test basic attributes
    assert report.data_info == complete_data_info
    assert len(report.checks_performed) == 2
    assert isinstance(report.timestamp, str)
    assert report.metadata == {"test": "metadata"}

    # Test passed_checks property
    passed = report.passed_checks
    assert len(passed) == 1
    assert passed[0].check_name == "check1"

    # Test failed_checks property
    failed = report.failed_checks
    assert len(failed) == 1
    assert failed[0].check_name == "check2"

    # Test get_checks_by_category
    completeness_checks = report.get_checks_by_category(
        QualityCheckCategory.COMPLETENESS
    )
    assert len(completeness_checks) == 1
    assert completeness_checks[0].check_name == "check1"

    accuracy_checks = report.get_checks_by_category(QualityCheckCategory.ACCURACY)
    assert len(accuracy_checks) == 1
    assert accuracy_checks[0].check_name == "check2"

    # Test getting checks for category with no checks
    temporal_checks = report.get_checks_by_category(QualityCheckCategory.TEMPORAL)
    assert len(temporal_checks) == 0
