"""EDA manager for orchestrating analysis operations."""

from typing import Dict, List, Optional, Type
from datetime import datetime
import logging

from gaubongai.data_management.interfaces import DataInfo
from gaubongai.analysis.quality.interfaces import QualityReport
from gaubongai.analysis.eda.interfaces import (
    DataAnalyzer,
    EDAReport,
    AnalysisType,
    AnalysisResult,
)

logger = logging.getLogger(__name__)


class EDAManager:
    """Manager class for coordinating EDA operations."""

    def __init__(self):
        """Initialize the EDA manager."""
        self._analyzers: Dict[str, DataAnalyzer] = {}

    def register_analyzer(self, analyzer: Type[DataAnalyzer]) -> None:
        """Register a new analyzer plugin."""
        instance = analyzer()
        self._analyzers[instance.name] = instance
        logger.info(f"Registered analyzer: {instance.name}")

    def get_compatible_analyzers(self, data: DataInfo) -> List[DataAnalyzer]:
        """Get all analyzers compatible with the given data."""
        return [
            analyzer
            for analyzer in self._analyzers.values()
            if analyzer.can_analyze(data)
        ]

    def analyze(
        self,
        data: DataInfo,
        quality_report: Optional[QualityReport] = None,
        analysis_types: Optional[List[AnalysisType]] = None,
        analyzer_names: Optional[List[str]] = None,
    ) -> EDAReport:
        """
        Perform EDA on the data.

        Args:
            data: The data to analyze
            quality_report: Optional quality report to inform analysis
            analysis_types: Optional list of analysis types to perform
            analyzer_names: Optional list of specific analyzers to use

        Returns:
            An EDAReport containing all analysis results
        """
        # Filter analyzers based on compatibility and user preferences
        compatible_analyzers = self.get_compatible_analyzers(data)

        if analysis_types:
            compatible_analyzers = [
                analyzer
                for analyzer in compatible_analyzers
                if analyzer.analysis_type in analysis_types
            ]

        if analyzer_names:
            compatible_analyzers = [
                analyzer
                for analyzer in compatible_analyzers
                if analyzer.name in analyzer_names
            ]

        if not compatible_analyzers:
            logger.warning("No compatible analyzers found for the given data")
            return EDAReport(
                data_info=data,
                quality_report=quality_report,
                analyses=[],
                timestamp=datetime.now().isoformat(),
                metadata={"status": "No compatible analyzers found"},
            )

        # Perform analyses
        results: List[AnalysisResult] = []
        for analyzer in compatible_analyzers:
            try:
                logger.info(f"Running analyzer: {analyzer.name}")
                result = analyzer.analyze(data, quality_report)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in analyzer {analyzer.name}: {str(e)}")
                continue

        return EDAReport(
            data_info=data,
            quality_report=quality_report,
            analyses=results,
            timestamp=datetime.now().isoformat(),
            metadata={
                "total_analyzers": len(compatible_analyzers),
                "successful_analyses": len(results),
            },
        )
