"""Exploratory Data Analysis (EDA) interfaces."""

from typing import Protocol, Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import pandas as pd
import numpy as np
from pathlib import Path

from gaubongai.data_management.types import DataContainer, DataCategory
from gaubongai.analysis.quality.interfaces import QualityReport


class AnalysisType(Enum):
    """Types of exploratory analysis that can be performed."""

    UNIVARIATE = auto()  # Single variable analysis
    BIVARIATE = auto()  # Two variable relationships
    MULTIVARIATE = auto()  # Multiple variable relationships
    TEMPORAL = auto()  # Time series analysis
    CATEGORICAL = auto()  # Categorical data analysis


@dataclass
class AnalysisResult:
    """Container for analysis results."""

    analysis_name: str
    analysis_type: AnalysisType
    data_container: DataContainer
    statistics: Dict[str, Any]  # Statistical measures
    visualizations: List[Any]  # List of plot objects
    insights: List[str]  # Key findings in human readable format
    metadata: Dict[str, Any]  # Additional analysis metadata


class DataAnalyzer(Protocol):
    """Protocol for analysis plugins."""

    name: str
    analysis_type: AnalysisType
    supported_categories: List[DataCategory]
    description: str

    def analyze(
        self, data: DataContainer, quality_report: Optional[QualityReport] = None
    ) -> AnalysisResult:
        """Perform analysis on the data."""
        ...

    def can_analyze(self, data: DataContainer) -> bool:
        """Check if this analyzer can handle the given data."""
        return data.category in self.supported_categories


@dataclass
class EDAReport:
    """Comprehensive EDA report combining multiple analyses."""

    data_container: DataContainer
    quality_report: Optional[QualityReport]
    analyses: List[AnalysisResult]
    timestamp: str
    metadata: Dict[str, Any]

    def get_analyses_by_type(self, analysis_type: AnalysisType) -> List[AnalysisResult]:
        """Get all analyses of a specific type."""
        return [
            analysis
            for analysis in self.analyses
            if analysis.analysis_type == analysis_type
        ]

    def get_analysis_by_name(self, name: str) -> Optional[AnalysisResult]:
        """Get a specific analysis by name."""
        matches = [
            analysis for analysis in self.analyses if analysis.analysis_name == name
        ]
        return matches[0] if matches else None

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a high-level summary of all analyses."""
        return {
            "total_analyses": len(self.analyses),
            "analysis_types": {
                analysis_type: len(self.get_analyses_by_type(analysis_type))
                for analysis_type in AnalysisType
            },
            "key_insights": [
                insight
                for analysis in self.analyses
                for insight in analysis.insights[:3]  # Top 3 insights per analysis
            ],
        }
