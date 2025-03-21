from typing import Dict, Any
import pandas as pd
import polars as pl
import numpy as np
from datetime import datetime

from gaubongai.data_management.interfaces import DataInfo, DataCategory
from .interfaces import QualityCheck, QualityCheckResult, QualityCheckCategory


class CompletenessCheck:
    """Check for missing values and completeness in tabular data."""

    name = "completeness_check"
    category = QualityCheckCategory.COMPLETENESS
    description = "Analyzes missing values and completeness patterns in tabular data"

    def can_handle(self, data: DataInfo) -> bool:
        """Check if data is tabular and can be analyzed."""
        return data.category == DataCategory.TABULAR and isinstance(
            data.data, (pd.DataFrame, pd.Series, pl.DataFrame)
        )

    def check(self, data: DataInfo) -> QualityCheckResult:
        """Perform completeness analysis on tabular data."""
        if not self.can_handle(data):
            raise ValueError("Data must be tabular (DataFrame or Series)")

        df = data.data
        if isinstance(df, pl.DataFrame):
            total_cells = len(df) * len(df.columns)
            # Get null counts for each column and sum them
            null_counts = df.null_count()
            total_missing = sum(null_counts.row(0))

            # Column-wise missing value analysis
            column_stats = {}
            for column in df.columns:
                missing_count = df[column].null_count()
                missing_percentage = (missing_count / len(df)) * 100
                column_stats[column] = {
                    "missing_count": int(missing_count),
                    "missing_percentage": round(missing_percentage, 2),
                    "dtype": str(df[column].dtype),
                }
        else:
            total_cells = df.size
            total_missing = df.isna().sum().sum()

            # Column-wise missing value analysis
            column_stats = {}
            for column in df.columns:
                missing_count = df[column].isna().sum()
                missing_percentage = (missing_count / len(df)) * 100
                column_stats[column] = {
                    "missing_count": int(missing_count),
                    "missing_percentage": round(missing_percentage, 2),
                    "dtype": str(df[column].dtype),
                }

        # Overall statistics
        overall_completeness = ((total_cells - total_missing) / total_cells) * 100

        # Prepare detailed results
        details = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "total_cells": total_cells,
            "total_missing": int(total_missing),
            "overall_completeness": round(overall_completeness, 2),
            "column_stats": column_stats,
        }

        # Determine status based on completeness threshold (e.g., 95%)
        status = overall_completeness >= 95.0

        # Create human-readable summary
        summary = (
            f"Dataset Completeness Analysis:\n"
            f"- Overall completeness: {overall_completeness:.2f}%\n"
            f"- Total missing values: {total_missing:,} out of {total_cells:,} cells\n"
            f"- {len(df.columns)} columns analyzed\n"
            f"- Status: {'PASSED' if status else 'FAILED'}"
        )

        return QualityCheckResult(
            check_name=self.name,
            category=self.category,
            status=status,
            details=details,
            summary=summary,
        )
