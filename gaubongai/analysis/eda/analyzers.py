"""Implementation of EDA analyzer plugins."""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

from gaubongai.data_management.interfaces import DataInfo, DataCategory
from gaubongai.analysis.quality.interfaces import QualityReport
from gaubongai.analysis.eda.interfaces import DataAnalyzer, AnalysisType, AnalysisResult


@dataclass
class UnivariateAnalyzer(DataAnalyzer):
    """Analyzer for univariate analysis of tabular data."""

    name: str = "univariate_analyzer"
    analysis_type: AnalysisType = AnalysisType.UNIVARIATE
    supported_categories: List[DataCategory] = field(
        default_factory=lambda: [DataCategory.TABULAR]
    )
    description: str = (
        "Performs univariate analysis on numerical and categorical columns"
    )

    def analyze(
        self, data: DataInfo, quality_report: Optional[QualityReport] = None
    ) -> AnalysisResult:
        """Perform univariate analysis on the data."""
        if not isinstance(data.data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")

        df = data.data
        stats: Dict[str, Any] = {}
        visualizations = []
        insights = []

        # Numerical analysis
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            # Basic statistics
            stats["numerical"] = {
                "summary": df[numerical_cols].describe().to_dict(),
                "missing": df[numerical_cols].isnull().sum().to_dict(),
                "skewness": df[numerical_cols].skew().to_dict(),
                "kurtosis": df[numerical_cols].kurtosis().to_dict(),
            }

            # Distribution plots for numerical variables
            for col in numerical_cols:
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                sns.histplot(data=df, x=col, ax=ax[0])
                sns.boxplot(data=df, y=col, ax=ax[1])
                ax[0].set_title(f"Distribution of {col}")
                ax[1].set_title(f"Box Plot of {col}")
                visualizations.append(fig)
                plt.close(fig)

                # Generate insights
                mean_val = df[col].mean()
                std_val = df[col].std()
                missing_pct = (df[col].isnull().sum() / len(df)) * 100

                insights.extend(
                    [
                        f"{col}: Mean = {mean_val:.2f}, Std = {std_val:.2f}",
                        f"{col}: {missing_pct:.1f}% missing values",
                        f"{col}: Distribution is {'positively' if df[col].skew() > 0 else 'negatively'} skewed",
                    ]
                )

        # Categorical analysis
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) > 0:
            # Value counts and frequencies
            stats["categorical"] = {
                col: {
                    "value_counts": df[col].value_counts().to_dict(),
                    "missing": df[col].isnull().sum(),
                    "unique_values": df[col].nunique(),
                }
                for col in categorical_cols
            }

            # Bar plots for categorical variables
            for col in categorical_cols:
                fig, ax = plt.subplots(figsize=(10, 6))
                value_counts = df[col].value_counts()
                sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
                ax.set_title(f"Value Counts for {col}")
                ax.tick_params(axis="x", rotation=45)
                visualizations.append(fig)
                plt.close(fig)

                # Generate insights
                top_category = value_counts.index[0]
                top_pct = (value_counts.iloc[0] / len(df)) * 100
                n_unique = df[col].nunique()

                insights.extend(
                    [
                        f"{col}: {n_unique} unique values",
                        f"{col}: Most common category is '{top_category}' ({top_pct:.1f}%)",
                        f"{col}: {(df[col].isnull().sum() / len(df)) * 100:.1f}% missing values",
                    ]
                )

        return AnalysisResult(
            analysis_name=self.name,
            analysis_type=self.analysis_type,
            data_info=data,
            statistics=stats,
            visualizations=visualizations,
            insights=insights,
            metadata={
                "n_numerical": len(numerical_cols),
                "n_categorical": len(categorical_cols),
                "total_variables": len(df.columns),
            },
        )
