"""Run EDA analysis on heart attack dataset."""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from gaubongai.data_management.interfaces import DataInfo, DataCategory
from gaubongai.analysis.eda import EDAManager, UnivariateAnalyzer, AnalysisType

# Create output directory for visualizations
output_dir = Path(__file__).parent / "eda_output"
output_dir.mkdir(exist_ok=True)

# Load the dataset
data_path = Path(__file__).parent / "heart_attack_dataset.csv"
df = pd.read_csv(data_path)

# Create DataInfo object
data_info = DataInfo(
    data=df,
    metadata={
        "description": "Heart Attack Risk Dataset",
        "source": "heart_attack_dataset.csv",
        "rows": len(df),
        "columns": len(df.columns),
    },
    category=DataCategory.TABULAR,
    source_path=data_path,
)

# Initialize and configure EDA manager
manager = EDAManager()
manager.register_analyzer(UnivariateAnalyzer)

# Run analysis
report = manager.analyze(data=data_info, analysis_types=[AnalysisType.UNIVARIATE])

# Print summary and insights
print("\n=== EDA Report Summary ===")
print(f"Total analyses performed: {report.summary['total_analyses']}")
print(f"Analysis types: {report.summary['analysis_types']}")
print("\nKey Insights:")
for insight in report.summary["key_insights"]:
    print(f"- {insight}")

# Save visualizations
if report.analyses:
    analysis = report.analyses[0]  # Get the univariate analysis
    print(f"\nSaving {len(analysis.visualizations)} visualizations to {output_dir}")

    for i, fig in enumerate(analysis.visualizations):
        output_path = output_dir / f"plot_{i+1}.png"
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"Saved: {output_path}")

# Print detailed statistics
if report.analyses:
    analysis = report.analyses[0]
    print("\n=== Detailed Statistics ===")

    if "numerical" in analysis.statistics:
        print("\nNumerical Variables Summary:")
        for col, stats in analysis.statistics["numerical"]["summary"].items():
            print(f"\n{col}:")
            for stat, value in stats.items():
                print(f"  {stat}: {value:.2f}")

    if "categorical" in analysis.statistics:
        print("\nCategorical Variables Summary:")
        for col, stats in analysis.statistics["categorical"].items():
            print(f"\n{col}:")
            print(f"  Unique values: {stats['unique_values']}")
            print(f"  Missing values: {stats['missing']}")
            print("  Top categories:")
            for category, count in list(stats["value_counts"].items())[:5]:
                print(f"    {category}: {count}")
