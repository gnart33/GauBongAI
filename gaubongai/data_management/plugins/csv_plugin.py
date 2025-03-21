"""CSV data plugin implementations."""

from pathlib import Path
import pandas as pd
import polars as pl

from gaubongai.data_management.interfaces import (
    DataPlugin,
    DataInfo,
    DataCategory,
)


class PandasCSVPlugin(DataPlugin):
    """CSV plugin using pandas implementation."""

    name = "pandas_csv"
    supported_extensions = [".csv"]
    data_category = DataCategory.TABULAR
    priority = 1

    def load(self, file_path: Path, **kwargs) -> DataInfo:
        """Load CSV file using pandas."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            data = pd.read_csv(file_path, **kwargs)
            metadata = {
                "rows": len(data),
                "columns": list(data.columns),
                "dtypes": data.dtypes.astype(str).to_dict(),
                "implementation": "pandas",
            }
            return DataInfo(data=data, metadata=metadata, category=self.data_category)
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")


class PolarsCSVPlugin(DataPlugin):
    """Memory-efficient CSV plugin using polars implementation."""

    name = "polars_csv"
    supported_extensions = [".csv"]
    data_category = DataCategory.TABULAR
    priority = 2  # Higher priority for large files

    def load(self, file_path: Path, **kwargs) -> DataInfo:
        """Load CSV file using polars."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            data = pl.read_csv(file_path, **kwargs)
            metadata = {
                "rows": data.height,
                "columns": data.columns,
                "dtypes": {
                    col: str(dtype) for col, dtype in zip(data.columns, data.dtypes)
                },
                "implementation": "polars",
                "memory_usage": data.estimated_size(),
            }
            return DataInfo(data=data, metadata=metadata, category=self.data_category)
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")
