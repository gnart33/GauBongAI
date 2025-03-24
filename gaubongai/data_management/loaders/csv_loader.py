"""CSV data plugin implementations."""

from pathlib import Path
import pandas as pd
import polars as pl
import logging

logger = logging.getLogger(__name__)
from gaubongai.data_management.types import DataCategory, DataContainer, BasePlugin


class PandasCSVLoader(BasePlugin):
    """CSV plugin using pandas implementation."""

    name = "pandas_csv"
    supported_extensions = [".csv", ".tsv", ".txt"]
    data_category = DataCategory.TABULAR
    priority = 1

    def load(self, file_path: Path, **kwargs) -> DataContainer:
        """Load CSV file using pandas."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.suffix.lower() in self.supported_extensions:
            raise ValueError(f"Unsupported file extension: {file_path.suffix}")
        try:
            data = pd.read_csv(file_path, **kwargs)
            metadata = {
                "rows": len(data),
                "columns": list(data.columns),
                "dtypes": data.dtypes.astype(str).to_dict(),
                "implementation": "pandas",
            }
            return DataContainer(
                data=data, metadata=metadata, category=self.data_category
            )
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")
