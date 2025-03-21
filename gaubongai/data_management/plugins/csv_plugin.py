from pathlib import Path
import pandas as pd
from typing import Dict, Any
from ..interfaces import DataPlugin, DataInfo, DataCategory


class CSVPlugin(DataPlugin):
    """Plugin for handling CSV files."""

    name = "csv_plugin"
    supported_extensions = [".csv"]
    data_category = DataCategory.TABULAR

    @classmethod
    def can_handle(cls, file_path: Path) -> bool:
        """Check if file is a CSV."""
        return file_path.suffix.lower() in cls.supported_extensions

    def load(self, file_path: Path, **kwargs) -> DataInfo:
        """Load data from CSV file."""
        # Basic CSV reading options
        read_options = {
            "encoding": kwargs.get("encoding", "utf-8"),
            "sep": kwargs.get("separator", ","),
            "na_values": kwargs.get("na_values", ["", "NA", "N/A"]),
        }

        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Read the CSV file
            df = pd.read_csv(file_path, **read_options)

            # Basic metadata about the data
            metadata: Dict[str, Any] = {
                "rows": len(df),
                "columns": list(df.columns),
                "missing_values": df.isna().sum().to_dict(),
                "dtypes": df.dtypes.astype(str).to_dict(),
            }

            return DataInfo(
                data=df,
                metadata=metadata,
                category=self.data_category,
                source_path=file_path,
            )
        except pd.errors.ParserError as e:
            # Re-raise parser errors for malformed CSV files
            raise pd.errors.ParserError(f"Error parsing CSV file: {e}")
