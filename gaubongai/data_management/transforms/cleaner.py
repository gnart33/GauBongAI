import re
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from ..interfaces import DataTransformation, DataInfo, DataCategory


class TextCleaner(DataTransformation):
    """Simple text cleaning processor."""

    name = "text_cleaner"
    supported_categories = [DataCategory.TABULAR, DataCategory.TEXT, DataCategory.MIXED]

    def __init__(self, columns=None):
        """Initialize cleaner with optional column selection."""
        self.columns = columns

    def can_transform(self, data_info: DataInfo) -> bool:
        """Check if data can be transformed."""
        if data_info.category not in self.supported_categories:
            return False

        if isinstance(data_info.data, (str, pd.Series)):
            return True

        if isinstance(data_info.data, pd.DataFrame):
            # For DataFrames, check if we have string columns to clean
            if self.columns:
                return any(col in data_info.data.columns for col in self.columns)
            return len(data_info.data.select_dtypes(include=["object"]).columns) > 0

        return False

    def _clean_text(self, text: Any) -> str:
        """Basic text cleaning operations."""
        if pd.isna(text):
            return text

        # Convert to string and lowercase
        text = str(text).lower()

        # Remove special characters and extra whitespace
        text = re.sub(r"[^\w\s.]", " ", text)  # Preserve dots for numbers
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        # If it's a number (with optional decimal point), remove spaces
        if re.match(r"^\d*\.?\d+$", text.replace(" ", "")):
            return text.replace(" ", "")

        return text

    def _get_target_columns(self, df: pd.DataFrame) -> List[str]:
        """Get columns to be processed."""
        if self.columns:
            return [col for col in self.columns if col in df.columns]
        return list(df.select_dtypes(include=["object"]).columns)

    def transform(self, data_info: DataInfo) -> DataInfo:
        """Transform the data by cleaning text."""
        if not self.can_transform(data_info):
            raise ValueError(f"Cannot transform data of type {type(data_info.data)}")

        # Handle different input types
        if isinstance(data_info.data, str):
            cleaned_data = self._clean_text(data_info.data)
            target_columns = "all"
        elif isinstance(data_info.data, pd.Series):
            cleaned_data = data_info.data.apply(self._clean_text)
            target_columns = "all"
        else:  # DataFrame
            data = data_info.data.copy()
            target_columns = self._get_target_columns(data)
            for col in target_columns:
                data[col] = data[col].apply(self._clean_text)
            cleaned_data = data

        # Track changes in metadata
        changes = {
            "processor": self.name,
            "columns_processed": target_columns,
        }

        # Update metadata
        new_metadata = data_info.metadata.copy()
        new_metadata["cleaning_history"] = new_metadata.get("cleaning_history", []) + [
            changes
        ]

        return DataInfo(
            data=cleaned_data,
            metadata=new_metadata,
            category=data_info.category,
            source_path=data_info.source_path,
        )
