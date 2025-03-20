import re
from typing import Dict, Any
import pandas as pd
from ..interfaces import DataProcessor, DataInfo, DataCategory


class TextCleaner(DataProcessor):
    """Simple text cleaning processor."""

    name = "text_cleaner"
    supported_categories = [DataCategory.TABULAR, DataCategory.TEXT]

    def __init__(self, columns=None):
        """Initialize cleaner with optional column selection."""
        self.columns = columns

    def can_process(self, data_info: DataInfo) -> bool:
        """Check if data can be processed."""
        return data_info.category in self.supported_categories and isinstance(
            data_info.data, (pd.DataFrame, pd.Series, str)
        )

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning operations."""
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and extra whitespace
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)

        # Strip leading/trailing whitespace
        return text.strip()

    def process(self, data_info: DataInfo) -> DataInfo:
        """Process the data by cleaning text."""
        if not self.can_process(data_info):
            raise ValueError(f"Cannot process data of type {type(data_info.data)}")

        # Handle different input types
        if isinstance(data_info.data, str):
            cleaned_data = self._clean_text(data_info.data)
        elif isinstance(data_info.data, pd.Series):
            cleaned_data = data_info.data.astype(str).apply(self._clean_text)
        else:  # DataFrame
            data = data_info.data.copy()
            # Clean only specified columns or all string columns
            target_columns = (
                self.columns or data.select_dtypes(include=["object"]).columns
            )
            for col in target_columns:
                if col in data.columns:
                    data[col] = data[col].astype(str).apply(self._clean_text)
            cleaned_data = data

        # Track changes in metadata
        changes = {
            "processor": self.name,
            "columns_processed": (
                list(target_columns)
                if isinstance(cleaned_data, pd.DataFrame)
                else "all"
            ),
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
