import re
from typing import Any, List, Dict, Union, Optional
import pandas as pd
from ..types import BasePlugin, DataContainer, DataCategory


class PandasDfTransformer(BasePlugin):
    """Transform DataFrame columns to specified data types.

    By default, this transformer uses 'coerce' for error handling in numeric and datetime conversions.
    This means invalid values will be converted to NaN (for numeric) or NaT (for datetime) instead of raising errors.
    You can override this by explicitly setting 'errors' in the column options.
    """

    name = "pandas_df_transformer"
    supported_categories = [DataCategory.TABULAR]

    def __init__(
        self,
        text_columns: Optional[Dict[str, Dict]] = None,
        datetime_columns: Optional[Dict[str, Dict]] = None,
        numeric_columns: Optional[Dict[str, Dict]] = None,
        categorical_columns: Optional[Dict[str, Dict]] = None,
        boolean_columns: Optional[Dict[str, Dict]] = None,
        rename_columns: Optional[Dict[str, str]] = None,
    ):
        """Initialize the transformer with column specifications and their conversion options.

        Args:
            text_columns: Dict of columns and their string conversion options
                e.g., {'name': {'dtype': 'string', 'na': 'unknown'}}
            datetime_columns: Dict of columns and their datetime conversion options
                e.g., {'date': {'format': '%Y-%m-%d'}}
                Default error handling: errors='coerce' (invalid values become NaT)
            numeric_columns: Dict of columns and their numeric conversion options
                e.g., {'price': {'dtype': 'float64'}}
                Default error handling: errors='coerce' (invalid values become NaN)
            categorical_columns: Dict of columns and their categorical conversion options
                e.g., {'status': {'ordered': True, 'categories': ['low', 'medium', 'high']}}
            boolean_columns: Dict of columns and their boolean conversion options
                e.g., {'is_active': {'true_values': ['yes', 'true', '1'], 'false_values': ['no', 'false', '0']}}
            rename_columns: Dict mapping old column names to new column names
                e.g., {'old_name': 'new_name', 'price_eur': 'price_usd'}
        """
        self.text_columns = text_columns or {}
        self.datetime_columns = datetime_columns or {}
        self.numeric_columns = numeric_columns or {}
        self.categorical_columns = categorical_columns or {}
        self.boolean_columns = boolean_columns or {}
        self.rename_columns = rename_columns or {}

    def can_transform(self, data_container: DataContainer) -> bool:
        """Check if data can be transformed."""
        if data_container.category != DataCategory.TABULAR:
            return False

        if not isinstance(data_container.data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")

        # Check if specified columns exist in the DataFrame
        df_columns = set(data_container.data.columns)
        all_specified_columns = (
            set(self.text_columns.keys())
            | set(self.datetime_columns.keys())
            | set(self.numeric_columns.keys())
            | set(self.categorical_columns.keys())
            | set(self.boolean_columns.keys())
        )
        if not all_specified_columns.issubset(df_columns):
            raise ValueError(
                f"Columns {all_specified_columns - df_columns} not found in DataFrame"
            )

        return True

    def _convert_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to string type with options."""
        for col, options in self.text_columns.items():
            df[col] = df[col].astype(**options)
        return df

    def _convert_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to datetime using specified options."""
        for col, options in self.datetime_columns.items():
            try:
                # Add 'coerce' as default error handling if not specified
                opts = {"errors": "coerce", **options}
                df[col] = pd.to_datetime(df[col], **opts)
            except Exception as e:
                if opts.get("errors") == "raise":
                    raise ValueError(f"Error converting {col} to datetime: {str(e)}")
        return df

    def _convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to specified numeric types with options."""
        for col, options in self.numeric_columns.items():
            try:
                # Add 'coerce' as default error handling if not specified
                opts = {"errors": "coerce", **options}
                df[col] = pd.to_numeric(df[col], **opts)
            except Exception as e:
                if opts.get("errors") == "raise":
                    raise ValueError(f"Error converting {col} to numeric: {str(e)}")
        return df

    def _convert_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to categorical type with options."""
        for col, options in self.categorical_columns.items():
            df[col] = df[col].astype("category", **options)
        return df

    def _convert_boolean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to boolean type with custom true/false values."""
        for col, options in self.boolean_columns.items():
            try:
                # Get true and false values from options
                true_values = options.get("true_values", ["true", "1", "yes", "y"])
                false_values = options.get("false_values", ["false", "0", "no", "n"])

                # Create mapping dictionary
                bool_map = {val.lower(): True for val in true_values}
                bool_map.update({val.lower(): False for val in false_values})

                if df[col].dtype == "object":
                    df[col] = df[col].str.lower().map(bool_map)

                # Convert to boolean with remaining options
                bool_options = {
                    k: v
                    for k, v in options.items()
                    if k not in ["true_values", "false_values"]
                }
                df[col] = df[col].astype("boolean", **bool_options)
            except Exception as e:
                if options.get("errors") == "raise":
                    raise ValueError(f"Error converting {col} to boolean: {str(e)}")
        return df

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns according to the rename_columns mapping."""
        if self.rename_columns:
            # Only rename columns that exist in the DataFrame
            valid_renames = {
                k: v for k, v in self.rename_columns.items() if k in df.columns
            }
            if valid_renames:
                df = df.rename(columns=valid_renames)
        return df

    def transform(self, data_container: DataContainer) -> DataContainer:
        """Transform the DataFrame by converting column data types."""
        if not self.can_transform(data_container):
            raise ValueError(
                "Data must be a pandas DataFrame with the specified columns"
            )

        df = data_container.data.copy()

        # Apply transformations in sequence
        df = self._convert_text_columns(df)
        df = self._convert_datetime_columns(df)
        df = self._convert_numeric_columns(df)
        df = self._convert_categorical_columns(df)
        df = self._convert_boolean_columns(df)
        df = self._rename_columns(df)  # Apply renaming after all transformations

        # Track changes in metadata
        changes = {
            "processor": self.name,
            "transformations": {
                "text_columns": self.text_columns,
                "datetime_columns": self.datetime_columns,
                "numeric_columns": self.numeric_columns,
                "categorical_columns": self.categorical_columns,
                "boolean_columns": self.boolean_columns,
                "rename_columns": self.rename_columns,  # Add rename operations to metadata
            },
        }

        # Update metadata
        new_metadata = data_container.metadata.copy()
        new_metadata["type_conversion_history"] = new_metadata.get(
            "type_conversion_history", []
        ) + [changes]
        new_metadata["column_dtypes"] = df.dtypes.astype(str).to_dict()

        return DataContainer(
            data=df,
            metadata=new_metadata,
            category=data_container.category,
            source_path=data_container.source_path,
        )
