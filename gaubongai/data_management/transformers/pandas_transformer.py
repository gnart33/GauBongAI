import re
from typing import Any, List, Dict, Union, Optional
import pandas as pd
from ..types import BasePlugin, DataContainer, DataCategory


class PandasDfTransformer(BasePlugin):
    """Transform DataFrame columns using pandas native operations and type conversions.

    The transformer uses a unified column_specs dictionary where each key is the original column name
    and the value is a specification dictionary containing:
        - 'dtype': The target pandas dtype ('string', 'datetime64[ns]', 'float64', 'category', 'boolean', etc.)
        - 'convert_args': Arguments passed to the conversion function (pd.to_datetime, pd.to_numeric, etc.)
        - 'astype_args': Additional arguments passed to astype() for categorical and other types
        - 'transform': Optional transformation function or lambda to apply before type conversion
        - 'na_values': Optional list of values to treat as NA/NaN
        - 'fillna': Optional value or method to fill NA/NaN values
        - 'rename': Optional new name for the column

    Examples:
    ```python
    specs = {
        # Simple type conversion
        'price': {
            'dtype': 'float64',
            'convert_args': {'errors': 'coerce'}
        },

        # Renaming with type conversion
        'date': {
            'dtype': 'datetime64[ns]',
            'convert_args': {'format': '%Y-%m-%d', 'errors': 'coerce'},
            'rename': 'transaction_date'
        },

        # Transformation with NA handling
        'category': {
            'dtype': 'category',
            'astype_args': {'ordered': True, 'categories': ['low', 'medium', 'high']},
            'transform': str.lower,
            'na_values': ['unknown', 'n/a'],
            'fillna': 'low'
        },

        # Boolean with custom mapping and renaming
        'status': {
            'dtype': 'boolean',
            'convert_args': {
                'true_values': ['yes', 'valid', '1'],
                'false_values': ['no', 'invalid', '0']
            },
            'rename': 'is_valid'
        }
    }
    transformer = PandasDfTransformer(column_specs=specs)
    ```
    """

    name = "pandas_df_transformer"
    supported_categories = [DataCategory.TABULAR]

    def __init__(
        self,
        column_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Initialize the transformer with column specifications.

        Args:
            column_specs: Dictionary mapping column names to their conversion specifications.
                Each spec can contain:
                - 'dtype': Required. The target pandas dtype
                - 'convert_args': Optional conversion function arguments
                - 'astype_args': Optional astype arguments
                - 'transform': Optional transformation function
                - 'na_values': Optional list of values to treat as NA
                - 'fillna': Optional value or method to fill NA values
                - 'rename': Optional new name for the column
        """
        self.column_specs = column_specs or {}

    def can_transform(self, data_container: DataContainer) -> bool:
        """Check if data can be transformed, by checking
        - the data is a pandas DataFrame
        - the specified columns exist in the DataFrame"""
        if not data_container.metadata.get("category") in self.supported_categories:
            return False

        if not isinstance(data_container.data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")

        # Check if specified columns exist in the DataFrame
        df_columns = set(data_container.data.columns)
        if not set(self.column_specs.keys()).issubset(df_columns):
            raise ValueError(
                f"Columns {set(self.column_specs.keys()) - df_columns} not found in DataFrame"
            )

        return True

    def _convert_column(self, series: pd.Series, spec: Dict[str, Any]) -> pd.Series:
        """Convert a single column according to its specification."""
        # Apply pre-conversion transformation if specified
        if spec.get("remove", False):
            return None
        if "transform" in spec:
            series = series.apply(spec["transform"])

        # Handle NA values
        if "na_values" in spec:
            series = series.replace(spec["na_values"], pd.NA)

        dtype = spec["dtype"]
        convert_args = spec.get("convert_args", {})
        astype_args = spec.get("astype_args", {})

        try:
            if dtype.startswith("datetime"):
                series = pd.to_datetime(series, **convert_args)
            elif dtype in ("float", "float64", "float32", "int", "int64", "int32"):
                series = pd.to_numeric(series, **convert_args)
            elif dtype == "boolean":
                # Handle boolean conversion with custom true/false values if provided
                if "true_values" in convert_args or "false_values" in convert_args:
                    true_values = convert_args.get(
                        "true_values", ["true", "1", "yes", "y"]
                    )
                    false_values = convert_args.get(
                        "false_values", ["false", "0", "no", "n"]
                    )
                    bool_map = {val.lower(): True for val in true_values}
                    bool_map.update({val.lower(): False for val in false_values})
                    if series.dtype == "object":
                        series = series.str.lower().map(bool_map)
                series = series.astype("boolean", **astype_args)
            else:
                # For all other types (string, category, etc.), use astype directly
                series = series.astype(dtype, **astype_args)

            # Handle NA filling after conversion
            if "fillna" in spec:
                if isinstance(spec["fillna"], str) and spec["fillna"] in [
                    "ffill",
                    "bfill",
                ]:
                    series = series.fillna(method=spec["fillna"])
                else:
                    series = series.fillna(spec["fillna"])

            return series

        except Exception as e:
            if convert_args.get("errors") == "raise":
                raise ValueError(f"Error converting column to {dtype}: {str(e)}")
            return series

    def transform(self, data_container: DataContainer) -> DataContainer:
        """Transform the DataFrame by converting column data types."""
        if not self.can_transform(data_container):
            raise ValueError(
                "Data must be a pandas DataFrame with the specified columns"
            )

        df = data_container.data.copy()

        # Process each column specification
        for col, spec in self.column_specs.items():
            # Apply transformations
            transformed_series = self._convert_column(df[col], spec)

            # Handle renaming
            new_name = spec.get("rename", col)
            df[new_name] = transformed_series
            if new_name != col:  # Only drop if actually renamed
                df = df.drop(columns=[col])
        df = df.dropna(axis=1, how="all")

        # Track changes in metadata
        changes = {
            "processor": self.name,
            "transformations": {
                "column_specs": str(self.column_specs),
            },
        }

        # Update metadata
        new_metadata = data_container.metadata.copy()
        new_metadata["dtypes"] = df.dtypes.astype(str).to_dict()
        new_metadata["transformation_history"] = new_metadata.get(
            "transformation_history", []
        ) + [changes]

        return DataContainer(
            data=df,
            metadata=new_metadata,
        )
