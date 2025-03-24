"""Data management module."""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from .loaders import Loaders
from .transformers import Transformers
from .types import DataContainer, BasePlugin, DataCategory


class DataProcessor:
    """Manages data processing using a hybrid loader-pipeline architecture.

    This class coordinates the complete data processing workflow:
    - Loading data through loaders
    - Transforming data through pipelines
    - Storing processed data

    Attributes:
        loaders: Manages data source loaders
        transformers: Manages data transformers
        data_storage: Handles data persistence
    """

    def __init__(self):
        self.loaders = Loaders()
        self.transformers = Transformers()

    def process_file(
        self,
        file_path: Path,
        loader: Optional[BasePlugin] = None,
        transformers: Optional[Union[List[BasePlugin], BasePlugin]] = None,
        **kwargs,
    ) -> DataContainer:
        """Process a file using appropriate loader and optional transformers.

        Args:
            file_path: Path to the file to process
            name: Optional name to reference the data
            loader: Optional loader to use
            transformers: Optional transformer or list of transformers to use
            **kwargs: Additional arguments passed to the loader's load method

        Returns:
            DataContainer containing the processed data and metadata
        """
        # Convert string path to Path object if needed
        if isinstance(file_path, str):
            file_path = Path(file_path)

        # Convert single transformer to list if needed
        if transformers is not None and not isinstance(transformers, list):
            transformers = [transformers]

        # Get appropriate loader
        if loader is None:
            if loader.can_handle(file_path):
                loader = loader()
            else:
                raise ValueError(f"No loader found for file: {file_path}")

        # Load data
        data_info = loader.load(file_path, **kwargs)

        # Add name to metadata if provided
        # if name:
        #     data_info.metadata["name"] = name

        # Process through transformers if specified
        if transformers:
            for transformer in transformers:
                if transformer.can_transform(data_info):
                    data_info = transformer.transform(data_info)
                else:
                    print(
                        f"Warning: Transformer {transformer.name} cannot transform the data"
                    )

        return data_info

    def get_data(
        self, name: str, category: Optional[DataCategory] = None
    ) -> Optional[DataContainer]:
        """Retrieve stored data by name and optionally category."""
        return self.data_storage.get(name, category)

    def get_metadata(
        self, name: str, category: Optional[DataCategory] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for stored data."""
        return self.data_storage.get_metadata(name, category)

    def list_data(
        self, category: Optional[DataCategory] = None
    ) -> Union[List[str], Dict[DataCategory, List[str]]]:
        """
        List available data.
        If category is specified, returns list of names in that category.
        Otherwise, returns dict of names by category.
        """
        if category:
            return self.data_storage.list_by_category(category)
        return self.data_storage.list_all()
