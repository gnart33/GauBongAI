"""Data management module."""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

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

    def __init__(
        self,
        loader: Optional[BasePlugin] = None,
        transformers: Optional[BasePlugin] = None,
    ):
        self.loader = loader
        self.transformers = transformers

    def process_file(
        self,
        file_path: Path,
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

        # Load data
        data_container = self.loader.load(file_path, **kwargs)

        if isinstance(self.transformers, BasePlugin):
            self.transformers = [self.transformers]

        # Process through transformers if specified
        if self.transformers:
            for transformer in self.transformers:
                if transformer.can_transform(data_container):
                    data_container = transformer.transform(data_container)
                else:
                    raise ValueError(
                        f"Transformer {transformer.name} cannot transform the data"
                    )
        return data_container

    def process_data_container(
        self, data_container: DataContainer, **kwargs
    ) -> DataContainer:
        """Process a DataContainer using appropriate transformers.

        Args:
            data_container: DataContainer to process
        """
        if isinstance(self.transformers, BasePlugin):
            self.transformers = [self.transformers]

        if self.transformers:
            for transformer in self.transformers:
                if transformer.can_transform(data_container):
                    data_container = transformer.transform(data_container)
                else:
                    raise ValueError(
                        f"Transformer {transformer.name} cannot transform the data"
                    )
        return data_container

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
