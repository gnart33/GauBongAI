from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import pandas as pd
import warnings

from .interfaces import DataInfo, DataCategory
from .managers import PluginManager, PipelineManager
from .storage import DataStorage


class DataProcessingManager:
    """Manages data processing using a hybrid plugin-pipeline architecture.

    This class coordinates the complete data processing workflow:
    - Loading data through plugins
    - Transforming data through pipelines
    - Storing processed data

    Attributes:
        plugin_manager: Manages data source plugins
        pipeline_manager: Manages data transformation pipelines
        data_storage: Handles data persistence
    """

    def __init__(self):
        self.plugin_manager = PluginManager()
        self.pipeline_manager = PipelineManager()
        self.data_storage = DataStorage()

    def process_file(
        self,
        file_path: Union[str, Path],
        name: Optional[str] = None,
        pipeline_name: Optional[str] = None,
        **kwargs,
    ) -> DataInfo:
        """
        Process a file using appropriate plugin and optional pipeline.

        Args:
            file_path: Path to the file to process
            name: Optional name to reference the data
            pipeline_name: Optional name of pipeline to process the data
            **kwargs: Additional arguments passed to the plugin's load method

        Returns:
            DataInfo containing the processed data and metadata
        """
        file_path = Path(file_path)
        if name is None:
            name = file_path.stem

        # Get appropriate plugin
        plugin_cls = self.plugin_manager.get_plugin(file_path)
        if plugin_cls is None:
            raise ValueError(f"No plugin found for file: {file_path}")

        # Load data using plugin
        plugin = plugin_cls()
        data_info = plugin.load(file_path, **kwargs)

        # Process through pipeline if specified
        if pipeline_name:
            pipeline = self.pipeline_manager.get_pipeline(pipeline_name)
            if pipeline is None:
                raise ValueError(f"Pipeline not found: {pipeline_name}")
            data_info = pipeline.execute(data_info)

        # Store the result
        self.data_storage.store(name, data_info)

        return data_info

    def get_data(
        self, name: str, category: Optional[DataCategory] = None
    ) -> Optional[DataInfo]:
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

    def list_plugins(self) -> List[str]:
        """List available plugins."""
        return self.plugin_manager.list_plugins()

    def list_pipelines(self) -> List[str]:
        """List available pipelines."""
        return self.pipeline_manager.list_pipelines()

    def create_pipeline(self, name: str, steps: List[str]) -> None:
        """Create a new processing pipeline."""
        self.pipeline_manager.create_pipeline(name, steps)

    def register_processor(self, processor: Any) -> None:
        """Register a new data processor."""
        self.pipeline_manager.register_processor(processor)

    # Backward compatibility for CSV loading
    def load_csv(
        self, file_path: Union[str, Path], name: Optional[str] = None, **pandas_kwargs
    ) -> pd.DataFrame:
        """Load a CSV file (maintained for backward compatibility)."""
        data_info = self.process_file(file_path, name, **pandas_kwargs)
        return data_info.data
