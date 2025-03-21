from typing import Dict, Type, List, Optional, Tuple, Any, Union
from pathlib import Path
import importlib
import pkgutil
from gaubongai.data_management.interfaces import (
    DataContainer,
    DataPlugin,
    DataTransformer,
    Pipeline,
    DataCategory,
    AnalysisContext,
)
from .storage import DataStorage
import pandas as pd


class PluginManager:
    """Manager for data source plugins."""

    def __init__(self):
        """Initialize plugin manager."""
        self.plugins: Dict[str, List[Type[DataPlugin]]] = {}
        self.plugin_package = "gaubongai.data_management.plugins"
        self._load_plugins()

    def _load_plugins(self):
        """Load plugins from package."""
        try:
            package = importlib.import_module(self.plugin_package)
            for _, name, _ in pkgutil.iter_modules(package.__path__):
                try:
                    module = importlib.import_module(f"{self.plugin_package}.{name}")
                    for item in dir(module):
                        obj = getattr(module, item)
                        if (
                            isinstance(obj, type)
                            and issubclass(obj, DataPlugin)
                            and obj != DataPlugin
                        ):
                            self.register_plugin(obj)
                except Exception as e:
                    print(f"Error loading plugin {name}: {e}")
        except Exception as e:
            print(f"Error loading plugins: {e}")

    def register_plugin(self, plugin_class: Type[DataPlugin]):
        """Register a new plugin."""
        for ext in plugin_class.supported_extensions:
            if ext not in self.plugins:
                self.plugins[ext] = []
            if plugin_class not in self.plugins[ext]:
                self.plugins[ext].append(plugin_class)
                # Sort plugins by priority (highest first)
                self.plugins[ext].sort(key=lambda p: p.priority)

    def get_plugin(
        self, file_path: Path, plugin_name: Optional[str] = None
    ) -> Optional[DataPlugin]:
        """Get appropriate plugin for file."""
        ext = file_path.suffix
        if ext not in self.plugins:
            return None

        available_plugins = self.plugins[ext]

        if plugin_name:
            # Find plugin by name
            for plugin_class in available_plugins:
                if plugin_class.name == plugin_name:
                    return plugin_class()
            raise ValueError(
                f"Plugin '{plugin_name}' not found. Available plugins: {[p.name for p in available_plugins]}"
            )

        # Return highest priority plugin
        return available_plugins[0]() if available_plugins else None

    def list_plugins(self) -> List[Tuple[str, str, int]]:
        """List all registered plugins with their names and priorities."""
        result = []
        for ext, plugins in self.plugins.items():
            for plugin in plugins:
                result.append((plugin.name, ext, plugin.priority))
        return sorted(
            result, key=lambda x: (-x[2], x[0])
        )  # Sort by priority (desc) then name


class PipelineManager:
    """Manages data processing pipelines."""

    def __init__(self):
        """Initialize the pipeline manager."""
        self._processors: Dict[str, DataTransformer] = {}
        self._pipelines: Dict[str, Pipeline] = {}

    @staticmethod
    def _execute_steps(
        data_info: DataContainer,
        steps: List[str],
        processors: Dict[str, DataTransformer],
    ) -> DataContainer:
        """Execute a sequence of processing steps on the data."""
        result = data_info
        for step in steps:
            processor = processors[step]
            if processor.can_transform(result):
                result = processor.transform(result)
        return result

    def register_processor(self, processor: DataTransformer) -> None:
        """Register a new data processor."""
        if processor.name in self._processors:
            raise ValueError(f"Processor '{processor.name}' already exists")
        self._processors[processor.name] = processor

    def create_pipeline(self, name: str, steps: List[str]) -> None:
        """Create a new pipeline."""
        if name in self._pipelines:
            raise ValueError(f"Pipeline '{name}' already exists")

        # Validate steps
        for step in steps:
            if step not in self._processors:
                raise ValueError(f"Invalid processor '{step}' in pipeline '{name}'")

        # Create pipeline
        pipeline = type(
            "DynamicPipeline",
            (),
            {
                "name": name,
                "steps": steps,
                "execute": lambda self, data_info: self._execute_pipeline(data_info),
                "_execute_pipeline": lambda self, data_info: PipelineManager._execute_steps(
                    data_info, steps, self._processors
                ),
                "_processors": self._processors,
            },
        )()

        self._pipelines[name] = pipeline

    def get_pipeline(self, name: str) -> Optional[Pipeline]:
        """Get a pipeline by name."""
        return self._pipelines.get(name)

    def list_pipelines(self) -> List[str]:
        """List all available pipelines."""
        return list(self._pipelines.keys())

    def list_processors(self) -> List[str]:
        """List all available processors."""
        return list(self._processors.keys())


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
        file_path: Path,
        name: Optional[str] = None,
        pipeline_name: Optional[str] = None,
        plugin_name: Optional[str] = None,
        **kwargs,
    ) -> DataContainer:
        """Process a file using appropriate plugin and optional pipeline.

        Args:
            file_path: Path to the file to process
            name: Optional name to reference the data
            pipeline_name: Optional name of pipeline to process the data
            plugin_name: Optional name of the plugin to use
            **kwargs: Additional arguments passed to the plugin's load method

        Returns:
            DataContainer containing the processed data and metadata
        """
        # Convert string path to Path object if needed
        if isinstance(file_path, str):
            file_path = Path(file_path)

        # Get appropriate plugin
        plugin = self.plugin_manager.get_plugin(file_path, plugin_name=plugin_name)
        if plugin is None:
            raise ValueError(f"No plugin found for file: {file_path}")

        # Load data
        data_info = plugin.load(file_path, **kwargs)

        # Add name to metadata if provided
        if name:
            data_info.metadata["name"] = name

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


class DataContextManager:
    """Manages analysis contexts with associated metadata, data, and notes."""

    def __init__(self, data_processor: DataProcessingManager):
        """Initialize with a data processor for handling data operations."""
        self._contexts: Dict[str, AnalysisContext] = {}
        self._data_processor = data_processor

    def create_context(
        self,
        name: str,
        file_path: Path = None,
        data: pd.DataFrame = None,
        metadata: Dict[str, Any] = None,
        pipeline_name: Optional[str] = None,
    ) -> None:
        """Create a new analysis context, optionally loading data from file."""
        if name in self._contexts:
            raise ValueError(f"Context '{name}' already exists")

        if file_path:
            container = self._data_processor.process_file(
                file_path, name=name, pipeline_name=pipeline_name
            )
            data = container.data
            metadata = container.metadata

        self._contexts[name] = AnalysisContext(
            data=data,
            metadata=metadata or {},
        )

    def get_context(self, name: str) -> AnalysisContext:
        """Get a context by name."""
        if name not in self._contexts:
            raise KeyError(f"Context '{name}' not found")
        return self._contexts[name]

    def update_context(
        self, name: str, data: pd.DataFrame, metadata: Dict[str, Any]
    ) -> None:
        """Update an existing context."""
        if name not in self._contexts:
            raise KeyError(f"Context '{name}' not found")
        self._contexts[name].data = data
        self._contexts[name].metadata = metadata

    def delete_context(self, name: str) -> None:
        """Delete a context and its associated notes."""
        if name not in self._contexts:
            raise KeyError(f"Context '{name}' not found")
        del self._contexts[name]

    def add_notes(self, name: str, note: str) -> None:
        """Add notes to a context."""
        if name not in self._contexts:
            raise KeyError(f"Context '{name}' not found")
        self._contexts[name].notes.append(note)

    def get_notes(self, name: str) -> List[str]:
        """Get notes for a context."""
        if name not in self._contexts:
            raise KeyError(f"Context '{name}' not found")
        return self._contexts[name].notes

    def list_contexts(self) -> List[str]:
        """List all available context names."""
        return list(self._contexts.keys())
