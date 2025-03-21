from typing import Dict, Type, List, Optional
from pathlib import Path
import importlib
import pkgutil
from gaubongai.data_management.interfaces import (
    DataCategory,
    DataInfo,
    DataPlugin,
    DataTransformation,
    Pipeline,
)


class PluginManager:
    """Manages data source plugins."""

    def __init__(self, plugin_package: str = "gaubongai.plugins"):
        self.plugin_package = plugin_package
        self.plugins: Dict[str, DataPlugin] = {}
        self._load_plugins()

    def _load_plugins(self) -> None:
        """Dynamically load all plugins from plugin package."""
        try:
            package = importlib.import_module(self.plugin_package)
            for _, name, _ in pkgutil.iter_modules(package.__path__):
                module = importlib.import_module(f"{self.plugin_package}.{name}")
                for item_name in dir(module):
                    item = getattr(module, item_name)
                    if (
                        isinstance(item, type)
                        and hasattr(item, "name")
                        and hasattr(item, "can_handle")
                    ):
                        self.plugins[item.name] = item
        except ImportError:
            # Log warning about no plugins found
            pass

    def register_plugin(self, plugin: DataPlugin) -> None:
        """Register a new plugin."""
        if plugin.name in self.plugins:
            raise ValueError(f"Plugin '{plugin.name}' already exists")
        self.plugins[plugin.name] = plugin

    def get_plugin(self, file_path: Path) -> Optional[DataPlugin]:
        """Get a plugin that can handle the given file."""
        for plugin in self.plugins.values():
            if plugin.can_handle(file_path):
                return plugin
        return None

    def list_plugins(self) -> List[str]:
        """List all available plugins."""
        return list(self.plugins.keys())

    def get_plugins_for_category(self, category: DataCategory) -> List[DataPlugin]:
        """Get all plugins that handle a specific data category."""
        return [
            plugin
            for plugin in self.plugins.values()
            if plugin.data_category == category
        ]


class PipelineManager:
    """Manages data processing pipelines."""

    def __init__(self):
        """Initialize the pipeline manager."""
        self._processors: Dict[str, DataTransformation] = {}
        self._pipelines: Dict[str, Pipeline] = {}

    @staticmethod
    def _execute_steps(
        data_info: DataInfo, steps: List[str], processors: Dict[str, DataTransformation]
    ) -> DataInfo:
        """Execute a sequence of processing steps on the data."""
        result = data_info
        for step in steps:
            processor = processors[step]
            if processor.can_transform(result):
                result = processor.transform(result)
        return result

    def register_processor(self, processor: DataTransformation) -> None:
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
