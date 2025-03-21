from typing import Dict, Type, List, Optional, Tuple
from pathlib import Path
import importlib
import pkgutil
from gaubongai.data_management.interfaces import (
    DataInfo,
    DataPlugin,
    DataTransformation,
    Pipeline,
)


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
