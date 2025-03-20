from typing import Dict, Type, List, Optional
from pathlib import Path
import importlib
import pkgutil
from .interfaces import DataPlugin, DataTransform, Pipeline, DataInfo, DataCategory


class PluginManager:
    """Manager for data handling plugins."""

    def __init__(self, plugin_package: str = "gauai.plugins"):
        self.plugin_package = plugin_package
        self.plugins: Dict[str, Type[DataPlugin]] = {}
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

    def get_plugin(self, file_path: Path) -> Optional[Type[DataPlugin]]:
        """Get appropriate plugin for the file."""
        return next((p for p in self.plugins.values() if p.can_handle(file_path)), None)

    def list_plugins(self) -> List[str]:
        """List all available plugins."""
        return list(self.plugins.keys())

    def get_plugins_for_category(self, category: DataCategory) -> List[str]:
        """Get plugins that handle a specific data category."""
        return [
            name
            for name, plugin in self.plugins.items()
            if plugin.data_category == category
        ]


class PipelineManager:
    """Manager for data processing pipelines."""

    def __init__(self):
        self.processors: Dict[str, DataProcessor] = {}
        self.pipelines: Dict[str, Pipeline] = {}

    def register_processor(self, processor: DataProcessor) -> None:
        """Register a new data processor."""
        self.processors[processor.name] = processor

    def register_pipeline(self, pipeline: Pipeline) -> None:
        """Register a new pipeline."""
        self.pipelines[pipeline.name] = pipeline

    def create_pipeline(self, name: str, steps: List[str]) -> None:
        """Create a new pipeline with given steps."""

        class DynamicPipeline:
            def __init__(self, pipeline_steps: List[DataProcessor]):
                self.name = name
                self.steps = [p.name for p in pipeline_steps]
                self._processors = pipeline_steps

            def execute(self, data_info: DataInfo) -> DataInfo:
                result = data_info
                for processor in self._processors:
                    if processor.can_process(result):
                        result = processor.process(result)
                return result

        # Get processors for each step
        processors = []
        for step in steps:
            processor = self.processors.get(step)
            if processor is None:
                raise ValueError(f"Processor not found: {step}")
            processors.append(processor)

        # Create and register pipeline
        pipeline = DynamicPipeline(processors)
        self.register_pipeline(pipeline)

    def get_pipeline(self, name: str) -> Optional[Pipeline]:
        """Get pipeline by name."""
        return self.pipelines.get(name)

    def list_pipelines(self) -> List[str]:
        """List all available pipelines."""
        return list(self.pipelines.keys())

    def list_processors(self) -> List[str]:
        """List all available processors."""
        return list(self.processors.keys())
