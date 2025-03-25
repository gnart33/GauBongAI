from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type
import copy
from abc import ABC
import importlib
import pkgutil
import inspect


class DataCategory(Enum):
    """Data category enumeration."""

    TABULAR = "tabular"
    TEXT = "text"
    DOCUMENT = "document"
    IMAGE = "image"
    MIXED = "mixed"


@dataclass(frozen=True)
class DataContainer:
    """Container for data and its associated metadata, category, notes, and source information."""

    data: Any
    metadata: Dict[str, Any]

    def __post_init__(self):
        """Make deep copies of mutable attributes."""
        object.__setattr__(self, "data", copy.deepcopy(self.data))
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))


class BasePlugin(ABC):
    """Base class for all plugins in the data management system.

    This class provides common functionality for plugin registration and management.
    Specific plugin types (loaders, transformers) should inherit from this class.
    """

    name: str = ""

    @classmethod
    def get_plugin_type(cls) -> str:
        """Get the type of plugin (e.g., 'loader', 'transformer')."""
        return cls.__name__.lower()


class PluginManager(ABC):
    """Base class for plugin management."""

    def __init__(self):
        """Initialize plugin manager."""
        self.plugins: Dict[str, Type[BasePlugin]] = {}
        self._plugin_registry: List[Type[BasePlugin]] = []
        self._register_builtin_plugins()

    def register_plugin(self, plugin_class: Type[BasePlugin]) -> None:
        """Register a plugin class."""
        if not plugin_class.name:
            raise ValueError(f"Plugin {plugin_class.__name__} must have a name")

        self.plugins[plugin_class.name] = plugin_class

    def _register_builtin_plugins(self) -> None:
        """Discover and register all plugins in the package."""
        # Get the directory containing the current plugin manager implementation
        current_module = self.__class__.__module__

        try:
            # Import the package containing the plugins
            package = importlib.import_module(current_module)
            package_path = getattr(package, "__path__", [])

            # Discover all modules in the package
            for _, module_name, _ in pkgutil.iter_modules(package_path):
                full_module_name = f"{current_module}.{module_name}"

                try:
                    module = importlib.import_module(full_module_name)

                    # Find all classes in the module
                    for name, obj in inspect.getmembers(module):
                        if (
                            inspect.isclass(obj)
                            and issubclass(obj, BasePlugin)
                            and obj != BasePlugin
                            and hasattr(obj, "name")
                            and obj.name  # Must have a non-empty name
                            and obj.__module__ == module.__name__
                        ):  # Only register plugins defined in this module
                            self.register_plugin(obj)
                            self._plugin_registry.append(obj)

                except Exception as e:
                    raise e

        except Exception as e:
            raise e

    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get plugin instance by name."""
        plugin_class = self.plugins.get(plugin_name)
        if plugin_class:
            return plugin_class()
        return None

    def list_plugins(self) -> List[str]:
        """List all registered plugins."""
        plugins = self.plugins
        return plugins

    def get_plugins_by_name(self, plugin_name: str) -> List[Type[BasePlugin]]:
        """Get all plugins that match a name pattern."""
        matching_plugins = [
            plugin_class
            for plugin_class in self.plugins.values()
            if plugin_name in plugin_class.name
        ]
        return matching_plugins
