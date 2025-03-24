from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Generic, TypeVar
from pathlib import Path
import copy
import pandas as pd
from abc import ABC, abstractmethod
import importlib
import pkgutil
import inspect
import logging

# Setup module logger
logger = logging.getLogger(__name__)
logger.disabled = True


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
    category: DataCategory
    source_path: Optional[Path] = None  # Original file path
    notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Make deep copies of mutable attributes."""
        object.__setattr__(self, "data", copy.deepcopy(self.data))
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))
        object.__setattr__(self, "notes", copy.deepcopy(self.notes))


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
        logger.debug("Initializing plugin manager: %s", self.__class__.__name__)
        self._register_builtin_plugins()

    def register_plugin(self, plugin_class: Type[BasePlugin]) -> None:
        """Register a plugin class."""
        if not plugin_class.name:
            raise ValueError(f"Plugin {plugin_class.__name__} must have a name")

        self.plugins[plugin_class.name] = plugin_class
        logger.debug(
            "Registered plugin: %s from %s", plugin_class.name, plugin_class.__module__
        )

    def _register_builtin_plugins(self) -> None:
        """Discover and register all plugins in the package."""
        # Get the directory containing the current plugin manager implementation
        current_module = self.__class__.__module__
        logger.debug("Scanning for plugins in module: %s", current_module)

        try:
            # Import the package containing the plugins
            package = importlib.import_module(current_module)
            package_path = getattr(package, "__path__", [])
            logger.debug("Package path: %s", package_path)

            # Discover all modules in the package
            for _, module_name, _ in pkgutil.iter_modules(package_path):
                full_module_name = f"{current_module}.{module_name}"
                logger.debug("Found module: %s", full_module_name)

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

                except Exception as e:
                    logger.error("Error loading module %s: %s", module_name, str(e))

        except Exception as e:
            logger.error("Error discovering plugins in %s: %s", current_module, str(e))

    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get plugin instance by name."""
        plugin_class = self.plugins.get(plugin_name)
        if plugin_class:
            return plugin_class()
        logger.debug("Plugin not found: %s", plugin_name)
        return None

    def list_plugins(self) -> List[str]:
        """List all registered plugins."""
        plugins = self.plugins
        logger.debug("Available plugins: %s", plugins)
        return plugins

    def get_plugins_by_name(self, plugin_name: str) -> List[Type[BasePlugin]]:
        """Get all plugins that match a name pattern."""
        matching_plugins = [
            plugin_class
            for plugin_class in self.plugins.values()
            if plugin_name in plugin_class.name
        ]
        logger.debug(
            "Found %d plugins matching pattern '%s'", len(matching_plugins), plugin_name
        )
        return matching_plugins
