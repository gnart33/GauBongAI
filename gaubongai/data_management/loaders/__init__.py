"""Data management loaders."""

from ..types import PluginManager

from .csv_loader import *

# __all__ = ["LoaderManager", "PandasCSVLoader"]


class LoaderManager(PluginManager):
    """Manager for data loading plugins."""

    def __init__(self):
        """Initialize loader manager."""
        super().__init__()


plugin_manager = LoaderManager()
__all__ = ["LoaderManager"]
for plugin in plugin_manager._plugin_registry:
    __all__.append(plugin.__name__)
    globals()[plugin.__name__] = plugin
