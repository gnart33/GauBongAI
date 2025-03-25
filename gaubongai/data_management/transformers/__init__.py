"""Data management transformers."""

from ..types import PluginManager


class TransformerManager(PluginManager):
    """Manager for data transformation plugins."""

    def __init__(self):
        """Initialize transformer manager."""
        super().__init__()


plugin_manager = TransformerManager()
__all__ = ["TransformerManager"]
for plugin in plugin_manager._plugin_registry:
    __all__.append(plugin.__name__)
    globals()[plugin.__name__] = plugin
