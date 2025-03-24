"""Data management transformers."""

from ..types import PluginManager
from .pandas_transformer import PandasDfTransformer

__all__ = ["PandasDfTransformer"]


class Transformers(PluginManager):
    """Manager for data transformation plugins."""

    def __init__(self):
        """Initialize transformer manager."""
        super().__init__()
