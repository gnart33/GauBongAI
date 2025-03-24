"""Data management loaders."""

from typing import Dict, Type, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
from ..types import PluginManager, BasePlugin
from .csv_loader import PandasCSVLoader

# Setup module logger
logger = logging.getLogger(__name__)


class Loaders(PluginManager):
    """Manager for data loading plugins."""

    def __init__(self):
        """Initialize loader manager."""
        super().__init__()

    # automate select loader when user dont specify
