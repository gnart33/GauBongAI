"""Logging configuration for the project."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    module_levels: Optional[dict] = None,
    disabled: Union[bool, list] = False,
) -> None:
    """Configure logging for the project.

    Args:
        level: Default logging level for all modules
        log_file: Optional file path to write logs to
        module_levels: Optional dict of module names and their specific log levels
            e.g. {"gaubongai.data_management": "DEBUG"}
        disabled: If True, disables all logging. If a list, disables logging for specified modules
            e.g. ["gaubongai.data_management", "gaubongai.data_management.loaders"]
    """
    # Create formatters
    console_formatter = logging.Formatter("%(levelname)-8s %(name)-12s: %(message)s")
    file_formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)-12s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Setup root logger
    root_logger = logging.getLogger("gaubongai")

    # Handle disabling loggers
    if disabled is True:
        # Disable all logging
        root_logger.disabled = True
        return
    elif isinstance(disabled, (list, tuple)):
        # Disable specific modules
        for module in disabled:
            logging.getLogger(module).disabled = True

    # Continue with normal setup if not fully disabled
    root_logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Set specific levels for modules if provided
    if module_levels:
        for module, level in module_levels.items():
            logging.getLogger(module).setLevel(level)

    # Prevent logs from propagating to the root logger
    root_logger.propagate = False


def disable_all_logging():
    """Disable all logging for the project."""
    logging.getLogger("gaubongai").disabled = True


def enable_all_logging():
    """Enable all logging for the project."""
    logging.getLogger("gaubongai").disabled = False


def disable_module_logging(module_name: str):
    """Disable logging for a specific module.

    Args:
        module_name: Name of the module to disable logging for
            e.g. "gaubongai.data_management"
    """
    logging.getLogger(module_name).disabled = True


def enable_module_logging(module_name: str):
    """Enable logging for a specific module.

    Args:
        module_name: Name of the module to enable logging for
            e.g. "gaubongai.data_management"
    """
    logging.getLogger(module_name).disabled = False
