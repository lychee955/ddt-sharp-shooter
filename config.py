"""
This module provides functions to load and dump configurations
"""

import os
import json
import tempfile
from typing import Any


def load_config(config_path: str) -> dict[str, Any] | None:
    """
    Load a configuration file from the given path.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        dict[str, Any]: The loaded configuration as a dictionary.
    """
    if not os.path.exists(config_path):
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def dump_config(config: dict[str, Any], config_path: str) -> None:
    """
    Dump a configuration dictionary to a file.

    Args:
        config (dict[str, Any]): The configuration to dump.
        config_path (str): The path to the configuration file.
    """
    # Keep the previous binding intact if writing or replacing the file fails.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False,
                                         dir=os.path.dirname(os.path.abspath(config_path))) as f:
            temporary = f.name
            json.dump(config, f, indent=4)
        os.replace(temporary, config_path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
