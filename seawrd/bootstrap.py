"""
hared bootstrap helpers for SEAWRD entrypoints.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 fallback
    import tomli as tomllib



def _merge_config(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """
    Merge two configuration dictionaries, with values from the overrides taking precedence over the defaults. Any nested
    dictionaries are merged recursively.

    Parameters
    ----------
    defaults : dict[str, Any]
        The default configuration values.
    overrides : dict[str, Any]
        The override configuration values.

    Returns
    -------
    dict[str, Any]
        The merged configuration values.
    """
    merged: dict[str, Any] = dict(defaults)

    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value

    return merged


def load_effective_raw_config(config_path: Path, default_config_path: Path) -> dict[str, Any]:
    """
    Load the effective configuration by merging the default configuration with the user-provided configuration. Allows
    for non-default configurations to be used in benchmarking.

    Parameters
    ----------
    config_path : Path
        Path to the user-provided configuration file.
    default_config_path : Path
        Path to the default configuration file.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the effective configuration values after merging defaults and overrides.
    """
    with config_path.open("rb") as handle:
        overrides = tomllib.load(handle)
    with default_config_path.open("rb") as handle:
        defaults = tomllib.load(handle)
    return _merge_config(defaults, overrides)


def set_device_env(device_mode: str) -> str:
    """
    Set CUDA visibility for the selected training device.

    Parameters
    ----------
    device_mode : str
        The mode in which to run the training (either "cpu" or "gpu").

    Returns
    -------
    str
        The device mode that was set.
    """
    if device_mode == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif device_mode == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        raise ValueError(f"Unknown device mode: {device_mode!r}")
    return device_mode
