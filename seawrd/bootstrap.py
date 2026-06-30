"""
hared bootstrap helpers for SEAWRD entrypoints.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 fallback
    import tomli as tomllib

if TYPE_CHECKING:
    from .config import SEAWRDConfig



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


def config_to_worker_payload(config: SEAWRDConfig | dict[str, Any] | Any) -> dict[str, Any]:
    """
    Convert a configuration object to a payload for the benchmark worker.
    
    The configuration object can be a SEAWRDConfig, a dictionary, or any object that provides a `to_dict` method. The
    function returns a dictionary representation of the configuration that can be serialized to JSON for the worker
    process. This allows the benchmark worker to receive the necessary configuration for training the model without
    requiring the full SEAWRDConfig class or other dependencies.

    Parameters
    ----------
    config : SEAWRDConfig | dict[str, Any] | Any
        The configuration object to convert.

    Returns
    -------
    dict[str, Any]
        The benchmark worker payload.

    Raises
    ------
    TypeError
        If the configuration object does not provide a `to_dict` method and is not a mapping.
    """
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    raise TypeError("config must provide to_dict() or be a mapping for benchmark serialization")


def get_min_gpu_speedup(config: SEAWRDConfig | dict[str, Any] | Any) -> float:
    """
    Get the minimum GPU speedup from the configuration object. The configuration object can be a SEAWRDConfig, a
    dictionary, or any object that provides a `device.min_gpu_speedup` attribute. The function returns the minimum GPU
    speedup as a float, which is used to determine whether to select the GPU for training based on the benchmark
    results.

    Parameters
    ----------
    config : SEAWRDConfig | dict[str, Any] | Any
        The configuration object to extract the minimum GPU speedup from.

    Returns
    -------
    float
        The minimum GPU speedup.

    Raises
    ------
    TypeError
        If the configuration object does not provide a `device.min_gpu_speedup` attribute and is not a mapping.
    """
    if isinstance(config, Mapping):
        device_section = config.get("device", {})
        if isinstance(device_section, Mapping) and "min_gpu_speedup" in device_section:
            return float(device_section["min_gpu_speedup"])

    device_section = getattr(config, "device", None)
    if device_section is not None and hasattr(device_section, "min_gpu_speedup"):
        return float(device_section.min_gpu_speedup)

    raise TypeError("config must provide device.min_gpu_speedup for benchmarking")


def get_cpu_name() -> str:
    """
    Get the CPU name based on the current platform.

    The most common method (platform.preprocessor()) often does not work properly for Linux, so this function has
    platform-specific implementations to retrieve the CPU name or description.

    Returns
    -------
    str
        The CPU name or description, or an empty string if it cannot be determined.
    """
    if platform.system() == "Windows":
        return platform.processor()
    if platform.system() == "Darwin":
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"]
        ).strip().decode()
    if platform.system() == "Linux":
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    return re.sub(".*model name.*:", "", line, 1).strip()
    return platform.processor()


def get_gpu_names() -> list[str]:
    """
    Get the names of any GPUs based on the current platform. This function uses platform-specific commands to retrieve
    the names of the GPUs available on the system. It supports Windows, Linux, and macOS

    Returns
    -------
    list[str]
        A list of GPU names or an empty list if no GPUs are found or the platform is unsupported.
    """
    system = platform.system()
    if system == "Windows":
        try:
            output = subprocess.check_output(["wmic", "path", "win32_VideoController", "get", "name"],
                stderr=subprocess.DEVNULL
            ).decode()
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            return lines[1:]  # skip the "Name" header
        except Exception:  # wmic is deprecated on newer Windows builds; fall back to PowerShell
            output = subprocess.check_output(
                ["powershell", "-Command",
                 "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"]
            ).decode()
            return [line.strip() for line in output.splitlines() if line.strip()]

    if system == "Linux":
        try:
            output = subprocess.check_output(
                "lspci | grep -i 'vga\\|3d\\|2d'", shell=True
            ).decode()
            return [line.strip() for line in output.splitlines()]
        except subprocess.CalledProcessError:  # sometimes lspci is not available or fails; return an empty list
            return []


    if system == "Darwin":
        output = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"]
        ).decode()
        names = [
            line.split(":")[1].strip()
            for line in output.splitlines()
            if "Chipset Model" in line
        ]
        return names

    return []
