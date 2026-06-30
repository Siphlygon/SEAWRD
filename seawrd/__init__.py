"""
SEAWRD: Surrogate Emulator for Aquatic World Radius Determination.

Tools for configuring, preprocessing data for, training, and managing neural-network surrogate models for ocean-world
radius prediction.
"""

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # Python <3.8 fallback, probably unnecessary
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("seawrd")
except PackageNotFoundError:
    # Package is being used from source without installation.
    __version__ = "0+unknown"


from .config import (
    SEAWRDConfig,
    ModelConfig,
    TrainingConfig,
    CompileConfig,
    CallbackConfig,
    DeviceConfig,
    OutputConfig,
)

from .config_manager import ConfigManager


# Expose configuration objects only to avoid heavier loading of tensorflow/keras in other modules
__all__ = [
    "__version__",
    "SEAWRDConfig",
    "ModelConfig",
    "TrainingConfig",
    "CompileConfig",
    "CallbackConfig",
    "DeviceConfig",
    "OutputConfig",
    "ConfigManager",
]