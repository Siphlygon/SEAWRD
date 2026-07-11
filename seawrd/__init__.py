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


# Expose configuration objects eagerly; heavier objects that pull in tensorflow/keras are loaded lazily via __getattr__
# below so that ``import seawrd`` stays cheap for config-only use.
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
    "Predictor",
]


def __getattr__(name: str):
    """
    Lazily import keras-backed objects on first access.

    Accessing ``seawrd.Predictor`` imports the predictor module (and therefore tensorflow/keras) only when it is
    actually needed, keeping a plain ``import seawrd`` lightweight for configuration-only workflows.
    """
    if name == "Predictor":
        from .predictor import Predictor

        return Predictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)