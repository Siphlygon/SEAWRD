from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+, tomlib is part of the standard library
except ModuleNotFoundError:
    import tomli as tomllib  # Python <=3.10 fallback, will require the `tomli` package to be installed


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load a configuration from a TOML file and validate its contents.

    Parameters
    ----------
    path : str | Path
        The path to the TOML configuration file.

    Returns
    -------
    dict[str, Any]
        The loaded and validated configuration dictionary.
    """
    path = Path(path)

    with path.open("rb") as f:
        config = tomllib.load(f)

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """
    Validate the provided configuration dictionary to ensure it contains all required sections and parameters.

    Parameters
    ----------
    config : dict[str, Any]
        The configuration dictionary to validate.

    Raises
    ------
    ValueError
        If any required section or parameter is missing or has an invalid value.
    NotImplementedError
        If the optimiser specified in the configuration is not supported.
    """
    required_sections = ["model", "training", "compile", "callbacks", "device", "output"]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing config section: [{section}]")

    model_cfg = config["model"]
    training_cfg = config["training"]
    compile_cfg = config["compile"]
    device_cfg = config["device"]

    # Simple validation checks for required parameters and their expected types/values
    if model_cfg["num_layers"] < 0:
        raise ValueError("model.num_layers must be >= 0")

    if model_cfg["num_neurons"] <= 0:
        raise ValueError("model.num_neurons must be > 0")

    if model_cfg["num_outputs"] <= 0:
        raise ValueError("model.num_outputs must be > 0")

    if training_cfg["num_epochs"] <= 0:
        raise ValueError("training.num_epochs must be > 0")

    if training_cfg["batch_size"] <= 0:
        raise ValueError("training.batch_size must be > 0")

    if not 0 < training_cfg["validation_split"] < 1:
        raise ValueError("training.validation_split must be between 0 and 1")

    if compile_cfg["optimiser"] != "adam":
        raise NotImplementedError("Only optimiser='adam' is currently supported.")

    if device_cfg["mode"] not in {"auto", "cpu", "gpu"}:
        raise ValueError("device.mode must be one of: 'auto', 'cpu', 'gpu'")
