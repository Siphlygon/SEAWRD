"""
Training entrypoint for SEAWRD.

This module stays lightweight until a device decision has been made so that CPU/GPU benchmarking can happen before
TensorFlow/Keras are imported into the main training process.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Sequence, TYPE_CHECKING

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 fallback
    import tomli as tomllib

from .bootstrap import load_effective_raw_config, set_device_env
from .device_selection import choose_training_device
from .utils import load_npz_bundle, create_validation_split, fit_normaliser

if TYPE_CHECKING:
    from .config import SEAWRDConfig



def _select_device(config: SEAWRDConfig | dict[str, dict[str, Any]],
                   input_features: np.ndarray,
                   input_labels: np.ndarray,
                   ) -> tuple[str, dict[str, Any] | None]:
    """
    Select the training device (CPU or GPU) based on the provided configuration and benchmark results before importing
    TensorFlow/Keras.
    
    If the device mode is set to "auto" and benchmarking is enabled, this function will run a benchmark to determine the
    best device for training. If benchmarking is disabled or the device mode is explicitly set to "cpu" or "gpu", the
    function will set the device accordingly without benchmarking

    Parameters
    ----------
    config : SEAWRDConfig | dict[str, dict[str, Any]]
        The configuration dictionary containing device and training settings.
    input_features : np.ndarray
        The input features for training.
    input_labels : np.ndarray
        The input labels for training.

    Returns
    -------
    tuple[str, dict[str, Any] | None]
        The selected device and benchmark result.
    """
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    device_config = config.get("device", {})
    device_mode = device_config.get("mode", "auto")

    if device_mode in {"cpu", "gpu"}:
        return set_device_env(device_mode), None

    if not device_config.get("benchmark_device", False):
        return "auto", None

    training_config = config.get("training", {})
    x_train, y_train, x_val, y_val = create_validation_split(
        input_features=input_features,
        input_labels=input_labels,
        validation_split=float(training_config["validation_split"]),
    )

    benchmark_result = choose_training_device(
        config=config,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        benchmark_epochs=int(training_config["num_epochs"]),
        benchmark_repeats=int(training_config["num_models"]),
        warmup_epochs=1,
    )

    selected_device = set_device_env(benchmark_result["device"])
    return selected_device, benchmark_result


def run_training(config: Any,
                 input_features: np.ndarray,
                 input_labels: np.ndarray,
                 test_features: np.ndarray,
                 test_labels: np.ndarray,
                 *,
                 selected_device: str,
                 benchmark_result: dict[str, Any] | None,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any] | None]:
    """
    Run the full training pipeline after the training device has been selected.

    Parameters
    ----------
    config : Any
        The configuration object containing model and training settings.
    input_features : np.ndarray
        The input features for training.
    input_labels : np.ndarray
        The input labels for training.
    test_features : np.ndarray
        The test features.
    test_labels : np.ndarray
        The test labels.
    selected_device : str
        The selected training device.
    benchmark_result : dict[str, Any] | None
        The benchmark result.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any] | None]
        The training losses, validation losses, test predictions, test losses, and benchmark result.
    """
    input_features = np.asarray(input_features, dtype=np.float32)
    input_labels = np.asarray(input_labels, dtype=np.float32).reshape(-1)
    test_features = np.asarray(test_features, dtype=np.float32)
    test_labels = np.asarray(test_labels, dtype=np.float32).reshape(-1)

    # Only now do we import keras by importing model or trainer
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    from .model import DNNManager
    from .trainer import DNNTrainer

    normaliser = fit_normaliser(input_features) if config.model.use_normalisation else None

    manager = DNNManager.from_config(
        model_config=config.model,
        input_shape=input_features.shape[1:],
        normaliser=normaliser,
    )
    trainer = DNNTrainer(model_manager=manager, config=config)

    results = trainer.train_models(
        input_features=input_features,
        input_labels=input_labels,
        test_features=test_features,
        test_labels=test_labels,
    )

    print(f"Selected device: {selected_device}")
    if benchmark_result is not None:
        print(f"Benchmark reason: {benchmark_result['reason']}")

    return (*results, benchmark_result)


def _build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser for the SEAWRD training script.

    Returns
    -------
    argparse.ArgumentParser
        The CLI argument parser for the SEAWRD training script.
    """
    parser = argparse.ArgumentParser(description="Train a SEAWRD model.")
    parser.add_argument(
        "bundle_path",
        type=Path,
        help=(
            "Path to a NumPy .npz bundle containing input_features, input_labels, "
            "test_features, and test_labels."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional path to a custom SEAWRD TOML configuration file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    CLI entrypoint for the SEAWRD training script. This function handles command-line arguments, loads the configuration
    and data bundle, selects the training device, and runs the training process.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        The command-line arguments, by default None
    """
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    default_config_path = Path(__file__).with_name("seawrd_default.toml")
    if args.config is not None:
        effective_raw_config = load_effective_raw_config(args.config, default_config_path)
    else:
        with default_config_path.open("rb") as handle:
                effective_raw_config = tomllib.load(handle)
    bundle = load_npz_bundle(args.bundle_path)

    selected_device, benchmark_result = _select_device(
        config=effective_raw_config,
        input_features=bundle["input_features"],
        input_labels=bundle["input_labels"],
    )

    os.environ.setdefault("KERAS_BACKEND", "tensorflow")

    from .config import SEAWRDConfig

    config = SEAWRDConfig.from_dict(effective_raw_config)
    run_training(
        config=config,
        input_features=bundle["input_features"],
        input_labels=bundle["input_labels"],
        test_features=bundle["test_features"],
        test_labels=bundle["test_labels"],
        selected_device=selected_device,
        benchmark_result=benchmark_result,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
