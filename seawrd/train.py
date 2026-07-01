"""
Training entrypoint for SEAWRD.

This module stays lightweight until a device decision has been made so that CPU/GPU benchmarking can happen before
TensorFlow/Keras are imported into the main training process.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 fallback
    import tomli as tomllib

from .bootstrap import load_effective_raw_config, set_device_env
from .device_selection import choose_training_device, DeviceChoice
from .utils import load_npz_bundle, create_validation_split, fit_normaliser, get_logger

from .config import SEAWRDConfig

# Disable TensorFlow logging to avoid cluttering the output with warnings and info messages during training
# This will not stop error messages e.g., from not being able to find CUDA
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.set_stderrthreshold(absl.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Also set up back-end for keras for later
os.environ["KERAS_BACKEND"] = "tensorflow"

# Set up logging configuration for the training script
logger = get_logger("seawrd.train")



def _select_device(config: dict[str, dict[str, Any]],
                   input_features: np.ndarray,
                   input_labels: np.ndarray,
                   ) -> DeviceChoice:
    """
    Select the training device (CPU or GPU) based on the provided configuration and benchmark results before importing
    TensorFlow/Keras.
    
    If the device mode is set to "auto" and benchmarking is enabled, this function will run a benchmark to determine the
    best device for training. If benchmarking is disabled or the device mode is explicitly set to "cpu" or "gpu", the
    function will set the device accordingly without benchmarking

    Parameters
    ----------
    config : dict[str, dict[str, Any]]
        The configuration dictionary (avoiding importing SEAWRDConfig) containing device and training settings.
    input_features : np.ndarray
        The input features for training.
    input_labels : np.ndarray
        The input labels for training.

    Returns
    -------
    DeviceChoice
        The device choice object containing the selected device and benchmark results.
    """
    device_config = config.get("device", {})
    device_mode = device_config.get("mode", "auto")

    # If the device mode is explicitly set to "cpu" or "gpu", we can set the device environment variable accordingly
    if device_mode in {"cpu", "gpu"}:
        logger.info("Device mode is explicitly set to '%s' in configuration. Skipping benchmarking.", device_mode)
        device_choice = DeviceChoice(
            device=device_mode,
            reason=f"Device mode explicitly set to '{device_mode}' in configuration.")
        set_device_env(device_mode)
        return device_choice

    # If benchmarking is not enabled, we can default to CPU without running a benchmark
    if not device_config.get("benchmark_device", False):
        logger.info("Benchmarking is disabled in configuration. Defaulting to CPU.")
        device_choice = DeviceChoice(
            device="cpu",
            reason="Benchmarking is disabled in configuration; defaulting to CPU.")
        set_device_env("cpu")
        return device_choice

    logger.info("Mode is set to 'auto' and benchmarking is enabled. "
                "Running benchmark to select the best training device.")

    training_config = config.get("training", {})
    x_train, y_train, x_val, y_val = create_validation_split(
        input_features=input_features,
        input_labels=input_labels,
        validation_split=float(training_config["validation_split"]),
    )

    device_config = config.get("device", {})
    device_choice = choose_training_device(
        config=config,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        benchmark_epochs=int(device_config["benchmark_epochs"]),
        benchmark_repeats=int(device_config["benchmark_repeats"]),
        warmup_epochs=int(device_config["warmup_epochs"]),
    )

    set_device_env(device_choice.device)
    return device_choice


def run_training(config: Any,
                 input_features: np.ndarray,
                 input_labels: np.ndarray,
                 test_features: np.ndarray,
                 test_labels: np.ndarray,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The training losses, validation losses, test predictions, and test losses.
    """
    input_features = np.asarray(input_features, dtype=np.float32)
    input_labels = np.asarray(input_labels, dtype=np.float32).reshape(-1)
    test_features = np.asarray(test_features, dtype=np.float32)
    test_labels = np.asarray(test_labels, dtype=np.float32).reshape(-1)

    # Only now do we import keras by importing model or trainer
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

    return results


def _build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser for the SEAWRD training script.
    
    Adds the following arguments:
    - bundle_path: Path to a NumPy .npz bundle containing input_features, input_labels, test_features, and test_labels.
    - --config: Optional path to a custom SEAWRD TOML configuration file for benchmarking and training.

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
        help="Optional path to a custom SEAWRD TOML configuration file for benchmarking and training.",
    )
    return parser


def _load_files_from_args(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """
    Load the configuration and data bundle from the command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command-line arguments.

    Returns
    -------
    tuple[dict[str, Any], dict[str, np.ndarray]]
        The loaded configuration dictionary and data bundle.
    """
    default_config_path = Path(__file__).with_name("seawrd_default.toml")
    if args.config is not None:
        effective_raw_config = load_effective_raw_config(args.config, default_config_path)
        logger.info("Loaded custom configuration from %s.", args.config)
    else:
        with default_config_path.open("rb") as handle:
            effective_raw_config = tomllib.load(handle)
        logger.info("Loaded default configuration from %s.", default_config_path)

    bundle = load_npz_bundle(args.bundle_path)
    logger.info("Loaded data bundle from %s.", args.bundle_path)

    return effective_raw_config, bundle



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

    raw_cfg, bundle = _load_files_from_args(args)

    logger.info("Selecting training device based on configuration and benchmark results.")
    device_choice = _select_device(
        config=raw_cfg,
        input_features=bundle["input_features"],
        input_labels=bundle["input_labels"],
    )
    logger.info("Selected training device: %s.", device_choice.device)
    logger.debug("GPU result: %s", device_choice.gpu_result)
    logger.debug("CPU result: %s", device_choice.cpu_result)

    config = SEAWRDConfig.from_dict(raw_cfg)

    logger.info("Starting training.")
    run_training(
        config=config,
        input_features=bundle["input_features"],
        input_labels=bundle["input_labels"],
        test_features=bundle["test_features"],
        test_labels=bundle["test_labels"],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
