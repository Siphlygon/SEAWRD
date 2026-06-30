"""
Unit tests for the train module entrypoint which allows for benchmarking device selection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from seawrd import train
from seawrd.config import SEAWRDConfig
from seawrd.device_selection import DeviceChoice


def test_select_device_auto_mode(monkeypatch: Any):
    """
    Test that the _select_device function correctly selects a device in 'auto' mode and returns a valid device string.
    This test uses monkeypatching to replace the choose_training_device function with a mock that returns a predefined
    device selection result.

    Parameters
    ----------
    monkeypatch : Any
        The pytest monkeypatch fixture, which allows us to temporarily modify the behavior of functions for testing
        purposes.
    """
    # Create dummy input data for testing
    x_train = np.zeros((10, 2), dtype=np.float32)
    y_train = np.zeros((10,), dtype=np.float32)

    # Create a dummy configuration object
    cfg = {
        "device": {
            "mode": "auto",
            "benchmark_device": True,
            "benchmark_epochs": 1,
            "benchmark_repeats": 1,
            "warmup_epochs": 0
        },
        "training": {
            "validation_split": 0.2
        }
    }

    # Define a mock function to replace choose_training_device
    def mock_choose_training_device(*args, **kwargs):
        return DeviceChoice(device="cpu", reason="mocked for testing")

    # Use monkeypatch to replace the choose_training_device function from selection with the mock
    monkeypatch.setattr(train, "choose_training_device", mock_choose_training_device)

    # Call the _select_device function and check the result
    device_choice = train._select_device(
        config=cfg,
        input_features=x_train,
        input_labels=y_train
    )

    assert device_choice.device == "cpu"
    assert device_choice.gpu_result is None
    assert device_choice.cpu_result is None


def test_select_device_manual_mode():
    """
    Test that the _select_device function correctly selects a device in 'manual' mode and returns the specified device
    string. This test does not require monkeypatching since it does not involve benchmarking.
    """
    # Create dummy input data for testing
    x_train = np.zeros((10, 2), dtype=np.float32)
    y_train = np.zeros((10,), dtype=np.float32)

    # Create a dummy configuration object with manual device selection
    cfg = {
        "device": {
            "mode": "manual",
            "benchmark_device": False,
            "benchmark_epochs": 1,
            "benchmark_repeats": 1,
            "warmup_epochs": 0,
            "manual_device": "cpu"
        },
        "training": {
            "validation_split": 0.2
        }
    }

    # Call the _select_device function and check the result
    device_choice = train._select_device(
        config=cfg,
        input_features=x_train,
        input_labels=y_train
    )

    assert device_choice.device == "cpu"
    assert device_choice.gpu_result is None
    assert device_choice.cpu_result is None


def test_run_training_returns_expected_shapes():
    """
    Test that the run_training function returns outputs with the expected shapes. This test uses dummy data and a simple
    configuration to ensure that the function executes without errors and produces outputs of the correct dimensions.
    """
    # Create dummy input data for testing
    x_train = np.zeros((10, 2), dtype=np.float32)
    y_train = np.zeros((10,), dtype=np.float32)
    x_test = np.zeros((5, 2), dtype=np.float32)
    y_test = np.zeros((5,), dtype=np.float32)

    # Create a dummy configuration object
    cfg = {
        "model": {
            "num_layers": 2,
            "num_neurons": 2
        },
        "training": {
            "num_epochs": 1,
            "batch_size": 1,
            "validation_split": 0.2,
            "num_models": 1
        }
    }
    cfg = SEAWRDConfig.from_dict(cfg)

    # Call the run_training function and check the output shapes
    train_losses, val_losses, test_predictions, test_losses = train.run_training(
        config=cfg,
        input_features=x_train,
        input_labels=y_train,
        test_features=x_test,
        test_labels=y_test,
    )

    assert isinstance(train_losses, np.ndarray)
    assert isinstance(val_losses, np.ndarray)
    assert isinstance(test_predictions, np.ndarray)
    assert isinstance(test_losses, np.ndarray)

    assert train_losses.shape[0] > 0
    assert val_losses.shape[0] > 0
    assert test_predictions.shape[0] == 1


def test_arg_parser_has_correct_arguments():
    """
    Test that the argument parser in the train module has the expected arguments. This test checks for the presence of
    specific command-line arguments that are required for running the training script.
    """
    parser = train._build_argument_parser()
    args = [action.dest for action in parser._actions]

    # Check that the expected arguments are present
    assert "config" in args
    assert "bundle_path" in args


def test_load_files_from_args_returns_expected_vaalues(tmp_path: Path):
    """
    Test that the load_files_from_args function returns the expected values for the configuration and data bundle. This
    test creates a temporary configuration file and a dummy .npz data bundle to simulate command-line arguments.

    Parameters
    ----------
    tmp_path : Path
        A temporary directory provided by pytest for creating temporary files during the test.
    """
    # Create a temporary configuration file
    config_path = tmp_path / "config.toml"
    config_content = """
            [model]
            num_layers = 2
            [training]
            batch_size = 1
            """
    config_path.write_text(config_content)

    # Create a temporary .npz data bundle
    bundle_path = tmp_path / "data_bundle.npz"
    np.savez(
        bundle_path,
        input_features=np.zeros((10, 2), dtype=np.float32),
        input_labels=np.zeros((10,), dtype=np.float32),
        test_features=np.zeros((5, 2), dtype=np.float32),
        test_labels=np.zeros((5,), dtype=np.float32),
    )

    # Create a dummy argparse.Namespace object to simulate command-line arguments
    class DummyArgs:
        def __init__(self):
            self.config = config_path
            self.bundle_path = bundle_path

    args = DummyArgs()
    config_dict, bundle = train._load_files_from_args(args)

    assert isinstance(config_dict, dict)
    assert isinstance(bundle, dict)

    # Check that the bundle contains the expected keys
    expected_keys = {"input_features", "input_labels", "test_features", "test_labels"}
    assert expected_keys.issubset(bundle.keys())

    # Check that the shapes of the loaded arrays match the expected shapes
    assert bundle["input_features"].shape == (10, 2)
    assert bundle["input_labels"].shape == (10,)
    assert bundle["test_features"].shape == (5, 2)
    assert bundle["test_labels"].shape == (5,)

    # Check that the configuration values match the expected values
    assert config_dict["model"]["num_layers"] == 2
    assert config_dict["training"]["batch_size"] == 1
