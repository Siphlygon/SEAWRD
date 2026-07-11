"""
Unit tests for the train module entrypoint which allows for benchmarking device selection.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tomli_w
import pytest

from seawrd import train
from seawrd.config import SEAWRDConfig
from seawrd.device_selection import DeviceChoice
from seawrd.bootstrap import load_effective_raw_config


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


def test_load_files_from_args_returns_expected_values(tmp_path: Path):
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


def test_load_effective_raw_config_merges_defaults_and_overrides(tmp_path: Path):
    """
    Test that the load_effective_raw_config function correctly merges the default configuration with the user-provided
    configuration, allowing for non-default configurations to be used in benchmarking.

    Parameters
    ----------
    tmp_path : Path
        A temporary directory provided by pytest for creating temporary files.
    """
    default_config = {
        "device": {"mode": "auto", "min_gpu_speedup": 1.2},
        "training": {"batch_size": 32, "validation_split": 0.2},
    }
    user_config = {
        "device": {"mode": "gpu"},
        "training": {"batch_size": 64},
    }

    default_config_path = tmp_path / "default_config.toml"
    user_config_path = tmp_path / "user_config.toml"

    # Write the default and user configurations to temporary TOML files
    with default_config_path.open("wb") as f:
        tomli_w.dump(default_config, f)

    with user_config_path.open("wb") as f:
        tomli_w.dump(user_config, f)

    effective_config = load_effective_raw_config(
        config_path=user_config_path,
        default_config_path=default_config_path,
    )

    assert effective_config["device"]["mode"] == "gpu"  # User override takes precedence
    assert effective_config["device"]["min_gpu_speedup"] == 1.2  # Default value retained
    assert effective_config["training"]["batch_size"] == 64  # User override takes precedence
    assert effective_config["training"]["validation_split"] == 0.2  # Default value retained


def test_load_npz_bundle_raises_value_error_for_missing_keys(tmp_path: Path):
    """
    Test that the load_npz_bundle function raises a ValueError when the provided .npz bundle is missing required keys.
    This test creates a temporary .npz file with missing keys to simulate an invalid data bundle.

    Parameters
    ----------
    tmp_path : Path
        A temporary directory provided by pytest for creating temporary files.
    """
    # Create a temporary .npz data bundle with missing keys
    bundle_path = tmp_path / "invalid_data_bundle.npz"
    np.savez(
        bundle_path,
        input_features=np.zeros((10, 2), dtype=np.float32),
        # Missing input_labels, test_features, and test_labels
    )

    # Attempt to load the invalid bundle and check for ValueError
    with pytest.raises(ValueError) as exc_info:
        train.load_npz_bundle(bundle_path)


def _write_bundle(bundle_path: Path, with_names: bool) -> None:
    """
    Write a minimal .npz training bundle, optionally including feature/label name metadata.

    Parameters
    ----------
    bundle_path : Path
        Where to write the bundle.
    with_names : bool
        Whether to include 'feature_names' and 'label_name' arrays.
    """
    arrays = {
        "input_features": np.zeros((10, 2), dtype=np.float32),
        "input_labels": np.zeros((10,), dtype=np.float32),
        "test_features": np.zeros((5, 2), dtype=np.float32),
        "test_labels": np.zeros((5,), dtype=np.float32),
    }
    if with_names:
        arrays["feature_names"] = np.asarray(["a", "b"], dtype=np.str_)
        arrays["label_name"] = np.asarray("R_p", dtype=np.str_)

    np.savez(bundle_path, **arrays)


def test_load_npz_bundle_surfaces_feature_and_label_names(tmp_path: Path):
    """
    Test that load_npz_bundle returns feature names as a list of str and the label name as a str when present.

    Parameters
    ----------
    tmp_path : Path
        A temporary directory provided by pytest.
    """
    bundle_path = tmp_path / "bundle.npz"
    _write_bundle(bundle_path, with_names=True)

    bundle = train.load_npz_bundle(bundle_path)

    assert bundle["feature_names"] == ["a", "b"], "Feature names should be returned as a plain list of str"
    assert bundle["label_name"] == "R_p", "Label name should be returned as a plain str"


def test_load_npz_bundle_omits_names_when_absent(tmp_path: Path):
    """
    Test that load_npz_bundle omits the optional name keys for bundles that do not record them (backward compatibility).

    Parameters
    ----------
    tmp_path : Path
        A temporary directory provided by pytest.
    """
    bundle_path = tmp_path / "bundle.npz"
    _write_bundle(bundle_path, with_names=False)

    bundle = train.load_npz_bundle(bundle_path)

    assert "feature_names" not in bundle, "feature_names should be absent for legacy bundles"
    assert "label_name" not in bundle, "label_name should be absent for legacy bundles"


@pytest.mark.tf
@pytest.mark.slow
def test_run_training_writes_manifest_from_bundle_names(tmp_path: Path):
    """
    Test that run_training writes a prediction manifest when feature/label names are threaded through from the bundle.

    Parameters
    ----------
    tmp_path : Path
        A temporary directory provided by pytest.
    """
    cfg = SEAWRDConfig.from_dict({
        "model": {"num_layers": 1, "num_neurons": 2, "use_normalisation": False},
        "training": {"num_epochs": 1, "batch_size": 1, "validation_split": 0.2, "num_models": 1},
        "output": {"save_model": True, "save_plots": False, "model_dir": str(tmp_path), "version": 1},
    })

    train.run_training(
        config=cfg,
        input_features=np.zeros((10, 2), dtype=np.float32),
        input_labels=np.zeros((10,), dtype=np.float32),
        test_features=np.zeros((5, 2), dtype=np.float32),
        test_labels=np.zeros((5,), dtype=np.float32),
        feature_names=["a", "b"],
        label_name="R_p",
    )

    manifests = list(tmp_path.glob("*_manifest.json"))
    assert len(manifests) == 1, f"Expected exactly one manifest to be written, found {len(manifests)}"

    manifest = json.loads(manifests[0].read_text())
    assert manifest["feature_names"] == ["a", "b"], "Manifest should record the threaded feature names"
    assert manifest["label_name"] == "R_p", "Manifest should record the threaded label name"
