"""
Unit tests for the _device_benchmark_worker module in the seawrd package. These tests verify the functionality of the
_fit_once function and the main function, ensuring that they behave as expected under various conditions, including
normal operation and error scenarios.
"""
from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from seawrd import _device_benchmark_worker
from seawrd.config import SEAWRDConfig


def test_fit_once_returns_float():
    """
    Test that the _fit_once function returns a float value representing the elapsed time in seconds for training the
    model. This test uses a simple configuration and dummy data to ensure that the function executes without errors and
    returns a valid float.
    """
    cfg = SEAWRDConfig.from_dict({"model": {"num_layers": 2}, "training": {"batch_size": 1}})
    x_train = np.zeros((4, 2), dtype=np.float32)
    y_train = np.zeros((4,), dtype=np.float32)
    x_val = np.zeros((2, 2), dtype=np.float32)
    y_val = np.zeros((2,), dtype=np.float32)

    elapsed_time = _device_benchmark_worker._fit_once(
        config=cfg,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=1,
        seed=42
    )

    assert isinstance(elapsed_time, float)


def test_main_prints_json(tmp_path: Path, monkeypatch: Any):
    """
    Test that the main function of the _device_benchmark_worker module prints a valid JSON object to stdout. This test
    uses monkeypatching to replace the sys.argv and Path.read_text methods to provide a fake configuration and data for
    the benchmark worker.

    Parameters
    ----------
    monkeypatch : Any
        The pytest monkeypatch fixture, which allows us to temporarily modify the behavior of sys.argv and
        Path.read_text for testing purposes.
    tmp_path : Path
        A temporary directory provided by pytest for creating temporary files during the test.
    """
    cfg = SEAWRDConfig.from_dict({"model": {"num_layers": 2}, "training": {"batch_size": 1}})
    fake_config_path = tmp_path / "fake_config.json"
    fake_data_path = tmp_path / "benchmark_data.npz"
    fake_config_json = json.dumps(
        {
            "data_path": str(fake_data_path),
            "seawrd_config": cfg.to_dict(),
            "benchmark_epochs": 1,
            "benchmark_repeats": 1,
            "warmup_epochs": 1,
        }
    )

    np.savez(
        fake_data_path,
        x_train=np.zeros((4, 2), dtype=np.float32),
        y_train=np.zeros((4,), dtype=np.float32),
        x_val=np.zeros((2, 2), dtype=np.float32),
        y_val=np.zeros((2,), dtype=np.float32),
    )

    # Monkeypatch sys.argv to simulate command line arguments for the benchmark worker.
    monkeypatch.setattr("sys.argv", ["_device_benchmark_worker.py", str(fake_config_path)])

    # Monkeypatch Path.read_text to return the fake configuration JSON when reading the config file.
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self: fake_config_json if str(self) == str(fake_config_path) else "",
    )

    captured_output = StringIO()
    sys.stdout = captured_output

    _device_benchmark_worker.main()

    # Restore stdout to its original state.
    sys.stdout = sys.__stdout__

    # Parse the captured output as JSON and assert that it contains the expected keys.
    output_json = json.loads(captured_output.getvalue())
    assert "gpu_detected" in output_json


def test_fit_once_with_invalid_data_raises_exception():
    """
    Test that the _fit_once function raises an exception when provided with invalid input data. This test uses a simple
    configuration and intentionally malformed data to ensure that the function correctly handles errors during model
    training.
    """
    cfg = SEAWRDConfig.from_dict({"model": {"num_layers": 2}, "training": {"batch_size": 1}})
    x_train = np.zeros((4, 2), dtype=np.float32)
    y_train = np.zeros((4,), dtype=np.float32)
    x_val = np.zeros((2, 2), dtype=np.float32)
    y_val = np.zeros((3,), dtype=np.float32)  # Invalid shape for y_val

    try:
        _device_benchmark_worker._fit_once(
            config=cfg,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            epochs=1,
            seed=42
        )
        assert False, "Expected an exception due to invalid input data, but none was raised."
    except Exception as e:
        assert isinstance(e, ValueError) or isinstance(e, RuntimeError), f"Unexpected exception type: {type(e)}"


def test_main_with_invalid_config_raises_exception(tmp_path: Path, monkeypatch: Any):
    """
    Test that the main function of the _device_benchmark_worker module raises an exception when provided with an invalid
    configuration. This test uses monkeypatching to replace the sys.argv and Path.read_text methods to provide a fake
    invalid configuration for the benchmark worker.

    Parameters
    ----------
    monkeypatch : Any
        The pytest monkeypatch fixture, which allows us to temporarily modify the behavior of sys.argv and
        Path.read_text for testing purposes.
    """
    fake_config_path = tmp_path / "fake_invalid_config.json"
    fake_data_path = tmp_path / "benchmark_data.npz"
    invalid_config_json = json.dumps(
        {
            "data_path": str(fake_data_path),
            "seawrd_config": {"model": {"num_layers": 2}, "training": {"batch_size": -1}},
            "benchmark_epochs": 1,
            "benchmark_repeats": 1,
            "warmup_epochs": 1,
        }
    )

    np.savez(
        fake_data_path,
        x_train=np.zeros((4, 2), dtype=np.float32),
        y_train=np.zeros((4,), dtype=np.float32),
        x_val=np.zeros((2, 2), dtype=np.float32),
        y_val=np.zeros((2,), dtype=np.float32),
    )

    # Monkeypatch sys.argv to simulate command line arguments for the benchmark worker.
    monkeypatch.setattr("sys.argv", ["_device_benchmark_worker.py", str(fake_config_path)])

    # Monkeypatch Path.read_text to return the invalid configuration JSON when reading the config file.
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self: invalid_config_json if str(self) == str(fake_config_path) else "",
    )

    with pytest.raises(ValueError):
        _device_benchmark_worker.main()


def test_main_with_missing_data_file_raises_exception(tmp_path: Path, monkeypatch: Any):
    """
    Test that the main function of the _device_benchmark_worker module raises an exception when the specified data file
    is missing. This test uses monkeypatching to replace the sys.argv and Path.read_text methods to provide a fake
    configuration that points to a non-existent data file.

    Parameters
    ----------
    monkeypatch : Any
        The pytest monkeypatch fixture, which allows us to temporarily modify the behavior of sys.argv and
        Path.read_text for testing purposes.
    """
    fake_config_path = tmp_path / "fake_missing_data_config.json"
    missing_data_path = tmp_path / "non_existent_data.npz"
    config_json = json.dumps(
        {
            "data_path": str(missing_data_path),
            "seawrd_config": {"model": {"num_layers": 2}, "training": {"batch_size": 1}},
            "benchmark_epochs": 1,
            "benchmark_repeats": 1,
            "warmup_epochs": 1,
        }
    )

    # Monkeypatch sys.argv to simulate command line arguments for the benchmark worker.
    monkeypatch.setattr("sys.argv", ["_device_benchmark_worker.py", str(fake_config_path)])

    # Monkeypatch Path.read_text to return the configuration JSON when reading the config file.
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self: config_json if str(self) == str(fake_config_path) else "",
    )

    with pytest.raises(FileNotFoundError):
        _device_benchmark_worker.main()


def test_main_with_invalid_json_raises_exception(tmp_path: Path, monkeypatch: Any):
    """
    Test that the main function of the _device_benchmark_worker module raises an exception when provided with an
    invalid JSON configuration. This test uses monkeypatching to replace the sys.argv and Path.read_text methods to
    provide a fake invalid JSON configuration for the benchmark worker.

    Parameters
    ----------
    monkeypatch : Any
        The pytest monkeypatch fixture, which allows us to temporarily modify the behavior of sys.argv and
        Path.read_text for testing purposes.
    """
    fake_config_path = tmp_path / "fake_invalid_json_config.json"
    invalid_json = "{invalid_json: true"  # Missing closing brace

    # Monkeypatch sys.argv to simulate command line arguments for the benchmark worker.
    monkeypatch.setattr("sys.argv", ["_device_benchmark_worker.py", str(fake_config_path)])

    # Monkeypatch Path.read_text to return the invalid JSON when reading the config file.
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self: invalid_json if str(self) == str(fake_config_path) else "",
    )

    with pytest.raises(json.JSONDecodeError):
        _device_benchmark_worker.main()
