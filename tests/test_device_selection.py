"""
Unit tests for the device selection benchmark helpers.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from seawrd import device_selection
from seawrd.device_selection import DeviceBenchmarkResult


def get_dummy_result(device: str, median_seconds: float, gpu_detected: bool = True) -> DeviceBenchmarkResult:
    """
    Create a dummy DeviceBenchmarkResult for testing purposes.

    Parameters
    ----------
    device : str
        The device type ("cpu" or "gpu").
    median_seconds : float
        The median time in seconds for the benchmark.
    gpu_detected : bool, optional
        Whether a GPU was detected, by default True.

    Returns
    -------
    DeviceBenchmarkResult
        A dummy DeviceBenchmarkResult instance.
    """
    return DeviceBenchmarkResult(
        device=device,
        times=[median_seconds],
        median_seconds=median_seconds,
        mean_seconds=median_seconds,
        std_seconds=0.0,
        steps_per_second=1.0 / median_seconds,
        seconds_per_epoch=median_seconds,
        gpu_detected=gpu_detected,
        gpu_devices=["GPU:0"] if gpu_detected else [],
    )


def get_dummy_config(min_gpu_speedup: float = 1.2, use_cache: bool = False) -> dict[str, Any]:
    """
    Create a dummy configuration dictionary for testing purposes.

    Returns
    -------
    dict[str, Any]
        A dummy configuration dictionary.
    """
    return {
        "device": {"min_gpu_speedup": min_gpu_speedup},
        "model": {"num_layers": 2},
        "training": {"batch_size": 1},
        "compile": {"metrics": ["mean_squared_error"]},
        "callbacks": {},
        "output": {"use_cache": use_cache, "cache_dir": "cache\\"},
    }


def test_choose_training_device_uses_worker_gpu_flag(monkeypatch : Any):
    """
    Test that the choose_training_device function correctly uses the GPU detection flag from the worker process to
    determine whether to select CPU or GPU for training.

    Parameters
    ----------
    monkeypatch : Any
        The pytest monkeypatch fixture, which allows us to temporarily modify the behavior of the _run_worker function
        for testing purposes.
    """
    cfg = get_dummy_config()

    calls: list[str] = []

    # Define a fake _run_worker function that simulates the behavior of the worker process for CPU and GPU benchmarking.
    def fake_run_worker(mode: str, worker_config_path):
        calls.append(mode)
        if mode == "cpu":
            return get_dummy_result(device="cpu", median_seconds=2.0, gpu_detected=True)
        return get_dummy_result(device="gpu", median_seconds=1.0, gpu_detected=False)  # Simulate GPU not detected

    # Use monkeypatch to replace the _run_worker function in the device_selection module with our fake implementation.
    monkeypatch.setattr(device_selection, "_run_worker", fake_run_worker)

    result = device_selection.choose_training_device(
        config=cfg,
        x_train=np.zeros((4, 2), dtype=np.float32),
        y_train=np.zeros((4,), dtype=np.float32),
        x_val=np.zeros((2, 2), dtype=np.float32),
        y_val=np.zeros((2,), dtype=np.float32),
        benchmark_epochs=1,
        benchmark_repeats=1,
        warmup_epochs=1,
    )

    assert calls == ["gpu"]  # so there is not a cpu call, as the gpu is not detected and cpu must be used
    assert result.device == "cpu"  # final choice should be cpu
    assert "not detect a GPU" in result.reason  # reason should mention GPU not detected


def test_choose_training_device_selects_gpu_when_fast_enough(monkeypatch : Any):
    """
    Test that the choose_training_device function selects GPU when it is faster than CPU by at least the minimum speedup
    specified in the configuration.

    Parameters
    ----------
    monkeypatch : Any
        The pytest monkeypatch fixture, which allows us to temporarily modify the behavior of the _run_worker function
        for testing purposes.
    """
    cfg = get_dummy_config(min_gpu_speedup=1.5)

    # Define a fake _run_worker function that simulates the behavior of the worker process for CPU and GPU benchmarking.
    def fake_run_worker(mode: str, worker_config_path):
        if mode == "cpu":
            return get_dummy_result(device="cpu", median_seconds=3.0)
        return get_dummy_result(device="gpu", median_seconds=1.5, gpu_detected=True)

    monkeypatch.setattr(device_selection, "_run_worker", fake_run_worker)

    result = device_selection.choose_training_device(
        config=cfg,
        x_train=np.zeros((4, 2), dtype=np.float32),
        y_train=np.zeros((4,), dtype=np.float32),
        x_val=np.zeros((2, 2), dtype=np.float32),
        y_val=np.zeros((2,), dtype=np.float32),
        benchmark_epochs=1,
        benchmark_repeats=1,
        warmup_epochs=1,
    )

    assert result.device == "gpu"
    assert "faster than CPU" in result.reason


def test_choose_training_device_selects_cpu_when_gpu_not_fast_enough(monkeypatch : Any):
    """
    Test that the choose_training_device function selects CPU when GPU is not faster than CPU by at least the minimum
    speedup specified in the configuration.

    Parameters
    ----------
    monkeypatch : Any
        The pytest monkeypatch fixture, which allows us to temporarily modify the behavior of the _run_worker function
        for testing purposes.
    """
    cfg = get_dummy_config(min_gpu_speedup=2.0)

    # Define a fake _run_worker function that simulates the behavior of the worker process for CPU and GPU benchmarking.
    def fake_run_worker(mode: str, worker_config_path):
        if mode == "cpu":
            return get_dummy_result(device="cpu", median_seconds=2.0)
        return get_dummy_result(device="gpu", median_seconds=1.5, gpu_detected=True)

    monkeypatch.setattr(device_selection, "_run_worker", fake_run_worker)

    result = device_selection.choose_training_device(
        config=cfg,
        x_train=np.zeros((4, 2), dtype=np.float32),
        y_train=np.zeros((4,), dtype=np.float32),
        x_val=np.zeros((2, 2), dtype=np.float32),
        y_val=np.zeros((2,), dtype=np.float32),
        benchmark_epochs=1,
        benchmark_repeats=1,
        warmup_epochs=1,
    )

    assert result.device == "cpu"
    assert "below the" in result.reason


def test_choose_training_device_selects_cpu_when_gpu_benchmark_fails(monkeypatch : Any):
    """
    Test that the choose_training_device function selects CPU when the GPU benchmark fails (raises an exception).

    Parameters
    ----------
    monkeypatch : Any
        The pytest monkeypatch fixture, which allows us to temporarily modify the behavior of the _run_worker function
        for testing purposes.
    """
    cfg = get_dummy_config()

    # Define a fake _run_worker function that simulates the behavior of the worker process for CPU and GPU benchmarking.
    def fake_run_worker(mode: str, worker_config_path):
        if mode == "cpu":
            return get_dummy_result(device="cpu", median_seconds=2.0)
        raise RuntimeError("GPU benchmark failed")

    monkeypatch.setattr(device_selection, "_run_worker", fake_run_worker)

    result = device_selection.choose_training_device(
        config=cfg,
        x_train=np.zeros((4, 2), dtype=np.float32),
        y_train=np.zeros((4,), dtype=np.float32),
        x_val=np.zeros((2, 2), dtype=np.float32),
        y_val=np.zeros((2,), dtype=np.float32),
        benchmark_epochs=1,
        benchmark_repeats=1,
        warmup_epochs=1,
    )

    assert result.device == "cpu"
    assert "GPU benchmark failed" in result.reason


def test_selection_reads_correct_json(monkeypatch : Any):
    """
    Test that the choose_training_device function correctly reads the JSON configuration and data files for the worker
    processes.

    Parameters
    ----------
    monkeypatch : Any
        The pytest monkeypatch fixture, which allows us to temporarily modify the behavior of the _run_worker function
        for testing purposes.
    """
    cfg = get_dummy_config()

    calls: list[str] = []

    # Define a fake _run_worker function that simulates the behavior of the worker process for CPU and GPU benchmarking.
    def fake_run_worker(mode: str, worker_config_path):
        calls.append(mode)
        with open(worker_config_path, "r") as f:
            config_data = json.load(f)
        assert "data_path" in config_data
        assert "seawrd_config" in config_data
        return get_dummy_result(device="gpu", median_seconds=1.0, gpu_detected=True)

    monkeypatch.setattr(device_selection, "_run_worker", fake_run_worker)

    device_selection.choose_training_device(
        config=cfg,
        x_train=np.zeros((4, 2), dtype=np.float32),
        y_train=np.zeros((4,), dtype=np.float32),
        x_val=np.zeros((2, 2), dtype=np.float32),
        y_val=np.zeros((2,), dtype=np.float32),
        benchmark_epochs=1,
        benchmark_repeats=1,
        warmup_epochs=1,
    )

    assert calls == ["gpu", "cpu"]


def test_benchmark_worker_config_serializes_config_and_tuning_fields():
    """
    Test that the benchmark worker payload stores the configuration as a raw mapping and serializes the benchmark
    tuning fields without loss.
    """
    cfg = get_dummy_config(min_gpu_speedup=3.0)

    payload = device_selection.BenchmarkWorkerConfig.from_config(
        config=cfg,
        data_path=Path("/tmp/benchmark_data.npz"),
        benchmark_epochs=2,
        benchmark_repeats=3,
        warmup_epochs=4,
    )

    payload_dict = payload.to_dict()

    assert payload_dict["data_path"] == "/tmp/benchmark_data.npz"
    assert payload_dict["benchmark_epochs"] == 2
    assert payload_dict["benchmark_repeats"] == 3
    assert payload_dict["warmup_epochs"] == 4
    assert payload_dict["seawrd_config"]["device"]["min_gpu_speedup"] == 3.0


def test_correct_worker_config_written(monkeypatch : Any):
    """
    Test that the choose_training_device function correctly writes the worker configuration to a JSON file for the
    worker processes.

    Parameters
    ----------
    monkeypatch : Any
        The pytest monkeypatch fixture, which allows us to temporarily modify the behavior of the _run_worker function
        for testing purposes.
    """
    cfg = get_dummy_config()

    calls: list[str] = []

    # Define a fake _run_worker function that simulates the behavior of the worker process for CPU and GPU benchmarking.
    def fake_run_worker(mode: str, worker_config_path):
        calls.append(mode)
        with open(worker_config_path, "r") as f:
            config_data = json.load(f)
        assert "data_path" in config_data
        assert "seawrd_config" in config_data
        assert config_data["benchmark_epochs"] == 2
        assert config_data["benchmark_repeats"] == 2
        assert config_data["warmup_epochs"] == 2
        return get_dummy_result(device=mode, median_seconds=1.0, gpu_detected=True)

    monkeypatch.setattr(device_selection, "_run_worker", fake_run_worker)

    device_selection.choose_training_device(
        config=cfg,
        x_train=np.zeros((4, 2), dtype=np.float32),
        y_train=np.zeros((4,), dtype=np.float32),
        x_val=np.zeros((2, 2), dtype=np.float32),
        y_val=np.zeros((2,), dtype=np.float32),
        benchmark_epochs=2,
        benchmark_repeats=2,
        warmup_epochs=2,
    )

    assert calls == ["gpu", "cpu"]


def test_correct_environment_variable_set_by_mode(monkeypatch : Any):
    """
    Test that the set_device_env function called within _run_worker correctly sets the CUDA_VISIBLE_DEVICES environment
    variable based on the specified device mode.

    Parameters
    ----------
    monkeypatch : Any
        The pytest monkeypatch fixture, which allows us to temporarily modify the behavior of the os.environ for testing
        purposes.
    """
    # Save the original environment variable to restore it later
    original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

    # Test for CPU mode
    device_selection.set_device_env("cpu")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""

    # Test for GPU mode
    device_selection.set_device_env("gpu")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"

    with pytest.raises(ValueError):
        device_selection.set_device_env("invalid_mode")

    # Restore the original environment variable
    if original_cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
    else:
        del os.environ["CUDA_VISIBLE_DEVICES"]
