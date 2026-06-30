"""
A module to benchmark CPU and GPU training performance and select the best device for training a deep neural network.
The module provides a function to run isolated subprocesses for benchmarking both CPU and GPU, and returns the selected
device along with the benchmark results.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

# Avoid importing from .config to keep this module lightweight, only use for type hints.
if TYPE_CHECKING:
    from .config import SEAWRDConfig


def choose_training_device(config: SEAWRDConfig,
                           x_train: np.ndarray,
                           y_train: np.ndarray,
                           x_val: np.ndarray,
                           y_val: np.ndarray,
                           benchmark_epochs: int = 10,
                           benchmark_repeats: int = 3,
                           warmup_epochs: int = 1) -> dict[str, Any]:
    """
    Benchmark CPU and GPU in isolated subprocesses and choose a backend for training based on the results.
    
    Parameters
    ----------
    config : SEAWRDConfig
        A SEAWRDConfig object containing the configuration for the model and training.
    x_train : np.ndarray
        The training input data.
    y_train : np.ndarray
        The training target data.
    x_val : np.ndarray
        The validation input data.
    y_val : np.ndarray
        The validation target data.
    benchmark_epochs : int, optional
        The number of epochs to train the model during benchmarking. Default is 10. 
    benchmark_repeats : int, optional
        The number of times to repeat the benchmark for each device. Default is 3.
    warmup_epochs : int, optional
        The number of warmup epochs to run before benchmarking. Default is 1.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the selected device, the reason for the selection, and the benchmark results for both
        CPU and GPU.
    """
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
    x_val = np.asarray(x_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32).reshape(-1)

    # Use a temporary directory to store the benchmark data and configuration for the worker processes
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_path = tmpdir / "benchmark_data.npz"
        worker_config_path = tmpdir / "worker_config.json"

        # Save the training and validation data to a .npz file for the worker processes to load
        np.savez(
            data_path,
            input_features=x_train,
            input_labels=y_train,
            test_features=x_val,
            test_labels=y_val,
        )

        # Create a worker configuration dictionary containing the paths to the data and the SEAWRD configuration, as
        # well as the benchmark parameters
        worker_config = {
            "data_path": str(data_path),
            "seawrd_config": _config_to_worker_payload(config),
            "benchmark_epochs": int(benchmark_epochs),
            "benchmark_repeats": int(benchmark_repeats),
            "warmup_epochs": int(warmup_epochs),
        }
        worker_config_path.write_text(json.dumps(worker_config))

        # Run the CPU benchmark first, then the GPU benchmark. If the GPU benchmark fails, we fall back to CPU.
        cpu_result = _run_worker("cpu", worker_config_path)

        try:
            gpu_result = _run_worker("gpu", worker_config_path)
        except Exception as exc: # default to CPU if the GPU benchmark fails
            return {
                "device": "cpu",
                "reason": f"CPU selected because the GPU benchmark failed: {exc}",
                "cpu": cpu_result,
                "gpu": None,
            }

    # Check if a GPU was detected in the GPU benchmark result. If not, we fall back to CPU.
    gpu_detected = bool(gpu_result.get("gpu_detected", gpu_result.get("gpu_devices")))
    if not gpu_detected:
        return {
            "device": "cpu",
            "reason": (
                "CPU selected because TensorFlow did not detect a GPU "
                "inside the GPU benchmark process."
            ),
            "cpu": cpu_result,
            "gpu": gpu_result,
        }

    # Compare the median training times for CPU and GPU to determine which device to use for training. If the GPU is
    # faster than the CPU by at least the minimum speedup specified in the configuration, we select the GPU. Otherwise,
    # we fall back to CPU.
    cpu_time = float(cpu_result["median_seconds"])
    gpu_time = float(gpu_result["median_seconds"])
    speedup = cpu_time / gpu_time
    min_gpu_speedup = _get_min_gpu_speedup(config)

    if speedup >= min_gpu_speedup:
        return {
            "device": "gpu",
            "reason": (
                f"GPU selected because it was {speedup:.2f}x faster than CPU, "
                f"clearing the {min_gpu_speedup:.2f}x threshold."
            ),
            "cpu": cpu_result,
            "gpu": gpu_result,
        }

    return {
        "device": "cpu",
        "reason": (
            f"CPU selected because GPU speedup was only {speedup:.2f}x, "
            f"below the {min_gpu_speedup:.2f}x threshold."
        ),
        "cpu": cpu_result,
        "gpu": gpu_result,
    }



def _run_worker(mode: str, worker_config_path: Path) -> dict[str, Any]:
    """
    Run a benchmark worker process for the specified device mode (CPU or GPU) and return the benchmark results.

    Parameters
    ----------
    mode : str
        The device mode to benchmark, either "cpu" or "gpu".
    worker_config_path : Path
        The path to the worker configuration file.

    Returns
    -------
    dict[str, Any]
        The benchmark results.

    Raises
    ------
    ValueError
        If the specified mode is not "cpu" or "gpu".
    RuntimeError
        If the worker process fails or produces no output.
    """
    env = os.environ.copy()

    # by setting CUDA_VISIBLE_DEVICES, we can control whether to use the GPU or not in the worker process.
    if mode == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif mode == "gpu":
        env["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        raise ValueError(f"Unknown benchmark mode: {mode!r}")

    # Run the benchmark worker process as a subprocess, passing the worker configuration file path as an argument.
    # Capture the output and check for errors.
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "seawrd._device_benchmark_worker",
                str(worker_config_path),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"{mode.upper()} benchmark worker failed.\n"
            f"stdout:\n{exc.stdout}\n\n"
            f"stderr:\n{exc.stderr}"
        ) from exc

    lines = [line for line in completed.stdout.splitlines() if line.strip()]

    if not lines:
        raise RuntimeError(
            f"{mode.upper()} benchmark worker produced no stdout.\n"
            f"stderr:\n{completed.stderr}"
        )

    # TensorFlow logs go to stderr; the worker prints one final JSON line to stdout.
    return json.loads(lines[-1])


def _config_to_worker_payload(config: SEAWRDConfig | dict[str, Any] | Any) -> dict[str, Any]:
    """
    Convert a configuration object to a payload for the benchmark worker. The configuration object can be a SEAWRDConfig, a dictionary, or any object that provides a `to_dict` method. The function returns a dictionary representation of the configuration that can be serialized to JSON for the worker process. This allows the benchmark worker to receive the necessary configuration for training the model without requiring the full SEAWRDConfig class or other dependencies.

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
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError(
        "config must provide to_dict() or be a mapping for benchmark serialization"
    )


def _get_min_gpu_speedup(config: SEAWRDConfig | dict[str, Any] | Any) -> float:
    """
    Get the minimum GPU speedup from the configuration object. The configuration object can be a SEAWRDConfig, a dictionary, or any object that provides a `device.min_gpu_speedup` attribute. The function returns the minimum GPU speedup as a float, which is used to determine whether to select the GPU for training based on the benchmark results.

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
    if hasattr(config, "device") and hasattr(config.device, "min_gpu_speedup"):
        return float(config.device.min_gpu_speedup)

    if isinstance(config, Mapping):
        device_section = config.get("device", {})
        if isinstance(device_section, Mapping) and "min_gpu_speedup" in device_section:
            return float(device_section["min_gpu_speedup"])

    raise TypeError("config must provide device.min_gpu_speedup for benchmarking")
