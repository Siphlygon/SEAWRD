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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:
    from .config import SEAWRDConfig

from .bootstrap import set_device_env
from .utils import get_logger

logger = get_logger("seawrd.device_selection")


@dataclass(frozen=True)
class DeviceBenchmarkResult:
    """
    A dataclass to hold the benchmark results for a specific device (CPU or GPU).
    
    It contains the device name, median, mean, and standard deviation of the training times, as well as the final
    validation loss and steps per second achieved during the benchmark. This class is used to encapsulate the results 
    the benchmarking process for easy comparison and selection of the best device for training.
    """
    device: str

    # Numerical results
    times : list[float]
    median_seconds: float
    mean_seconds: float
    std_seconds: float
    steps_per_second: float
    seconds_per_epoch: float

    # Information
    gpu_detected: bool | None = None
    gpu_devices: list[str] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeviceBenchmarkResult:
        """
        Create a DeviceBenchmarkResult instance from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            A dictionary containing the benchmark results for a device.

        Returns
        -------
        DeviceBenchmarkResult
            An instance of DeviceBenchmarkResult populated with the data from the dictionary.
        """
        return cls(
            device=data["device"],  # fail early if no device is present
            times=data.get("times", []),
            median_seconds=float(data.get("median_seconds", 0.0)),
            mean_seconds=float(data.get("mean_seconds", 0.0)),
            std_seconds=float(data.get("std_seconds", 0.0)),
            steps_per_second=float(data.get("steps_per_second", 0.0)),
            seconds_per_epoch=float(data.get("seconds_per_epoch", 0.0)),
            gpu_detected=data.get("gpu_detected"),
            gpu_devices=data.get("gpu_devices"),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the DeviceBenchmarkResult instance into a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the DeviceBenchmarkResult instance, suitable for JSON serialisation.
        """
        return {
            "device": self.device,
            "times": self.times,
            "median_seconds": self.median_seconds,
            "mean_seconds": self.mean_seconds,
            "std_seconds": self.std_seconds,
            "steps_per_second": self.steps_per_second,
            "seconds_per_epoch": self.seconds_per_epoch,
            "gpu_detected": self.gpu_detected,
            "gpu_devices": self.gpu_devices,
        }


@dataclass(frozen=True)
class DeviceChoice:
    """
    A dataclass to hold the result of the device selection process. 
    
    It contains the selected device (CPU or GPU), the reason for the selection, and the benchmark results for both CPU
    and GPU. This class is used to encapsulate the outcome of the device selection process, providing a clear 
    structured way to access the selected device and the associated benchmark results.
    """
    device: str
    reason: str
    cpu_result: DeviceBenchmarkResult | None = None
    gpu_result: DeviceBenchmarkResult | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeviceChoice:
        """
        Create a DeviceChoice instance from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            A dictionary containing the device selection results.

        Returns
        -------
        DeviceChoice
            An instance of DeviceChoice populated with the data from the dictionary.
        """
        cpu_result = data.get("cpu_result", None)
        gpu_result = data.get("gpu_result", None)

        if not isinstance(cpu_result, DeviceBenchmarkResult) and cpu_result is not None:
            cpu_result = DeviceBenchmarkResult.from_dict(cpu_result)
        if not isinstance(gpu_result, DeviceBenchmarkResult) and gpu_result is not None:
            gpu_result = DeviceBenchmarkResult.from_dict(gpu_result)

        return cls(
            device=data["device"],
            reason=data["reason"],
            cpu_result=cpu_result,
            gpu_result=gpu_result,
        )


@dataclass(frozen=True)
class BenchmarkWorkerConfig:
    """
    A lightweight payload for the benchmark worker subprocess.

    The payload keeps the serialised SEAWRD configuration as a plain dictionary so the benchmark path can stay decoupled
    from Keras imports until the worker process reconstructs the validated config.
    """
    data_path: str
    seawrd_config: dict[str, Any]

    # Somewhat redudant as these are present in the config, but here for convenience
    benchmark_epochs: int
    benchmark_repeats: int
    warmup_epochs: int

    @classmethod
    def from_config(
        cls,
        config: SEAWRDConfig | Mapping[str, Any] | Any,
        data_path: str | Path,
        benchmark_epochs: int,
        benchmark_repeats: int,
        warmup_epochs: int,
    ) -> "BenchmarkWorkerConfig":
        """
        Build a benchmark worker payload from a SEAWRD configuration or raw mapping.
        
        Parameters
        ----------
        config : SEAWRDConfig | Mapping[str, Any] | Any
            The configuration object to convert into a worker payload.
        data_path : str | Path
            The path to the training and validation data for the benchmark worker.
        benchmark_epochs : int
            The number of epochs to train the model during benchmarking.
        benchmark_repeats : int
            The number of times to repeat the benchmark for each device.
        warmup_epochs : int
            The number of warmup epochs to run before benchmarking.
        
        Returns
        -------
        BenchmarkWorkerConfig
            An instance of BenchmarkWorkerConfig populated with the provided parameters and configuration.
        """
        return cls(
            data_path=str(data_path),
            seawrd_config=_config_to_worker_payload(config),
            benchmark_epochs=benchmark_epochs,
            benchmark_repeats=benchmark_repeats,
            warmup_epochs=warmup_epochs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkWorkerConfig":
        """
        Create a BenchmarkWorkerConfig instance from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            A dictionary containing the benchmark worker configuration.

        Returns
        -------
        BenchmarkWorkerConfig
            An instance of BenchmarkWorkerConfig populated with the data from the dictionary.
        """
        return cls(
            data_path=data["data_path"],
            seawrd_config=data["seawrd_config"],
            benchmark_epochs=data["benchmark_epochs"],
            benchmark_repeats=data["benchmark_repeats"],
            warmup_epochs=data["warmup_epochs"],
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the worker payload into a JSON-serializable dictionary.
        
        Returns
        -------
        dict[str, Any]
            A dictionary representation of the BenchmarkWorkerConfig instance, suitable for JSON serialisation.
        """
        return {
            "data_path": self.data_path,
            "seawrd_config": self.seawrd_config,
            "benchmark_epochs": self.benchmark_epochs,
            "benchmark_repeats": self.benchmark_repeats,
            "warmup_epochs": self.warmup_epochs,
        }



def choose_training_device(config: SEAWRDConfig | Mapping[str, Any],
                           x_train: np.ndarray,
                           y_train: np.ndarray,
                           x_val: np.ndarray,
                           y_val: np.ndarray,
                           benchmark_epochs: int = 10,
                           benchmark_repeats: int = 3,
                           warmup_epochs: int = 10) -> DeviceChoice:
    """
    Benchmark CPU and GPU in isolated subprocesses and choose a backend for training based on the results.
    
    Parameters
    ----------
    config : SEAWRDConfig | Mapping[str, Any]
        A SEAWRDConfig object or raw mapping containing the configuration for the model and training.
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
        The number of warmup epochs to run before benchmarking. Default is 10.

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
        logger.debug("Using temporary directory for benchmarking: %s", tmpdir)
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
        worker_config = BenchmarkWorkerConfig.from_config(
            config=config,
            data_path=data_path,
            benchmark_epochs=benchmark_epochs,
            benchmark_repeats=benchmark_repeats,
            warmup_epochs=warmup_epochs,
        )
        worker_config_dict = worker_config.to_dict()
        logger.debug("Creating worker configuration: %s", worker_config_dict)
        worker_config_path.write_text(json.dumps(worker_config_dict))

        # Run GPU benchmark first as it can fail if there is issues detecting GPU
        try:
            gpu_result = _run_worker("gpu", worker_config_path)
        except Exception as exc:
            logger.warning("GPU benchmark failed: %s", exc)
            return DeviceChoice(
                device="cpu",
                reason=f"GPU benchmark failed: {exc}",
                cpu_result=None,
                gpu_result=None
            )

        # Check if a GPU was detected in the GPU benchmark result. If not, we fall back to CPU.
        # note: done this way to avoid instantiating tensorflow outside of the subprocess
        gpu_detected = bool(gpu_result.gpu_detected)
        if not gpu_detected:
            logger.warning("GPU benchmark did not detect a GPU. Falling back to CPU.")
            return DeviceChoice(
                device="cpu",
                reason="CPU selected because TensorFlow did not detect a GPU inside the GPU benchmark process.",
                cpu_result=None,
                gpu_result=gpu_result
            )

        # Otherwise, now benchmark the CPU.
        cpu_result = _run_worker("cpu", worker_config_path)

    # Compare the median training times for CPU and GPU to determine which device to use for training. If the GPU is
    # faster than the CPU by at least the minimum speedup specified in the configuration, we select the GPU. Otherwise,
    # we fall back to CPU.
    cpu_time = cpu_result.median_seconds
    gpu_time = gpu_result.median_seconds
    speedup = cpu_time / gpu_time
    min_gpu_speedup = _get_min_gpu_speedup(config)

    if speedup >= min_gpu_speedup:
        logger.info("GPU selected for training: GPU was %.2fx faster than CPU, "
                    "exceeding the %.2fx minimum speedup threshold.", speedup, min_gpu_speedup)
        return DeviceChoice(
            device="gpu",
            reason=(f"GPU selected because it was {speedup:.2f}x faster than CPU, "
                    f"exceeding the {min_gpu_speedup:.2f}x minimum speedup threshold."),
            cpu_result=cpu_result,
            gpu_result=gpu_result
        )

    logger.info(
        "CPU selected for training: GPU was only %.2fx the speed of CPU, below the %.2fx minimum speedup threshold.",
        speedup, min_gpu_speedup,
    )
    return DeviceChoice(
        device="cpu",
        reason=(
            f"CPU selected because GPU speedup was only {speedup:.2f}x below the {min_gpu_speedup:.2f}x threshold."
        ),
        cpu_result=cpu_result,
        gpu_result=gpu_result,
    )


def _run_worker(mode: str, worker_config_path: Path) -> DeviceBenchmarkResult:
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
    DeviceBenchmarkResult
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
    set_device_env(mode)

    # Run the benchmark worker process as a subprocess, passing the worker configuration file path as an argument.
    # Capture the output and check for errors.
    try:
        logger.info("Running %s benchmark worker subprocess...", mode.upper())
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "seawrd._device_benchmark_worker",
                str(worker_config_path),
            ],
            env=env,
            # capture_output=True,
            stdout=subprocess.PIPE,  # capture only stdout, not logging in stderr
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

    # The last line of stdout is expected to be a JSON object containing the benchmark results. Parse it and return a
    # DeviceBenchmarkResult instance.
    return DeviceBenchmarkResult.from_dict(json.loads(lines[-1]))


def _config_to_worker_payload(config: SEAWRDConfig | dict[str, Any] | Any) -> dict[str, Any]:
    """
    Convert a configuration object to a payload for the benchmark worker.
    
    The configuration object can be a SEAWRDConfig, a dictionary, or any object that provides a `to_dict` method. The
    function returns a dictionary representation of the configuration that can be serialized to JSON for the worker
    process. This allows the benchmark worker to receive the necessary configuration for training the model without
    requiring the full SEAWRDConfig class or other dependencies.

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
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    raise TypeError("config must provide to_dict() or be a mapping for benchmark serialization")


def _get_min_gpu_speedup(config: SEAWRDConfig | dict[str, Any] | Any) -> float:
    """
    Get the minimum GPU speedup from the configuration object. The configuration object can be a SEAWRDConfig, a
    dictionary, or any object that provides a `device.min_gpu_speedup` attribute. The function returns the minimum GPU
    speedup as a float, which is used to determine whether to select the GPU for training based on the benchmark
    results.

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
    if isinstance(config, Mapping):
        device_section = config.get("device", {})
        if isinstance(device_section, Mapping) and "min_gpu_speedup" in device_section:
            return float(device_section["min_gpu_speedup"])

    device_section = getattr(config, "device", None)
    if device_section is not None and hasattr(device_section, "min_gpu_speedup"):
        return float(device_section.min_gpu_speedup)

    raise TypeError("config must provide device.min_gpu_speedup for benchmarking")
