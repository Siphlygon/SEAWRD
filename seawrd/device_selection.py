"""
A module to benchmark CPU and GPU training performance and select the best device for training a deep neural network.
The module provides a function to run isolated subprocesses for benchmarking both CPU and GPU, and returns the selected
device along with the benchmark results.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:
    from .config import SEAWRDConfig

from .bootstrap import (
    config_to_worker_payload,
    get_min_gpu_speedup,
    set_device_env,
    get_cpu_name,
    get_gpu_names,
)
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

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the DeviceChoice instance into a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the DeviceChoice instance, suitable for JSON serialisation.
        """
        return {
            "device": self.device,
            "reason": self.reason,
            "cpu_result": self.cpu_result.to_dict() if self.cpu_result else None,
            "gpu_result": self.gpu_result.to_dict() if self.gpu_result else None,
        }


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
            seawrd_config=config_to_worker_payload(config),
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


@dataclass(frozen=True)
class BenchmarkCache:
    """
    A dataclass to hold the cached benchmark results for CPU and GPU.
    
    It contains the benchmark results for both CPU and GPU, allowing for quick access to previously computed results
    without the need to rerun the benchmarks. This class is used to encapsulate the cached benchmark results, providing
    a structured way to access the results for both devices.
    """
    # Configuration
    seawrd_config: dict[str, Any]
    num_inputs : int

    # Device information
    tensorflow_version: str
    keras_version: str
    gpu_names: tuple[str, ...]
    cpu_name: str

    # Cached benchmark results
    device_choice: DeviceChoice

    # The benchmark relevant fields inside the configuration that are used to generate a unique cache key
    _RELEVANT_FIELDS = (
        "num_layers",
        "num_neurons",
        "num_outputs",
        "activation",
        "use_normalisation",
        "batch_size",
        "shuffle",
        "loss",
        "optimiser",
        "learning_rate",
        "metrics",
        "steps_per_execution",
        "jit_compile",
        "min_gpu_speedup", # todo: could intelligently deal with this changing; dynamically see if to load cache
        "warmup_epochs",
        "benchmark_epochs",
        "benchmark_repeats",
    )


    def _get_relevant_config(self) -> dict[str, Any]:
        """
        Extract the relevant configuration fields from the SEAWRD configuration for generating a unique cache key.
        
        Returns
        -------
        dict[str, Any]
            A dictionary containing only the relevant configuration fields necessary for generating a unique cache key.
        """
        model_config = self.seawrd_config.get("model", {})
        training_config = self.seawrd_config.get("training", {})
        compile_config = self.seawrd_config.get("compile", {})
        device_config = self.seawrd_config.get("device", {})

        relevant_config = {
            "model": {k: model_config[k] for k in self._RELEVANT_FIELDS if k in model_config},
            "training": {k: training_config[k] for k in self._RELEVANT_FIELDS if k in training_config},
            "compile": {k: compile_config[k] for k in self._RELEVANT_FIELDS if k in compile_config},
            "device": {k: device_config[k] for k in self._RELEVANT_FIELDS if k in device_config},
        }
        return relevant_config


    def to_overhead(self) -> dict[str, Any]:
        """
        Convert the BenchmarkCache instance into a dictionary suitable for generating a cache key. This does not contain
        the benchmark results themselves, but only the configuration and device information necessary to uniquely
        identify the cache entry.
        
        Returns
        -------
        dict[str, Any]
            A dictionary representation of the BenchmarkCache instance, containing only the configuration and device
            information necessary for generating a unique cache key.
        """
        return {
            "seawrd_config": self._get_relevant_config(),
            "num_inputs": self.num_inputs,
            "tensorflow_version": self.tensorflow_version,
            "keras_version": self.keras_version,
            "gpu_names": list(self.gpu_names),
            "cpu_name": self.cpu_name,
        }



def _cache_key(overhead : dict[str, Any]) -> str:
    """
    Generate a unique cache key based on the provided cache overhead. The cache key is a JSON string representation of
    the overhead, sorted by keys to ensure consistency. This allows for easy comparison and retrieval of cached
    benchmark results based on the configuration used for training and the device information.

    Parameters
    ----------
    overhead : dict[str, Any]
        A dictionary containing the cache overhead (configuration and device information) for which to generate a cache
        key.

    Returns
    -------
    str
        A unique cache key based on the cache overhead.
    """
    encoded = json.dumps(overhead, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16] # Use the first 16 characters of the SHA256 hash as the cache key


def _create_benchmark_cache(config : Mapping[str, Any],
                            num_inputs: int,
                            device_choice: DeviceChoice) -> BenchmarkCache:
    """
    Create a BenchmarkCache instance from the provided configuration, number of inputs and outputs, and device choice.

    Parameters
    ----------
    config : Mapping[str, Any]
        A raw mapping containing the configuration for the model and training.
    num_inputs : int
        The number of input features for the model.
    device_choice : DeviceChoice
        The selected device and benchmark results.

    Returns
    -------
    BenchmarkCache
        An instance of BenchmarkCache populated with the provided parameters.
    """
    return BenchmarkCache(
        seawrd_config=config_to_worker_payload(config),
        num_inputs=num_inputs,
        tensorflow_version=version("tensorflow"),
        keras_version=version("keras"),
        gpu_names=tuple(get_gpu_names()),
        cpu_name=get_cpu_name(),
        device_choice=device_choice,
    )


def _save_to_cache(config : Mapping[str, Any],
                   num_inputs: int,
                   device_choice: DeviceChoice) -> None:
    """
    Save the benchmark cache to a JSON file in the specified cache directory. The cache file is named based on a unique
    cache key generated from the cache overhead.

    Parameters
    ----------
    config : Mapping[str, Any]
        A raw mapping containing the configuration for the model and training.
    num_inputs : int
        The number of input features for the model.
    device_choice : DeviceChoice
        The selected device and benchmark results.

    Raises
    ------
    OSError
        If there is an error creating the cache directory or writing the cache file.
    """
    cache_dir = Path(config["output"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a BenchmarkCache instance to generate the cache overhead and key
    cache = _create_benchmark_cache(
        config=config,
        num_inputs=num_inputs,
        device_choice=device_choice,
    )
    overhead = cache.to_overhead()
    cache_key = _cache_key(overhead)

    # Save the cache to a JSON file named based on the cache key in the specified cache directory
    cache_path = cache_dir / f"benchmark_{cache_key}.json"
    logger.info("Saving benchmark results to cache: %s", cache_path)
    with open(cache_path, "w") as f:
        json.dump({
            "overhead": overhead,
            "device_choice": cache.device_choice.to_dict(),
        }, f, indent=4)


def _load_from_cache(config : Mapping[str, Any],
                     num_inputs: int) -> DeviceChoice | None:
    """
    Load the benchmark cache from a JSON file in the specified cache directory. The cache file is named based on a
    unique cache key generated from the cache overhead. If the cache file exists, it is loaded and the DeviceChoice is
    returned. If the cache file does not exist, None is returned.

    Parameters
    ----------
    config : Mapping[str, Any]
        A raw mapping containing the configuration for the model and training.
    num_inputs : int
        The number of input features for the model.
    
    Returns
    -------
    DeviceChoice | None
        The loaded device choice or None if no cache is available.
    """
    cache_dir = Path(config["output"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a cache overhead dictionary containing the configuration and device information necessary to generate
    # a unique cache key.
    benchmark_cache = _create_benchmark_cache(
        config=config,
        num_inputs=num_inputs,
        device_choice=DeviceChoice(
            device="unknown",
            reason="Cache overhead only; no benchmark results yet.",
            cpu_result=None,
            gpu_result=None,
        ),
    )
    cache_overhead = benchmark_cache.to_overhead()
    cache_key = _cache_key(cache_overhead)
    cache_path = cache_dir / f"benchmark_{cache_key}.json"

    # Only load the cache if the cache file exists. If it does not exist, return None to indicate that no cached
    # benchmark results are available.
    if cache_path.exists():
        logger.info("Loading benchmark results from cache: %s", cache_path)
        cached_data = json.loads(cache_path.read_text())
        return DeviceChoice.from_dict(cached_data["device_choice"])
    return None


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


def choose_training_device(config: Mapping[str, Any],
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
    config : Mapping[str, Any]
        A raw mapping containing the configuration for the model and training.
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

    output_config = config.get("output", {})
    use_cache = output_config.get("use_cache", True)
    if use_cache:
        logger.info("Checking for cached benchmark results...")
        result = _load_from_cache(
            config=config,
            num_inputs=x_train.shape[1],
        )
        if result: # if we have a cached result, return it immediately
            logger.info("Using cached benchmark results for device selection.")
            return result
        logger.info("No cached benchmark results found; proceeding with benchmarking.")
    else:
        logger.info("Benchmark caching is disabled; proceeding with benchmarking.")

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
    min_gpu_speedup = get_min_gpu_speedup(config)

    if speedup >= min_gpu_speedup:
        logger.info("GPU selected for training: GPU was %.2fx faster than CPU, "
                    "exceeding the %.2fx minimum speedup threshold.", speedup, min_gpu_speedup)
        device_choice = DeviceChoice(
            device="gpu",
            reason=(f"GPU selected because it was {speedup:.2f}x faster than CPU, "
                    f"exceeding the {min_gpu_speedup:.2f}x minimum speedup threshold."),
            cpu_result=cpu_result,
            gpu_result=gpu_result
        )

        if use_cache:
            _save_to_cache(
                config=config,
                num_inputs=x_train.shape[1],
                device_choice=device_choice
                )

        return device_choice

    logger.info(
        "CPU selected for training: GPU was only %.2fx the speed of CPU, below the %.2fx minimum speedup threshold.",
        speedup, min_gpu_speedup,
    )

    device_choice = DeviceChoice(
        device="cpu",
        reason=(f"CPU selected because GPU speedup was only {speedup:.2f}x below the {min_gpu_speedup:.2f}x threshold."),
        cpu_result=cpu_result,
        gpu_result=gpu_result
    )
    if use_cache:
        _save_to_cache(
            config=config,
            num_inputs=x_train.shape[1],
            num_outputs=y_train.shape[1] if y_train.ndim > 1 else 1,
            device_choice=device_choice
        )

    return device_choice
