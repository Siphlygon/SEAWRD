"""
A worker script to benchmark the training performance of a deep neural network on CPU and GPU devices. This script is
intended to be run as a subprocess by the main device selection module, which will provide the necessary configuration
and data paths.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
import tensorflow as tf

from .config import SEAWRDConfig
from .model import DNNManager
from .trainer import DNNTrainer
from .utils import fit_normaliser, load_npz_bundle, get_logger

logger = get_logger("seawrd.device_benchmark_worker")


def _fit_once(config: SEAWRDConfig,
              x_train: np.ndarray,
              y_train: np.ndarray,
              x_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int,
              seed: int) -> float:
    """
    Fit the model once and return the elapsed time in seconds.

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
    epochs : int
        The number of epochs to train the model.
    seed : int
        The random seed for reproducibility.

    Returns
    -------
    float
        The elapsed time in seconds.
    """
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)

    # Train a normaliser if required
    normal = fit_normaliser(x_train) if config.model.use_normalisation is not False else None

    manager = DNNManager.from_config(
        model_config=config.model,
        input_shape=x_train.shape[1:],
        normaliser=normal if config.model.use_normalisation else None
    )
    trainer = DNNTrainer(
        model_manager=manager,
        config=config
    )

    model = trainer.model_manager.clone_model()
    manager.compile_from_config(model, config.compile)

    start = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=config.training.batch_size,
        verbose=0,
        shuffle=config.training.shuffle
    )
    end = time.perf_counter()

    return end - start


def main():
    """
    The main function to run the benchmark worker. It reads the worker configuration from a JSON file specified in the
    command line arguments, loads the training and validation data, and runs the benchmark training for both CPU and GPU
    devices. The results are printed as a JSON object to stdout.
    """
    # First check if a GPU is available in the current environment. If not, we can skip the GPU benchmark and just run
    # the CPU benchmark.
    gpu_devices = tf.config.list_physical_devices("GPU")
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "0" and not gpu_devices:
        logger.warning(
            "CUDA_VISIBLE_DEVICES is set to '0' but no GPU was detected. "
            "Skipping GPU benchmark and falling back to CPU."
        )
        result = {
            "gpu_detected": False,
            "gpu_devices": [],
            "median_seconds": None,
            "times": []
        }
        print(json.dumps(result))
        return

    # Read the worker config from the command line argument
    logger.debug("Reading worker configuration from %s", sys.argv[1])
    worker_config = json.loads(Path(sys.argv[1]).read_text())

    # Load the benchmark data from the specified path in the worker config
    logger.debug("Loading benchmark data from %s", worker_config["data_path"])
    data = load_npz_bundle(Path(worker_config["data_path"]))
    x_train = data["input_features"]
    y_train = data["input_labels"]
    x_val = data["test_features"]
    y_val = data["test_labels"]
    config = SEAWRDConfig.from_dict(
        worker_config["seawrd_config"]
    )

    # Extract the benchmark parameters from the worker config
    warmup_epochs = worker_config["warmup_epochs"]
    benchmark_epochs = worker_config["benchmark_epochs"]
    repeats = worker_config["benchmark_repeats"]
    logger.info(
        "Starting benchmark with %d warmup epochs, %d benchmark epochs, and %d repeats.",
        warmup_epochs, benchmark_epochs, repeats
    )

    # Run a warmup training to stabilise performance before benchmarking
    # This is important because the first training run is always slower due to TensorFlow's initialisation..
    logger.debug("Running warmup training...")
    _fit_once(
        config,
        x_train,
        y_train,
        x_val,
        y_val,
        warmup_epochs,
        seed=0
    )

    # Run the benchmark training for the specified number of repeats and record the elapsed times
    times = []
    for repeat in range(repeats):
        elapsed = _fit_once(
            config,
            x_train,
            y_train,
            x_val,
            y_val,
            benchmark_epochs,
            seed=repeat+1
        )
        logger.info("Benchmark repeat %d/%d: %d epochs completed in %.3f seconds (%.5f seconds per epoch)",
                    repeat+1, repeats, benchmark_epochs, elapsed, elapsed/benchmark_epochs)
        times.append(elapsed)

    # Print the benchmark results as a JSON object to stdout, including whether a GPU was detected and the list of
    # available GPU devices. The main process will parse this output to determine which device to use for training.
    result = {
        "gpu_detected": bool(gpu_devices),
        "gpu_devices": [device.name for device in gpu_devices],
        "median_seconds": float(np.median(times)),
        "times": times
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
