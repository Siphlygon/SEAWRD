from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
from pathlib import Path
from typing import TYPE_CHECKING
import logging

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import keras


def validate_keras_field(field_name: str, field_type: str):
    """
    Validate that Keras can resolve a given field identifier.
    
    Keras provides a mechanism to retrieve various components (like losses, metrics, and activations) using string identifiers. This function checks if the provided field identifier can be resolved by Keras for the specified field type.

    Parameters
    ----------
    field_name : str
        The name of the field to validate (e.g., 'loss', 'metric', 'activation').
    field_type : str
        The name of the Keras module to use for validation (e.g., losses, metrics, activations).
    
    Raises
    ------    
    ValueError
        If the field identifier or type is unknown or unsupported.
    """
    import keras  # use a local import to avoid unnecessary dependencies when importing the package

    match field_type:
        case "losses":
            keras_module = keras.losses
        case "metrics":
            keras_module = keras.metrics
        case "activations":
            keras_module = keras.activations
        case _:
            raise ValueError(f"Unknown or unsupported Keras field type: {field_type!r}")

    try:
        keras_module.get(field_name)
    except Exception as exc:
        raise ValueError(f"Unknown or unsupported Keras {field_type}: {field_name!r}") from exc


def fit_normaliser(train_features: pd.DataFrame | np.ndarray) -> "keras.layers.Normalization":
    """
    Fit a Keras Normalization layer to the training features.

    Parameters
    ----------
    train_features : pd.DataFrame | np.ndarray
        Training feature pandas DataFrame or numpy array to fit the normalizer.

    Returns
    -------
    keras.layers.Normalization
        Fitted Keras Normalization layer.
    """
    import keras  # use a local import to avoid unnecessary dependencies when importing the package

    normaliser = keras.layers.Normalization(axis=-1, name="feature_normaliser")
    if isinstance(train_features, pd.DataFrame):
        values = train_features.to_numpy(dtype=np.float32)
    else:
        values = np.asarray(train_features, dtype=np.float32)

    normaliser.adapt(values)
    return normaliser


def load_npz_bundle(bundle_path: Path) -> dict[str, np.ndarray]:
    """
    Load a NumPy .npz bundle containing training and testing features and labels.

    Parameters
    ----------
    bundle_path : Path
        Path to the .npz bundle file.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary containing the loaded arrays.

    Raises
    ------
    ValueError
        If the bundle is missing any of the required keys: 'input_features', 'input_labels', 'test_features', 'test_labels'.
    """
    bundle = np.load(bundle_path, allow_pickle=False)
    required_keys = {
        "input_features",
        "input_labels",
        "test_features",
        "test_labels",
    }

    missing_keys = required_keys - set(bundle.files)
    if missing_keys:
        raise ValueError(
            f"Training bundle {bundle_path} is missing keys: {sorted(missing_keys)}"
        )

    return {key: np.asarray(bundle[key]) for key in required_keys}


def create_validation_split(input_features: np.ndarray | pd.DataFrame,
                            input_labels: np.ndarray | pd.Series,
                            validation_split: float
                            ) -> tuple[np.ndarray | pd.DataFrame,
                                        np.ndarray | pd.Series,
                                        np.ndarray | pd.DataFrame,
                                        np.ndarray | pd.Series]:
    """
    Create a validation split from the input features and labels based on the given validation split ratio.
    
    Parameters
    ----------
    input_features : np.ndarray | pd.DataFrame
        The features of the input dataset, which will be split into training and validation sets.
    input_labels : np.ndarray | pd.Series
        The labels of the input dataset, which will be split into training and validation sets.
    validation_split : float
        The fraction of the dataset to be used for validation. Must be between 0 and 1.

    Returns
    -------
    x_train : np.ndarray | pd.DataFrame
        The features of the training dataset.
    y_train : np.ndarray | pd.Series
        The labels of the training dataset.
    x_val : np.ndarray | pd.DataFrame
        The features of the validation dataset.
    y_val : np.ndarray | pd.Series
        The labels of the validation dataset.
    """
    n_val = int(len(input_features) * validation_split)
    if n_val <= 0 or n_val >= len(input_features):
        raise ValueError("Unable to create a split from the provided data and validation_split value.")

    x_train, x_val = input_features[:-n_val], input_features[-n_val:]
    y_train, y_val = input_labels[:-n_val], input_labels[-n_val:]
    return x_train, y_train, x_val, y_val


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with the given name and level. If the logger already has
    handlers, clear and reset them.

    Parameters
    ----------
    name : str
        The name of the logger.
    level : int, optional
        The logging level, by default logging.INFO

    Returns
    -------
    logging.Logger
        The logger with the given name and level.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():  # Check if the logger already has handlers
        logger.handlers.clear()  # Clear the default handlers
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s (%(name)s): %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
