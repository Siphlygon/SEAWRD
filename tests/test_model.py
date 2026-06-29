"""
Unit tests for the DNNManager class in the seawrd.model module.
"""

from pathlib import Path
from typing import Any

import keras
import numpy as np
import pytest

from seawrd.config import CompileConfig, ModelConfig
from seawrd.model import DNNManager


def fitted_normalizer():
    """
    Create and return a fitted Keras Normalization layer.

    Returns
    -------
    keras.layers.Normalization
        A Keras Normalization layer that has been adapted to a sample dataset.
    """
    normalizer = keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array([[1., 2.], [3., 4.]], dtype=np.float32))
    return normalizer


# ----------------- Tests for Kerass objects -----------------
def normalize_config(obj : Any) -> Any:
    """
    Convert tuples to lists recursively so Keras configs compare stably.
    
    While the default Keras config for input_shape specifies a tuple, under serialisation and deserialisation, this may
    be converted to a list. This does not change the functionality of the model but will cause an assertion of the two
    configs to fail. This function recursively converts tuples to lists so that the configs can be compared stably.

    Parameters
    ----------
    obj : Any
        The object to normalize. This can be a tuple, list, dict, or any other type.

    Returns
    -------
    Any
        The normalized object.
    """
    if isinstance(obj, tuple):
        return [normalize_config(x) for x in obj]  # convert tuple to list and normalize elements
    if isinstance(obj, list):
        return [normalize_config(x) for x in obj]  # normalize elements of the list
    if isinstance(obj, dict):
        return {k: normalize_config(v) for k, v in obj.items()}
    return obj


def assert_same_keras_model(model_a : keras.Model,
                            model_b : keras.Model,
                            same_weights : bool = True):
    """
    Assert that two Keras models have the same architecture and (optionally) weights.

    Parameters
    ----------
    model_a : keras.Model
        The first Keras model to compare.
    model_b : keras.Model
        The second Keras model to compare.
    same_weights : bool, optional
        Whether to perform a comparison of weights, by default True
    """
    # Assert that the two models have the exact same architecture (layers)
    config_a = normalize_config(model_a.get_config())
    config_b = normalize_config(model_b.get_config())
    assert config_a == config_b, f"Model architectures differ:\n{config_a}\n{config_b}"

    # Assert that the two models have the same number of weight arrays
    weights_a = model_a.get_weights()
    weights_b = model_b.get_weights()
    assert len(weights_a) == len(weights_b), f"Number of weight arrays differ: {len(weights_a)} != {len(weights_b)}"

    # Assert that the two models have the same weight values
    for i, (wa, wb) in enumerate(zip(weights_a, weights_b)):
        assert wa.shape == wb.shape, f"Weight {i} shape differs: {wa.shape} != {wb.shape}"

        if same_weights:  # If exact comparison is required, check for exact equality
            assert np.array_equal(wa, wb), f"Weight {i} values differ"


def assert_same_history(history_a : keras.callbacks.History,
                        history_b : keras.callbacks.History,
                        rtol : float = 1e-7,
                        atol : float = 1e-8):
    """
    Assert that two Keras History objects have the same keys and values within a specified tolerance.

    Parameters
    ----------
    history_a : keras.callbacks.History
        The first Keras History object to compare.
    history_b : keras.callbacks.History
        The second Keras History object to compare.
    rtol : float, optional
        The relative tolerance for comparison, by default 1e-7
    atol : float, optional
        The absolute tolerance for comparison, by default 1e-8
    """
    # Extract the history dictionaries from the History objects
    hist_a = history_a.history
    hist_b = history_b.history

    assert hist_a.keys() == hist_b.keys(), f"History keys differ: {hist_a.keys()} != {hist_b.keys()}"

    for key in hist_a:
        values_a = np.asarray(hist_a[key])
        values_b = np.asarray(hist_b[key])

        assert values_a.shape == values_b.shape, (
            f"History entry {key!r} has different shape: "
            f"{values_a.shape} != {values_b.shape}"
        )

        np.testing.assert_allclose(
            values_a,
            values_b,
            rtol=rtol,
            atol=atol,
            err_msg=f"History entry {key!r} differs",
        )

# ----------------- Tests for DNNManager -----------------
def test_from_config_requires_normalizer_when_enabled():
    """
    Test that DNNManager.from_config raises a ValueError when use_normalisation is True but no normaliser is provided.
    """
    cfg = ModelConfig(use_normalisation=True)

    with pytest.raises(ValueError, match="normaliser"):
        DNNManager.from_config(cfg, input_shape=(2,), normaliser=None)


def test_from_config_builds_expected_model_with_normalizer():
    """
    Test that DNNManager.from_config builds a model with the expected number of layers and output
    units when use_normalisation is True and a normaliser is provided.
    """
    cfg = ModelConfig(
        num_layers=2,
        num_neurons=4,
        num_outputs=1,
        activation="relu",
        use_normalisation=True,
    )

    manager = DNNManager.from_config(cfg, input_shape=(2,), normaliser=fitted_normalizer())

    dense_layers = [layer for layer in manager.model.layers if isinstance(layer, keras.layers.Dense)]
    assert len(dense_layers) == 3, f"Expected 3 Dense layers (2 hidden + 1 output), got {len(dense_layers)}"
    assert dense_layers[-1].units == 1, f"Expected output layer to have 1 unit, got {dense_layers[-1].units}"
    assert manager.model.layers[0].name.startswith("normalization"), (
        f"Expected first layer to be a Normalization layer, got {manager.model.layers[0].name}")


def test_compile_from_config_compiles_model():
    """
    Test that DNNManager._compile_from_config compiles the model with the expected optimizer and metrics.
    """
    cfg = ModelConfig(use_normalisation=False, num_layers=1, num_neurons=4)
    manager = DNNManager.from_config(cfg, input_shape=(2,))

    DNNManager.compile_from_config(
        manager.model,
        CompileConfig(learning_rate=0.001, metrics=("mean_squared_error",)),
    )

    assert manager.model.optimizer is not None, "Model optimizer is None after compilation"


def test_latest_version_returns_zero_when_no_models(tmp_path : Path):
    """
    Test that DNNManager.get_latest_version returns 0 when there are no saved models in the specified directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        The path to the directory containing saved models. This is a temporary directory provided by pytest.
    """
    manager = DNNManager(model=keras.Sequential(), history=None, version=0, model_name="demo1")

    assert manager.get_latest_version(tmp_path, "demo1") == 0, (
        "Expected latest version to be 0 when no models are present")


def test_latest_version_returns_highest_version(tmp_path : Path):
    """
    Test that DNNManager.get_latest_version returns the highest version number from the saved models in the specified
    directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        The path to the directory containing saved models. This is a temporary directory provided by pytest.
    """
    # Create dummy model files with different version numbers, format is {model_name}_v{version}_model.keras
    (tmp_path / "demo_v1_model.keras").touch()
    (tmp_path / "demo_v2_model.keras").touch()
    (tmp_path / "demo_v3_model.keras").touch()

    manager = DNNManager(model=keras.Sequential(), history=None, version=0, model_name="demo")

    assert manager.get_latest_version(tmp_path, "demo") == 3, (
        "Expected latest version to be 3 based on saved model files")


def test_model_save_and_load(tmp_path : Path):
    """
    Test that DNNManager.save_model and DNNManager.load_model correctly save and load a model.

    Parameters
    ----------
    tmp_path : pathlib.Path
        The path to the directory where the model will be saved. This is a temporary directory provided by pytest.
    """
    cfg = ModelConfig(use_normalisation=False, num_layers=1, num_neurons=4)
    manager = DNNManager.from_config(cfg, input_shape=(2,))

    # Save the model
    # todo: it is likely these methods will change to use config or class members, so change tests when appropriate
    manager.save_model_version(
        model=manager.model,
        history=manager.history,
        model_dir=tmp_path,
        model_name=manager.model_name,
        version=manager.version
    )

    # Load the model
    manager.load_model_version(
        model_dir=tmp_path,
        model_name=manager.model_name,
        version=manager.version
    )

    assert_same_keras_model(manager.model, manager.model, same_weights=True)


def test_model_from_previous_model(tmp_path : Path):
    """
    Test that DNNManager.from_previous_model correctly creates a new DNNManager instance with the same model and
    history as the previous instance.
    """
    cfg = ModelConfig(use_normalisation=False, num_layers=1, num_neurons=4)
    manager1 = DNNManager.from_config(cfg, input_shape=(2,))
    manager1.save_model_version(
        model=manager1.model,
        history=manager1.history,
        model_dir=tmp_path,
        model_name=manager1.model_name,
        version=manager1.version
    )

    manager2 = DNNManager.from_previous_model(
        model_dir=tmp_path,
        model_name=manager1.model_name,
        version=manager1.version
    )

    assert_same_keras_model(manager1.model, manager2.model, same_weights=True)
    assert manager2.version == manager1.version, "Loaded version does not match the saved version"
    assert manager2.model_name == manager1.model_name, "Loaded model name does not match the saved model name"
    assert_same_history(manager1.history, manager2.history)


def test_clone_model_correctly_clones_model():
    """
    Test that DNNManager.clone_model correctly creates a new model with the same architecture and normalisation layer
    (if present) as the original model. The cloned model will NOT have the same weights.
    """
    cfg = ModelConfig(use_normalisation=False, num_layers=1, num_neurons=4)
    manager = DNNManager.from_config(cfg, input_shape=(2,))
    model1 = manager.model
    model2 = manager.clone_model()

    assert_same_keras_model(model1, model2, same_weights=False)


# def test_model_name_counts_hidden_layers_without_normalizer():
#     """
#     Test that the model name generated by DNNManager.from_config correctly counts the number of hidden layers when
#     use_normalisation is False.
#     """
#     cfg = ModelConfig(
#         num_layers=2,
#         num_neurons=4,
#         num_outputs=1,
#         use_normalisation=False,
#     )

#     manager = DNNManager.from_config(cfg, input_shape=(3,))

#     assert manager.model_name == "R(2x4_3i_1o)"
