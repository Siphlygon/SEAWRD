"""
Unit tests for the DNNManager class in the seawrd.model module.
"""
from pathlib import Path

import keras
import numpy as np
import pytest

from seawrd.config import CompileConfig, ModelConfig
from seawrd.model import DNNManager
from tests.helpers import assert_same_keras_model, assert_same_history, assert_same_keras_normalisation


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


@pytest.mark.tf
@pytest.mark.slow
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
    original_model = manager.model
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

    assert_same_keras_model(original_model, manager.model, same_weights=True)


@pytest.mark.tf
@pytest.mark.slow
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
    cfg = ModelConfig(use_normalisation=True, num_layers=1, num_neurons=4)
    manager = DNNManager.from_config(cfg, input_shape=(2,), normaliser=fitted_normalizer())
    model1 = manager.model
    model2 = manager.clone_model()

    assert_same_keras_normalisation(model1, model2)


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
