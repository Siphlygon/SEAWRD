"""
Unit tests for the Predictor class and the model manifest helpers.
"""
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import pytest

from seawrd.config import ModelConfig
from seawrd.model import DNNManager
from seawrd.predictor import Predictor


def simple_model(num_features: int = 3, num_outputs: int = 1) -> keras.Model:
    """
    Build a tiny compiled-free Keras model for prediction tests.

    Parameters
    ----------
    num_features : int, optional
        The number of input features, by default 3.
    num_outputs : int, optional
        The number of output units, by default 1.

    Returns
    -------
    keras.Model
        A minimal Keras Sequential model.
    """
    return keras.Sequential([
        keras.Input(shape=(num_features,)),
        keras.layers.Dense(num_outputs),
    ])


# ----------------- Tests for the manifest helpers -----------------
def test_save_and_load_manifest_round_trip(tmp_path: Path):
    """
    Test that DNNManager.save_manifest writes a manifest that load_manifest reads back with the same contents.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    """
    manager = DNNManager(model=simple_model(3), history=None, version=0, model_name="demo")

    manager.save_manifest(tmp_path, "demo", version=1, feature_names=["a", "b", "c"], label_name="R_p")
    manifest = DNNManager.load_manifest(tmp_path, "demo", version=1)

    assert manifest is not None, "Expected a manifest to be loaded"
    assert manifest["feature_names"] == ["a", "b", "c"], "Feature names did not round-trip"
    assert manifest["label_name"] == "R_p", "Label name did not round-trip"
    assert manifest["num_outputs"] == 1, "num_outputs was not recorded correctly"


def test_load_manifest_returns_none_when_absent(tmp_path: Path):
    """
    Test that DNNManager.load_manifest returns None when no manifest file exists.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    """
    assert DNNManager.load_manifest(tmp_path, "demo", version=1) is None, (
        "Expected None when no manifest is present")


# ----------------- Tests for Predictor feature preparation -----------------
def test_predictor_reorders_dataframe_columns_to_training_order():
    """
    Test that Predictor._prepare_features selects and reorders DataFrame columns to match the training feature order.
    """
    predictor = Predictor(model=simple_model(3), feature_names=["a", "b", "c"])

    # Columns deliberately supplied out of order and with an extra column that should be ignored
    df = pd.DataFrame({"c": [3.0], "extra": [99.0], "a": [1.0], "b": [2.0]})
    prepared = predictor._prepare_features(df)

    np.testing.assert_array_equal(prepared, np.array([[1.0, 2.0, 3.0]], dtype=np.float32))


def test_predictor_raises_on_missing_features():
    """
    Test that Predictor._prepare_features raises a ValueError when a required feature column is missing.
    """
    predictor = Predictor(model=simple_model(3), feature_names=["a", "b", "c"])

    with pytest.raises(ValueError, match="missing required features"):
        predictor._prepare_features(pd.DataFrame({"a": [1.0], "b": [2.0]}))


def test_predictor_raises_on_wrong_feature_count_for_array():
    """
    Test that Predictor._prepare_features raises a ValueError when an array has the wrong number of features.
    """
    predictor = Predictor(model=simple_model(3))

    with pytest.raises(ValueError, match="expects 3 features"):
        predictor._prepare_features(np.array([[1.0, 2.0]], dtype=np.float32))


def test_predictor_promotes_single_sample_to_batch():
    """
    Test that Predictor._prepare_features reshapes a single 1D sample into a batch of one.
    """
    predictor = Predictor(model=simple_model(3))

    prepared = predictor._prepare_features(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert prepared.shape == (1, 3), f"Expected shape (1, 3), got {prepared.shape}"


def test_manifest_takes_precedence_over_explicit_args():
    """
    Test that a manifest's feature names and label take precedence over explicitly passed arguments.
    """
    manifest = {"feature_names": ["x", "y"], "label_name": "M_p"}
    predictor = Predictor(model=simple_model(2), feature_names=["a", "b"], label_name="R_p", manifest=manifest)

    assert predictor.feature_names == ["x", "y"], "Manifest feature names should win"
    assert predictor.label_name == "M_p", "Manifest label name should win"


# ----------------- Tests for Predictor prediction (require a real Keras call) -----------------
@pytest.mark.tf
@pytest.mark.slow
def test_predict_returns_flat_array_for_single_output():
    """
    Test that Predictor.predict returns a 1D array with one prediction per input row for a single-output model.
    """
    predictor = Predictor(model=simple_model(3, num_outputs=1), feature_names=["a", "b", "c"])
    df = pd.DataFrame({"a": [1.0, 4.0], "b": [2.0, 5.0], "c": [3.0, 6.0]})

    predictions = predictor.predict(df)

    assert predictions.shape == (2,), f"Expected shape (2,), got {predictions.shape}"


@pytest.mark.tf
@pytest.mark.slow
def test_predict_dataframe_appends_prediction_column():
    """
    Test that Predictor.predict_dataframe returns the input with a named prediction column and preserved index.
    """
    predictor = Predictor(model=simple_model(3, num_outputs=1), feature_names=["a", "b", "c"], label_name="R_p")
    df = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]}, index=[42])

    result = predictor.predict_dataframe(df)

    assert "R_p_pred" in result.columns, "Expected a prediction column named after the label"
    assert list(result.index) == [42], "Input index should be preserved"


@pytest.mark.tf
@pytest.mark.slow
def test_from_saved_round_trip_uses_manifest(tmp_path: Path):
    """
    Test that a model saved with feature names can be reloaded via Predictor.from_saved and predict on a DataFrame whose
    columns are in a different order.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    """
    cfg = ModelConfig(use_normalisation=False, num_layers=1, num_neurons=4)
    manager = DNNManager.from_config(cfg, input_shape=(3,))

    manager.save_model_version(
        model=manager.model,
        history=manager.history,
        model_dir=tmp_path,
        model_name="demo",
        version=1,
        feature_names=["a", "b", "c"],
        label_name="R_p",
    )

    predictor = Predictor.from_saved(tmp_path, "demo", version=1)
    assert predictor.feature_names == ["a", "b", "c"], "Predictor should load feature names from the manifest"

    # Supply columns out of order; the manifest ordering should make the prediction well-defined
    shuffled = pd.DataFrame({"c": [3.0], "a": [1.0], "b": [2.0]})
    ordered = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})

    np.testing.assert_allclose(predictor.predict(shuffled), predictor.predict(ordered))
