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
from seawrd.predictor import EnsemblePredictor, Predictor


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


def test_save_and_load_ensemble_manifest_round_trip(tmp_path: Path):
    """
    Test that DNNManager.save_ensemble_manifest writes a manifest that load_ensemble_manifest reads back correctly.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    """
    manager = DNNManager(model=simple_model(3), history=None, version=0, model_name="demo")

    manager.save_ensemble_manifest(
        tmp_path, "demo", version=1,
        member_names=["demo_member0", "demo_member1"],
        feature_names=["a", "b", "c"],
        label_name="R_p",
    )
    manifest = DNNManager.load_ensemble_manifest(tmp_path, "demo", version=1)

    assert manifest is not None, "Expected an ensemble manifest to be loaded"
    assert manifest["member_names"] == ["demo_member0", "demo_member1"], "Member names did not round-trip in order"
    assert manifest["num_members"] == 2
    assert manifest["feature_names"] == ["a", "b", "c"]
    assert manifest["label_name"] == "R_p"


def test_load_ensemble_manifest_returns_none_when_absent(tmp_path: Path):
    """
    Test that DNNManager.load_ensemble_manifest returns None when no ensemble manifest file exists.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    """
    assert DNNManager.load_ensemble_manifest(tmp_path, "demo", version=1) is None, (
        "Expected None when no ensemble manifest is present")


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


# ----------------- Tests for EnsemblePredictor -----------------
class ConstantModel:
    """
    A duck-typed stand-in for a keras.Model that always predicts a fixed value for every row, used to test
    EnsemblePredictor's aggregation logic without needing real trained models.
    """
    def __init__(self, value: float, num_features: int = 2):
        """
        Initialise the constant model.

        Parameters
        ----------
        value : float
            The value predicted for every row.
        num_features : int, optional
            The number of input features expected, by default 2.
        """
        self.value = value
        self.input_shape = (None, num_features)

    def predict(self, x, batch_size=None, verbose=0):
        """
        Simulate keras.Model.predict by returning the configured constant value for every row.

        Parameters
        ----------
        x : np.ndarray
            The input features; only its length is used.
        batch_size : int | None, optional
            Ignored; present only to match the keras.Model.predict signature.
        verbose : int, optional
            Ignored; present only to match the keras.Model.predict signature.

        Returns
        -------
        np.ndarray
            An array of shape (len(x), 1) filled with the constant value.
        """
        return np.full((len(x), 1), self.value, dtype=np.float32)


def constant_ensemble(values, feature_names=("a", "b"), label_name="R_p") -> EnsemblePredictor:
    """
    Build an EnsemblePredictor whose members each always predict one of the given constant values.

    Parameters
    ----------
    values : Sequence[float]
        One constant prediction value per ensemble member.
    feature_names : tuple[str, ...], optional
        The feature names to report, by default ("a", "b")
    label_name : str, optional
        The label name to report, by default "R_p"

    Returns
    -------
    EnsemblePredictor
        An EnsemblePredictor whose members deterministically predict the given values.
    """
    predictors = [
        Predictor(model=ConstantModel(value, num_features=len(feature_names)),
                 feature_names=list(feature_names), label_name=label_name)
        for value in values
    ]
    return EnsemblePredictor(predictors)


def test_ensemble_predictor_requires_at_least_one_predictor():
    """
    Test that EnsemblePredictor raises a ValueError when constructed with no member predictors.
    """
    with pytest.raises(ValueError, match="at least one"):
        EnsemblePredictor([])


def test_ensemble_predictor_predict_with_uncertainty_computes_mean_and_std():
    """
    Test that predict_with_uncertainty returns the per-sample mean and standard deviation across ensemble members.
    """
    ensemble = constant_ensemble([1.0, 2.0, 3.0])
    x = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    mean, std = ensemble.predict_with_uncertainty(x)

    expected_std = np.std([1.0, 2.0, 3.0])
    np.testing.assert_allclose(mean, [2.0, 2.0])
    np.testing.assert_allclose(std, [expected_std, expected_std])


def test_ensemble_predictor_predict_returns_mean_only():
    """
    Test that EnsemblePredictor.predict returns just the ensemble mean, matching Predictor's return shape/type.
    """
    ensemble = constant_ensemble([1.0, 3.0])
    result = ensemble.predict(np.array([[0.0, 0.0]], dtype=np.float32))

    np.testing.assert_allclose(result, [2.0])


def test_ensemble_predictor_predict_dataframe_default_and_custom_columns():
    """
    Test that predict_dataframe appends default-named prediction/uncertainty columns, and respects custom names.
    """
    ensemble = constant_ensemble([1.0, 3.0])
    df = pd.DataFrame({"a": [0.0], "b": [0.0]})

    result = ensemble.predict_dataframe(df)
    assert "R_p_pred" in result.columns, "Expected a default prediction column named after the label"
    assert "R_p_std" in result.columns, "Expected a default uncertainty column named after the label"
    assert result["R_p_pred"].iloc[0] == pytest.approx(2.0)
    assert result["R_p_std"].iloc[0] == pytest.approx(1.0)

    custom = ensemble.predict_dataframe(df, prediction_column="mean_pred", uncertainty_column="unc")
    assert "mean_pred" in custom.columns and "unc" in custom.columns


def test_ensemble_predictor_from_saved_raises_without_ensemble_manifest(tmp_path: Path):
    """
    Test that EnsemblePredictor.from_saved raises a FileNotFoundError when a model exists but was not trained with
    output.save_ensemble enabled (i.e. no ensemble manifest is present).

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    """
    # Only the presence of a "*_v{version}_model.keras" file is needed to resolve the latest version; no real model
    # needs to be loaded before the missing-ensemble-manifest check fires.
    (tmp_path / "demo_v1_model.keras").touch()

    with pytest.raises(FileNotFoundError, match="No ensemble manifest"):
        EnsemblePredictor.from_saved(tmp_path, "demo", version=1)


@pytest.mark.tf
@pytest.mark.slow
def test_ensemble_predictor_from_saved_round_trip(tmp_path: Path):
    """
    Test that an ensemble saved via DNNManager (mirroring what DNNTrainer.train_models does with
    output.save_ensemble=True) can be reloaded via EnsemblePredictor.from_saved and produces per-sample uncertainty.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    """
    cfg = ModelConfig(use_normalisation=False, num_layers=1, num_neurons=4)
    feature_names = ["a", "b", "c"]
    member_names = []

    for seed in range(3):
        manager = DNNManager.from_config(cfg, input_shape=(3,))
        member_name = f"demo_member{seed}"
        member_names.append(member_name)
        manager.save_model_version(
            model=manager.model,
            history=manager.history,
            model_dir=tmp_path,
            model_name=member_name,
            version=1,
            feature_names=feature_names,
            label_name="R_p",
        )

    # The "best" model itself, saved under the ensemble's parent name, is what get_latest_version resolves against
    manager.save_model_version(
        model=manager.model, history=manager.history, model_dir=tmp_path,
        model_name="demo", version=1, feature_names=feature_names, label_name="R_p",
    )
    manager.save_ensemble_manifest(
        tmp_path, "demo", version=1, member_names=member_names,
        feature_names=feature_names, label_name="R_p",
    )

    ensemble = EnsemblePredictor.from_saved(tmp_path, "demo", version=1)
    assert len(ensemble.predictors) == 3

    df = pd.DataFrame({"c": [3.0], "a": [1.0], "b": [2.0]})  # deliberately shuffled
    result = ensemble.predict_dataframe(df)

    assert "R_p_pred" in result.columns
    assert "R_p_std" in result.columns
    assert np.isfinite(result["R_p_pred"].iloc[0])
    assert result["R_p_std"].iloc[0] >= 0.0
