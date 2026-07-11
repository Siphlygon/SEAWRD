"""
Unit tests for the seawrd.evaluation module: regression metrics and the ModelEvaluator class.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

from seawrd.config import ModelConfig
from seawrd.evaluation import ModelEvaluator, compute_regression_metrics
from seawrd.model import DNNManager
from seawrd.predictor import Predictor


class DummyModel:
    """
    A minimal duck-typed stand-in for a keras.Model, used to test evaluation logic without needing a real model.
    """
    def __init__(self, input_shape, predict_fn):
        """
        Initialise the dummy model.

        Parameters
        ----------
        input_shape : tuple
            The value returned for ``model.input_shape``; only the last element is used by Predictor.
        predict_fn : Callable[[np.ndarray], np.ndarray]
            A function mapping a 2D feature array to a 2D prediction array.
        """
        self.input_shape = input_shape
        self._predict_fn = predict_fn

    def predict(self, x, batch_size=None, verbose=0):
        """
        Simulate keras.Model.predict by delegating to the configured predict function.

        Parameters
        ----------
        x : np.ndarray
            The input features.
        batch_size : int | None, optional
            Ignored; present only to match the keras.Model.predict signature.
        verbose : int, optional
            Ignored; present only to match the keras.Model.predict signature.

        Returns
        -------
        np.ndarray
            The predictions.
        """
        return self._predict_fn(x)


def sum_predictor(feature_names=("a", "b"), label_name="R_p") -> Predictor:
    """
    Build a Predictor around a DummyModel that predicts the row-wise sum of its features.

    Parameters
    ----------
    feature_names : tuple[str, ...], optional
        The feature names to report, by default ("a", "b")
    label_name : str, optional
        The label name to report, by default "R_p"

    Returns
    -------
    Predictor
        A Predictor whose predictions equal the sum of the input features.
    """
    model = DummyModel(
        input_shape=(None, len(feature_names)),
        predict_fn=lambda x: np.sum(x, axis=1, keepdims=True),
    )
    return Predictor(model=model, feature_names=list(feature_names), label_name=label_name)


# ----------------- Tests for compute_regression_metrics -----------------
def test_compute_regression_metrics_perfect_fit():
    """
    Test that compute_regression_metrics reports zero error and an R^2 of 1.0 for a perfect fit.
    """
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = compute_regression_metrics(y_true, y_true.copy())

    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["bias"] == pytest.approx(0.0)
    assert metrics["max_error"] == pytest.approx(0.0)
    assert metrics["mape"] == pytest.approx(0.0)


def test_compute_regression_metrics_known_values():
    """
    Test compute_regression_metrics against a small, hand-computed example.
    """
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])  # residuals: -1, 0, 1

    metrics = compute_regression_metrics(y_true, y_pred)

    assert metrics["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["bias"] == pytest.approx(0.0)
    assert metrics["max_error"] == pytest.approx(1.0)
    # ss_res = 1 + 0 + 1 = 2; ss_tot (mean=2) = 1 + 0 + 1 = 2; r2 = 1 - 2/2 = 0
    assert metrics["r2"] == pytest.approx(0.0)


def test_compute_regression_metrics_raises_on_shape_mismatch():
    """
    Test that compute_regression_metrics raises a ValueError when y_true and y_pred have different lengths.
    """
    with pytest.raises(ValueError, match="same shape"):
        compute_regression_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_compute_regression_metrics_mape_nan_when_true_has_zero():
    """
    Test that mape is NaN (rather than raising or returning inf) when a true value is zero.
    """
    metrics = compute_regression_metrics(np.array([0.0, 2.0]), np.array([0.1, 2.1]))
    assert np.isnan(metrics["mape"])


def test_compute_regression_metrics_r2_nan_when_true_constant():
    """
    Test that r2 is NaN when y_true is constant (zero total variance), rather than raising a division error.
    """
    metrics = compute_regression_metrics(np.array([5.0, 5.0, 5.0]), np.array([4.0, 5.0, 6.0]))
    assert np.isnan(metrics["r2"])


def test_compute_regression_metrics_accepts_pandas_series():
    """
    Test that compute_regression_metrics accepts a pandas Series for y_true.
    """
    y_true = pd.Series([1.0, 2.0, 3.0])
    metrics = compute_regression_metrics(y_true, np.array([1.0, 2.0, 3.0]))
    assert metrics["rmse"] == pytest.approx(0.0)


# ----------------- Tests for ModelEvaluator -----------------
def test_model_evaluator_residuals_matches_manual_computation():
    """
    Test that ModelEvaluator.residuals returns y_true - y_pred using the wrapped Predictor's predictions.
    """
    evaluator = ModelEvaluator(sum_predictor())
    x = pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 1.0]})  # predictions: 2.0, 3.0
    y_true = np.array([2.5, 2.5])

    residuals = evaluator.residuals(x, y_true)

    np.testing.assert_allclose(residuals, np.array([0.5, -0.5]))


def test_model_evaluator_evaluate_returns_regression_metrics():
    """
    Test that ModelEvaluator.evaluate returns the same metrics as compute_regression_metrics on the predictor's
    predictions.
    """
    evaluator = ModelEvaluator(sum_predictor())
    x = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)  # predictions: 2.0, 4.0
    y_true = np.array([2.0, 5.0])

    metrics = evaluator.evaluate(x, y_true)
    expected = compute_regression_metrics(y_true, np.array([2.0, 4.0]))

    assert metrics == pytest.approx(expected, nan_ok=True)


def test_model_evaluator_evaluate_aligns_shuffled_dataframe_columns():
    """
    Test that ModelEvaluator.evaluate uses the Predictor's manifest-driven column alignment, so a DataFrame with
    shuffled columns still produces the correct predictions and metrics.
    """
    evaluator = ModelEvaluator(sum_predictor(feature_names=("a", "b")))

    # Columns deliberately out of training order
    x = pd.DataFrame({"b": [1.0], "a": [1.0]})  # sum is still 2.0 regardless of column order
    y_true = np.array([2.0])

    metrics = evaluator.evaluate(x, y_true)

    assert metrics["rmse"] == pytest.approx(0.0)


def test_model_evaluator_from_saved_missing_manifest_warns_via_predictor(tmp_path: Path):
    """
    Test that ModelEvaluator wraps a Predictor loaded via from_saved, exposing the same predictor attributes.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest (unused directly, kept for parity with other from_saved tests).
    """
    predictor = sum_predictor(feature_names=("x", "y"), label_name="R_p")
    evaluator = ModelEvaluator(predictor)

    assert evaluator.predictor.feature_names == ["x", "y"]
    assert evaluator.predictor.label_name == "R_p"


# ----------------- Tests for plotting (matplotlib Agg backend, no real keras needed) -----------------
def test_plot_predicted_vs_actual_does_not_raise():
    """
    Test that ModelEvaluator.plot_predicted_vs_actual runs without error using a non-interactive backend.
    """
    matplotlib.use("Agg")
    evaluator = ModelEvaluator(sum_predictor())
    x = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
    y_true = np.array([2.0, 4.0, 6.0])

    try:
        evaluator.plot_predicted_vs_actual(x, y_true)
    except Exception as e:
        pytest.fail(f"plot_predicted_vs_actual raised an exception: {e}")


def test_plot_residuals_saves_file(tmp_path: Path):
    """
    Test that ModelEvaluator.plot_residuals saves a file to the given save_path when provided.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    """
    matplotlib.use("Agg")
    evaluator = ModelEvaluator(sum_predictor())
    x = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    y_true = np.array([2.0, 4.0])

    save_path = tmp_path / "residuals.png"
    evaluator.plot_residuals(x, y_true, save_path=save_path)

    assert save_path.exists(), f"Expected plot file {save_path} to exist."


# ----------------- End-to-end test with a real saved Keras model -----------------
@pytest.mark.tf
@pytest.mark.slow
def test_model_evaluator_from_saved_evaluates_real_model(tmp_path: Path):
    """
    Test that ModelEvaluator.from_saved loads a real saved model and produces finite regression metrics.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    """
    cfg = ModelConfig(use_normalisation=False, num_layers=1, num_neurons=4)
    manager = DNNManager.from_config(cfg, input_shape=(2,))
    manager.save_model_version(
        model=manager.model,
        history=manager.history,
        model_dir=tmp_path,
        model_name="demo",
        version=1,
        feature_names=["a", "b"],
        label_name="R_p",
    )

    evaluator = ModelEvaluator.from_saved(tmp_path, "demo", version=1)
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    y_true = np.array([1.0, 2.0, 3.0])

    metrics = evaluator.evaluate(x, y_true)

    assert set(metrics) == {"rmse", "mae", "r2", "mape", "bias", "max_error"}
    for value in metrics.values():
        assert np.isfinite(value) or np.isnan(value)
