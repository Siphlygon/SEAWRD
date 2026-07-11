"""
Evaluation utilities for trained SEAWRD models.

Provides regression metrics (RMSE, MAE, R^2, bias, ...) and diagnostic plots (predicted-vs-actual, residuals) for
judging a surrogate model's accuracy on held-out data. This complements DNNTrainer's training-time loss curves with
test-time performance summaries, and reuses Predictor so evaluation benefits from the same manifest-driven feature
alignment as inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .predictor import Predictor


def compute_regression_metrics(y_true: np.ndarray | pd.Series,
                               y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute standard regression metrics comparing predictions against true values.

    Parameters
    ----------
    y_true : np.ndarray | pd.Series
        The true (observed) values.
    y_pred : np.ndarray
        The predicted values, in the same order as ``y_true``.

    Returns
    -------
    dict[str, float]
        A dictionary with the following keys:

        - ``rmse``: root-mean-squared error.
        - ``mae``: mean absolute error.
        - ``r2``: coefficient of determination (1.0 is a perfect fit; NaN if ``y_true`` is constant).
        - ``mape``: mean absolute percentage error, as a percentage (NaN if any true value is zero).
        - ``bias``: mean signed error (``mean(y_true - y_pred)``); indicates systematic over/under-prediction.
        - ``max_error``: the largest absolute error.

    Raises
    ------
    ValueError
        If ``y_true`` and ``y_pred`` do not have the same number of elements.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}")

    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    if np.any(y_true == 0):
        mape = float("nan")
    else:
        mape = float(np.mean(np.abs(residuals / y_true)) * 100.0)

    return {
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "mae": float(np.mean(np.abs(residuals))),
        "r2": float(r2),
        "mape": mape,
        "bias": float(np.mean(residuals)),
        "max_error": float(np.max(np.abs(residuals))),
    }


class ModelEvaluator:
    """
    Evaluate a trained SEAWRD model's predictive performance on held-out data.

    Wraps a :class:`~seawrd.predictor.Predictor`, so evaluation gets the same manifest-driven column alignment as
    inference: a DataFrame's columns are matched to the training feature order automatically when a manifest is
    available.
    """

    def __init__(self, predictor: Predictor):
        """
        Initialise a ModelEvaluator wrapping an existing Predictor.

        Parameters
        ----------
        predictor : Predictor
            The predictor to evaluate. Typically obtained via :meth:`Predictor.from_saved` or built directly around
            a freshly trained, in-memory model.
        """
        self.predictor = predictor

    @classmethod
    def from_saved(cls,
                   model_dir: Path | str,
                   model_name: str,
                   version: int | None = None) -> "ModelEvaluator":
        """
        Load a saved model (and its manifest, if present) into a ModelEvaluator.

        Parameters
        ----------
        model_dir : Path | str
            The directory containing the saved model files.
        model_name : str
            The name of the model.
        version : int | None, optional
            The version number of the model to load. If None, the latest available version is loaded. By default
            None.

        Returns
        -------
        ModelEvaluator
            A ModelEvaluator wrapping the loaded model.
        """
        # Only now do we import keras by importing predictor
        from .predictor import Predictor

        return cls(Predictor.from_saved(model_dir, model_name, version))

    @staticmethod
    def _as_array(y: np.ndarray | pd.Series) -> np.ndarray:
        """
        Convert labels to a flat float64 NumPy array.

        Parameters
        ----------
        y : np.ndarray | pd.Series
            The labels to convert.

        Returns
        -------
        np.ndarray
            The labels as a flat float64 array.
        """
        if isinstance(y, pd.Series):
            return y.to_numpy(dtype=np.float64)
        return np.asarray(y, dtype=np.float64).reshape(-1)

    def residuals(self,
                 x: pd.DataFrame | np.ndarray,
                 y_true: np.ndarray | pd.Series) -> np.ndarray:
        """
        Compute residuals (``y_true - y_pred``) for the given inputs.

        Parameters
        ----------
        x : pd.DataFrame | np.ndarray
            The input features to predict on.
        y_true : np.ndarray | pd.Series
            The true values to compare predictions against.

        Returns
        -------
        np.ndarray
            The residuals, one per row of ``x``.
        """
        y_pred = self.predictor.predict(x)
        return self._as_array(y_true) - y_pred

    def evaluate(self,
                x: pd.DataFrame | np.ndarray,
                y_true: np.ndarray | pd.Series) -> dict[str, float]:
        """
        Compute regression metrics for the model's predictions on the given data.

        Parameters
        ----------
        x : pd.DataFrame | np.ndarray
            The input features to predict on.
        y_true : np.ndarray | pd.Series
            The true values to compare predictions against.

        Returns
        -------
        dict[str, float]
            The regression metrics, as returned by :func:`compute_regression_metrics`.
        """
        y_pred = self.predictor.predict(x)
        return compute_regression_metrics(self._as_array(y_true), y_pred)

    def plot_predicted_vs_actual(self,
                                 x: pd.DataFrame | np.ndarray,
                                 y_true: np.ndarray | pd.Series,
                                 save_path: Path | str | None = None):
        """
        Plot predicted values against actual values, with a diagonal reference line for a perfect prediction.

        Parameters
        ----------
        x : pd.DataFrame | np.ndarray
            The input features to predict on.
        y_true : np.ndarray | pd.Series
            The true values to compare predictions against.
        save_path : Path | str | None, optional
            If provided, the plot is saved to this path before being shown. By default None (not saved).
        """
        y_pred = self.predictor.predict(x)
        y_true_arr = self._as_array(y_true)
        label = self.predictor.label_name or "value"

        lims = [
            float(min(y_true_arr.min(), y_pred.min())),
            float(max(y_true_arr.max(), y_pred.max())),
        ]

        plt.scatter(y_true_arr, y_pred, alpha=0.4, s=10, color="indigo")
        plt.plot(lims, lims, color="black", linestyle="--", linewidth=1, label="Perfect prediction")
        plt.xlabel(f"Actual {label}")
        plt.ylabel(f"Predicted {label}")
        plt.title("Predicted vs Actual")
        plt.legend()
        plt.grid(alpha=0.3)

        if save_path is not None:
            plt.savefig(save_path, dpi=300)
        plt.show()
        plt.close()

    def plot_residuals(self,
                       x: pd.DataFrame | np.ndarray,
                       y_true: np.ndarray | pd.Series,
                       save_path: Path | str | None = None):
        """
        Plot residuals against predicted values, with a horizontal reference line at zero error.

        Parameters
        ----------
        x : pd.DataFrame | np.ndarray
            The input features to predict on.
        y_true : np.ndarray | pd.Series
            The true values to compare predictions against.
        save_path : Path | str | None, optional
            If provided, the plot is saved to this path before being shown. By default None (not saved).
        """
        y_pred = self.predictor.predict(x)
        residuals = self._as_array(y_true) - y_pred
        label = self.predictor.label_name or "value"

        plt.scatter(y_pred, residuals, alpha=0.4, s=10, color="seagreen")
        plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
        plt.xlabel(f"Predicted {label}")
        plt.ylabel("Residual (actual - predicted)")
        plt.title("Residuals vs Predicted")
        plt.grid(alpha=0.3)

        if save_path is not None:
            plt.savefig(save_path, dpi=300)
        plt.show()
        plt.close()
