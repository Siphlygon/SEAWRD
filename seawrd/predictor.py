from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
import pandas as pd

from .model import DNNManager


class Predictor:
    """
    Run predictions with a trained SEAWRD model.

    This is the inference counterpart to :class:`~seawrd.trainer.DNNTrainer`: where the trainer produces and saves a
    model, the ``Predictor`` loads one and turns new planetary data into radius predictions. Normalisation of feature
    vectors, if used, is handled automatically in the first layer of the model, so raw features can be passed straight
    through. 

    A ``Predictor`` optionally carries a *manifest* describing the feature names and ordering the model was trained on
    (see :meth:`~seawrd.model.DNNManager.save_manifest`). When present, prediction on a :class:`pandas.DataFrame`
    selects and reorders columns to match the training order and errors loudly on missing features, which makes
    inference robust to differently-arranged input tables.
    """

    def __init__(self,
                 model: keras.Model,
                 feature_names: Sequence[str] | None = None,
                 label_name: str | None = None,
                 manifest: dict[str, Any] | None = None):
        """
        Initialise a Predictor from an in-memory Keras model.

        Most users should prefer :meth:`from_saved`, which loads a model and its manifest from disk. This constructor is
        useful for wrapping a freshly trained model, or in tests.

        Parameters
        ----------
        model : keras.Model
            The trained Keras model to run predictions with.
        feature_names : Sequence[str] | None, optional
            The names of the input features, in the order the model expects them. Required for DataFrame prediction if
            no manifest is provided. By default None.
        label_name : str | None, optional
            The name of the predicted label (e.g. ``"R_p"``), used to name the output. By default None.
        manifest : dict[str, Any] | None, optional
            A manifest dictionary as produced by :meth:`~seawrd.model.DNNManager.save_manifest`. When provided, its
            ``feature_names`` and ``label_name`` take precedence over the explicit arguments. By default None.
        """
        self.model = model
        self.manifest = manifest

        if manifest is not None:
            self.feature_names = list(manifest.get("feature_names")) if manifest.get("feature_names") else None
            self.label_name = manifest.get("label_name", label_name)
        else:
            self.feature_names = list(feature_names) if feature_names is not None else None
            self.label_name = label_name


    @classmethod
    def from_saved(cls,
                   model_dir: Path | str,
                   model_name: str,
                   version: int | None = None) -> "Predictor":
        """
        Load a saved model (and its manifest, if present) into a Predictor.

        Parameters
        ----------
        model_dir : Path | str
            The directory containing the saved model files.
        model_name : str
            The name of the model.
        version : int | None, optional
            The version number of the model to load. If None, the latest available version is loaded. By default None.

        Returns
        -------
        Predictor
            A Predictor wrapping the loaded model and any accompanying manifest.
        """
        manager = DNNManager.from_previous_model(model_dir, model_name, version)
        manifest = DNNManager.load_manifest(model_dir, model_name, manager.version)

        return cls(model=manager.model, manifest=manifest)


    def _prepare_features(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Validate input features and return a float32 array in the model's expected column order.

        Parameters
        ----------
        x : pd.DataFrame | np.ndarray
            The input features. A DataFrame is aligned to the known feature names (selected and reordered); an array is
            assumed to already be in training order.

        Returns
        -------
        np.ndarray
            A 2D float32 array of features ready to feed to the model.

        Raises
        ------
        ValueError
            If a DataFrame is missing any required features, or if the number of input features does not match what the
            model expects.
        """
        expected = self.model.input_shape[-1] if self.model.input_shape else None

        if isinstance(x, pd.DataFrame):
            if self.feature_names is not None:
                missing = [name for name in self.feature_names if name not in x.columns]
                if missing:
                    raise ValueError(f"Input is missing required features: {missing}")
                x = x[self.feature_names]
            values = x.to_numpy(dtype=np.float32)
        else:
            values = np.asarray(x, dtype=np.float32)

        # Promote a single sample (1D) to a batch of one so the model always receives a 2D array
        if values.ndim == 1:
            values = values.reshape(1, -1)

        if expected is not None and values.shape[-1] != expected:
            raise ValueError(
                f"Model expects {expected} features but received {values.shape[-1]}. "
                "Provide features in the order the model was trained on, or supply a DataFrame with named columns."
            )

        return values


    def predict(self,
                x: pd.DataFrame | np.ndarray,
                batch_size: int | None = None,
                verbose: int = 0) -> np.ndarray:
        """
        Predict planet radii for the provided input features.

        Parameters
        ----------
        x : pd.DataFrame | np.ndarray
            The input features. If a DataFrame and feature names are known, columns are selected and reordered to match
            the training order. If an array, features are assumed to already be in training order.
        batch_size : int | None, optional
            The batch size passed to ``keras.Model.predict``. By default None (Keras chooses).
        verbose : int, optional
            Verbosity mode passed to ``keras.Model.predict``. By default 0 (silent).

        Returns
        -------
        np.ndarray
            The predictions. For a single-output model this is a 1D array with one entry per input row; for a
            multi-output model it is a 2D array of shape ``(n_rows, n_outputs)``.
        """
        features = self._prepare_features(x)
        predictions = self.model.predict(features, batch_size=batch_size, verbose=verbose)  # type: ignore

        # Collapse a single-output column into a flat 1D array for convenience
        if predictions.ndim == 2 and predictions.shape[-1] == 1:
            predictions = predictions.reshape(-1)

        return predictions


    def predict_dataframe(self,
                          df: pd.DataFrame,
                          prediction_column: str | None = None,
                          batch_size: int | None = None,
                          verbose: int = 0) -> pd.DataFrame:
        """
        Predict on a DataFrame and return a copy with the predictions attached as a new column.

        This preserves the input DataFrame's index and all its original columns, which is convenient for joining
        predictions back onto a catalogue of planets.

        Parameters
        ----------
        df : pd.DataFrame
            The input features. Columns are aligned to the model's expected feature order when the feature names are
            known.
        prediction_column : str | None, optional
            The name of the column to store predictions in. Defaults to ``"{label_name}_pred"`` when the label name is
            known, otherwise ``"prediction"``. By default None.
        batch_size : int | None, optional
            The batch size passed to ``keras.Model.predict``. By default None.
        verbose : int, optional
            Verbosity mode passed to ``keras.Model.predict``. By default 0.

        Returns
        -------
        pd.DataFrame
            A copy of the input DataFrame with a prediction column appended.

        Raises
        ------
        ValueError
            If the model produces multiple outputs, which cannot be stored in a single column.
        """
        predictions = self.predict(df, batch_size=batch_size, verbose=verbose)

        if predictions.ndim > 1:
            raise ValueError(
                "predict_dataframe supports single-output models only; use predict() for multi-output models."
            )

        if prediction_column is None:
            prediction_column = f"{self.label_name}_pred" if self.label_name else "prediction"

        result = df.copy()
        result[prediction_column] = predictions
        return result


if __name__ == "__main__":
    # Minimal end-to-end example: build a tiny model, wrap it, and predict on a DataFrame.
    demo_features = ["a", "b", "c"]
    demo_model = keras.Sequential([keras.Input(shape=(3,)), keras.layers.Dense(1)])

    predictor = Predictor(model=demo_model, feature_names=demo_features, label_name="R_p")

    # Note the deliberately shuffled column order: the Predictor realigns it to the training order.
    sample = pd.DataFrame({"c": [3.0, 6.0], "a": [1.0, 4.0], "b": [2.0, 5.0]})
    print(predictor.predict_dataframe(sample))
