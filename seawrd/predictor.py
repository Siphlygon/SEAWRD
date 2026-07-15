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


class EnsemblePredictor:
    """
    Run predictions with an ensemble of trained SEAWRD models, reporting per-sample uncertainty.

    Where :class:`Predictor` wraps a single saved model, ``EnsemblePredictor`` wraps every model trained in one
    :meth:`~seawrd.trainer.DNNTrainer.train_models` call (saved when ``output.save_ensemble`` is enabled), each as
    its own :class:`Predictor`. Because the members were trained from different random initialisations on the same
    data, the *spread* between their individual predictions on a new sample is a useful (epistemic) uncertainty
    estimate -- it tends to be larger where the training data was sparse or the problem harder to fit, and small
    where the ensemble agrees.

    ``predict()`` returns just the ensemble mean, so an ``EnsemblePredictor`` can be used anywhere a ``Predictor``
    is expected (e.g. with :class:`~seawrd.evaluation.ModelEvaluator`); use :meth:`predict_with_uncertainty` or
    :meth:`predict_dataframe` to also get the per-sample standard deviation across members.
    """

    def __init__(self, predictors: Sequence[Predictor], label_name: str | None = None):
        """
        Initialise an EnsemblePredictor from a non-empty sequence of member Predictors.

        Most users should prefer :meth:`from_saved`, which loads all ensemble members from disk. This constructor is
        useful for wrapping freshly trained, in-memory models.

        Parameters
        ----------
        predictors : Sequence[Predictor]
            The ensemble member predictors. All members are expected to share the same feature order (they should
            come from the same training run).
        label_name : str | None, optional
            The name of the predicted label. Defaults to the first member's label name if not given.

        Raises
        ------
        ValueError
            If no member predictors are provided.
        """
        if not predictors:
            raise ValueError("EnsemblePredictor requires at least one member Predictor")

        self.predictors = list(predictors)
        self.feature_names = self.predictors[0].feature_names
        self.label_name = label_name or self.predictors[0].label_name

    @classmethod
    def from_saved(cls,
                   model_dir: Path | str,
                   model_name: str,
                   version: int | None = None) -> "EnsemblePredictor":
        """
        Load a saved ensemble into an EnsemblePredictor.

        Parameters
        ----------
        model_dir : Path | str
            The directory containing the saved model files.
        model_name : str
            The name of the ensemble's parent model (the same name the single best model was saved under).
        version : int | None, optional
            The version number of the ensemble to load. If None, the latest available version is loaded. By default
            None.

        Returns
        -------
        EnsemblePredictor
            An EnsemblePredictor wrapping every saved ensemble member.

        Raises
        ------
        FileNotFoundError
            If no ensemble manifest is found for the resolved model name/version (i.e. the model was not trained
            with ``output.save_ensemble`` enabled).
        """
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)

        if version is None:
            # A cheap, model-free instance: get_latest_version only inspects file names on disk.
            version = DNNManager(model=None, history=None, version=0, model_name=model_name).get_latest_version(
                model_dir, model_name)

        manifest = DNNManager.load_ensemble_manifest(model_dir, model_name, version)
        if manifest is None:
            raise FileNotFoundError(
                f"No ensemble manifest found for model '{model_name}' version {version} in {model_dir}. "
                "Train with output.save_ensemble=true to produce one."
            )

        member_predictors = [
            Predictor.from_saved(model_dir, member_name, version)
            for member_name in manifest["member_names"]
        ]

        return cls(member_predictors, label_name=manifest.get("label_name"))

    def predict_with_uncertainty(self,
                                 x: pd.DataFrame | np.ndarray,
                                 batch_size: int | None = None,
                                 verbose: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict with every ensemble member and summarise the results as a mean and standard deviation per sample.

        Parameters
        ----------
        x : pd.DataFrame | np.ndarray
            The input features to predict on, as accepted by :meth:`Predictor.predict`.
        batch_size : int | None, optional
            The batch size passed to each member's ``keras.Model.predict``. By default None.
        verbose : int, optional
            Verbosity mode passed to each member's ``keras.Model.predict``. By default 0.

        Returns
        -------
        mean : np.ndarray
            The mean prediction across ensemble members, one per input row.
        std : np.ndarray
            The standard deviation of predictions across ensemble members (per-sample uncertainty), one per input
            row.
        """
        member_predictions = np.stack(
            [predictor.predict(x, batch_size=batch_size, verbose=verbose) for predictor in self.predictors],
            axis=0,
        )
        return member_predictions.mean(axis=0), member_predictions.std(axis=0)

    def predict(self,
               x: pd.DataFrame | np.ndarray,
               batch_size: int | None = None,
               verbose: int = 0) -> np.ndarray:
        """
        Predict the ensemble mean for the provided input features.

        Provided so an EnsemblePredictor can be used as a drop-in replacement for a single Predictor wherever only a
        point prediction is needed (e.g. with ModelEvaluator); use :meth:`predict_with_uncertainty` to also get the
        per-sample spread across members.

        Parameters
        ----------
        x : pd.DataFrame | np.ndarray
            The input features to predict on.
        batch_size : int | None, optional
            The batch size passed to each member's ``keras.Model.predict``. By default None.
        verbose : int, optional
            Verbosity mode passed to each member's ``keras.Model.predict``. By default 0.

        Returns
        -------
        np.ndarray
            The ensemble mean prediction, one per input row.
        """
        mean, _ = self.predict_with_uncertainty(x, batch_size=batch_size, verbose=verbose)
        return mean

    def predict_dataframe(self,
                          df: pd.DataFrame,
                          prediction_column: str | None = None,
                          uncertainty_column: str | None = None,
                          batch_size: int | None = None,
                          verbose: int = 0) -> pd.DataFrame:
        """
        Predict on a DataFrame and return a copy with the ensemble mean and uncertainty attached as new columns.

        Parameters
        ----------
        df : pd.DataFrame
            The input features. Columns are aligned to the ensemble's expected feature order when known.
        prediction_column : str | None, optional
            The name of the column to store the ensemble mean in. Defaults to ``"{label_name}_pred"`` when the
            label name is known, otherwise ``"prediction"``. By default None.
        uncertainty_column : str | None, optional
            The name of the column to store the per-sample standard deviation in. Defaults to
            ``"{label_name}_std"`` when the label name is known, otherwise ``"uncertainty"``. By default None.
        batch_size : int | None, optional
            The batch size passed to each member's ``keras.Model.predict``. By default None.
        verbose : int, optional
            Verbosity mode passed to each member's ``keras.Model.predict``. By default 0.

        Returns
        -------
        pd.DataFrame
            A copy of the input DataFrame with prediction and uncertainty columns appended.

        Raises
        ------
        ValueError
            If the underlying models produce multiple outputs, which cannot be stored in a single pair of columns.
        """
        mean, std = self.predict_with_uncertainty(df, batch_size=batch_size, verbose=verbose)

        if mean.ndim > 1:
            raise ValueError(
                "predict_dataframe supports single-output models only; use predict_with_uncertainty for "
                "multi-output models."
            )

        if prediction_column is None:
            prediction_column = f"{self.label_name}_pred" if self.label_name else "prediction"
        if uncertainty_column is None:
            uncertainty_column = f"{self.label_name}_std" if self.label_name else "uncertainty"

        result = df.copy()
        result[prediction_column] = mean
        result[uncertainty_column] = std
        return result


if __name__ == "__main__":
    # Minimal end-to-end example: build a tiny model, wrap it, and predict on a DataFrame.
    demo_features = ["a", "b", "c"]
    demo_model = keras.Sequential([keras.Input(shape=(3,)), keras.layers.Dense(1)])

    predictor = Predictor(model=demo_model, feature_names=demo_features, label_name="R_p")

    # Note the deliberately shuffled column order: the Predictor realigns it to the training order.
    sample = pd.DataFrame({"c": [3.0, 6.0], "a": [1.0, 4.0], "b": [2.0, 5.0]})
    print(predictor.predict_dataframe(sample))
