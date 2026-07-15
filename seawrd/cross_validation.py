"""
K-fold cross-validation for SEAWRD models.

Where DNNTrainer trains several randomly-initialised models on a single train/validation split and keeps the best,
KFoldCrossValidator instead trains on several different train/validation *splits* of the same data (via
DataPreprocessor.k_fold_splits) and reports how performance varies across them. This is a better estimate of how a
model architecture/configuration generalises than any single held-out split, at the cost of training n_splits times
as many models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .config import SEAWRDConfig
from .utils import fit_normaliser, get_logger

if TYPE_CHECKING:
    from .preprocessing_data import DataPreprocessor

logger = get_logger("seawrd.cross_validation")


@dataclass
class CrossValidationResult:
    """
    The outcome of a K-fold cross-validation run: per-fold regression metrics and training histories.
    """
    fold_metrics: list[dict[str, float]] = field(default_factory=list)
    fold_histories: list[Any] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return the per-fold metrics as a DataFrame, one row per fold.

        Returns
        -------
        pd.DataFrame
            A DataFrame with one row per fold and one column per metric (rmse, mae, r2, mape, bias, max_error).
        """
        return pd.DataFrame(self.fold_metrics)

    def summary(self) -> dict[str, dict[str, float]]:
        """
        Summarise each metric's mean and standard deviation across folds.

        A large standard deviation relative to the mean indicates the model's performance is sensitive to which
        rows end up in the training vs. validation set -- a sign the single train/test split used elsewhere may be
        misleading, or that more data/regularisation is needed.

        Returns
        -------
        dict[str, dict[str, float]]
            A mapping from metric name to ``{"mean": ..., "std": ...}``.
        """
        df = self.to_dataframe()
        return {
            column: {"mean": float(df[column].mean()), "std": float(df[column].std())}
            for column in df.columns
        }


class KFoldCrossValidator:
    """
    Run K-fold cross-validation for a SEAWRD model configuration.

    Model saving is intentionally disabled for every fold (regardless of the passed-in config's output section):
    cross-validation is for estimating how well a configuration generalises, not for producing a deployable model.
    Once you're happy with a configuration, train it normally with DNNTrainer to produce a model worth saving.
    """

    def __init__(self,
                 config: SEAWRDConfig,
                 n_splits: int = 5,
                 shuffle: bool = True,
                 random_state: int | None = None):
        """
        Initialise a KFoldCrossValidator.

        Parameters
        ----------
        config : SEAWRDConfig
            The configuration to cross-validate. Its model/training/compile/callbacks sections are used as-is for
            every fold; its output section is overridden (see class docstring) so folds are never saved to disk.
        n_splits : int, optional
            The number of folds to split the data into, by default 5.
        shuffle : bool, optional
            Whether to shuffle the data before splitting into folds, by default True.
        random_state : int | None, optional
            Random seed for the fold shuffle, by default None.

        Raises
        ------
        ValueError
            If n_splits is less than 2.
        """
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")

        self.config = config
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def run(self, preprocessor: "DataPreprocessor") -> CrossValidationResult:
        """
        Run cross-validation using a prepared DataPreprocessor's cleaned data.

        For each fold, a fresh model is built and trained (via DNNManager/DNNTrainer, honouring
        config.training.num_models exactly as a normal training run would) on that fold's training split, then
        evaluated on its held-out validation split.

        Parameters
        ----------
        preprocessor : DataPreprocessor
            A DataPreprocessor that has already validated, filtered, and derived columns for the input data (i.e.
            an ordinary, already-constructed DataPreprocessor instance). Its k_fold_splits method is used to
            generate the folds; its single train/test split (from split()/get_training_data()) is not used.

        Returns
        -------
        CrossValidationResult
            The per-fold metrics and training histories.
        """
        # Only now do we import keras by importing model/trainer/predictor/evaluation
        from .evaluation import ModelEvaluator
        from .model import DNNManager
        from .predictor import Predictor
        from .trainer import DNNTrainer

        # Folds are transient by design; never let a stray output config write dozens of models/plots to disk.
        fold_config = self.config.with_update_section(
            "output", save_model=False, save_plots=False, save_ensemble=False)

        fold_metrics: list[dict[str, float]] = []
        fold_histories: list[Any] = []

        folds = preprocessor.k_fold_splits(self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        for fold_idx, (train_features, val_features, train_labels, val_labels) in enumerate(folds):
            logger.info("Cross-validation fold %d/%d", fold_idx + 1, self.n_splits)

            x_train = train_features.to_numpy(dtype=np.float32)
            y_train = train_labels.to_numpy(dtype=np.float32)
            x_val = val_features.to_numpy(dtype=np.float32)
            y_val = val_labels.to_numpy(dtype=np.float32)

            normaliser = fit_normaliser(x_train) if fold_config.model.use_normalisation else None

            manager = DNNManager.from_config(
                model_config=fold_config.model,
                input_shape=x_train.shape[1:],
                normaliser=normaliser,
            )
            trainer = DNNTrainer(model_manager=manager, config=fold_config)
            trainer.train_models(
                input_features=x_train,
                input_labels=y_train,
                test_features=x_val,
                test_labels=y_val,
            )

            predictor = Predictor(
                model=trainer.best_model,
                feature_names=preprocessor.feature_names_,
                label_name=preprocessor.label_name_,
            )
            metrics = ModelEvaluator(predictor).evaluate(val_features, val_labels)

            fold_metrics.append(metrics)
            fold_histories.append(trainer.best_history)

        return CrossValidationResult(fold_metrics=fold_metrics, fold_histories=fold_histories)
