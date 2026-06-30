from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import keras

from seawrd.config import SEAWRDConfig
from seawrd.model import DNNManager
from seawrd.preprocessing_data import DataPreprocessor
from seawrd.trainer import DNNTrainer


def create_minimal_config() -> SEAWRDConfig:
    """
    Create a minimal SEAWRDConfig for testing purposes.

    Returns
    -------
    SEAWRDConfig
        A minimal configuration object.
    """
    return SEAWRDConfig.from_dict({
        "model": {
            "num_layers": 1,
            "num_neurons": 4,
            "num_outputs": 1,
            "activation": "relu",
            "use_normalisation": True,
        },
        "training": {
            "num_epochs": 2,
            "batch_size": 8,
            "validation_split": 0.25,
            "num_models": 1,
            "shuffle": False,
        },
        "compile": {
            "learning_rate": 0.01,
            "metrics": ["mean_squared_error"],
        },
        "callbacks": {
            "reduce_lr": False,
            "early_stopping": False,
        },
        "output": {
            "save_model": False,
            "save_plots": False,
        },
    })


def create_minimal_dataset(n_samples: int = 64, random_state: int = 123) \
    -> tuple[keras.layers.Normalization, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create a minimal dataset for testing purposes.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples to generate, by default 64
    random_state : int, optional
        Random seed for reproducibility, by default 123

    Returns
    -------
    keras.layers.Normalization
        A fitted Keras Normalization layer.
    pd.DataFrame
        The training features.
    pd.DataFrame
        The test features.
    pd.Series
        The training labels.
    pd.Series
        The test labels.
    """
    rng = np.random.default_rng(random_state)

    df = pd.DataFrame({
        "x_core'": rng.uniform(0.1, 0.9, n_samples),
        "x_H2O": rng.uniform(0.1, 0.9, n_samples),
        "T_irr": rng.uniform(100, 500, n_samples),
        "T_b": rng.uniform(150, 650, n_samples),
        "M_b": rng.uniform(0.5, 3.0, n_samples),
        "M_a": rng.uniform(1.0, 6.0, n_samples),
        "R_b": rng.uniform(0.5, 3.0, n_samples),
        "R_a": rng.uniform(1.0, 6.0, n_samples),
        "errcode": np.zeros(n_samples, dtype=int),
    })

    preprocessor = DataPreprocessor(
        df,
        features=["x_core'", "x_H2O", "T_irr", "T_b", "M_b", "M_a"],
        label="R_p",
        test_size=0.25,
        random_state=123,
        normalise=True,
    )

    normaliser, x_train, x_test, y_train, y_test = preprocessor.get_training_data(
        return_array=False
    )

    return normaliser, x_train, x_test, y_train, y_test # type: ignore


@pytest.mark.tf
@pytest.mark.slow
def test_minimal_training_pipeline_runs_end_to_end():
    """
    Test that the minimal training pipeline runs end-to-end without errors, including data preprocessing, model
    creation, training, and evaluation.
    
    This test uses a small dataset and a simple model configuration to ensure that the entire pipeline can execute
    successfully.
    """
    normaliser, x_train, x_test, y_train, y_test = create_minimal_dataset(64, 123)

    cfg = create_minimal_config()

    manager = DNNManager.from_config(
        cfg.model,
        input_shape=(x_train.shape[1],),
        normaliser=normaliser,
    )

    trainer = DNNTrainer(manager, cfg)

    pred_means, pred_stds, losses, val_losses = trainer.train_models(
        input_features=x_train,
        input_labels=y_train,
        test_features=x_test,
        test_labels=y_test,
    )

    assert trainer._trained
    assert trainer.best_model is not None

    assert pred_means.shape == (1,)
    assert pred_stds.shape == (1,)
    assert losses.shape == (1,)
    assert val_losses.shape == (1,)

    assert np.all(np.isfinite(pred_means))
    assert np.all(np.isfinite(pred_stds))
    assert np.all(np.isfinite(losses))
    assert np.all(np.isfinite(val_losses))

    assert "loss" in trainer.best_history.history
    assert "val_loss" in trainer.best_history.history


@pytest.mark.tf
@pytest.mark.slow
def test_training_pipeline_can_save_and_reload_best_model(tmp_path : Path):
    """
    Test that the training pipeline can save the best model and reload it correctly.

    Parameters
    ----------
    tmp_path : pathlib.Path
        A temporary directory provided by pytest for saving the model and history files.
    """
    normaliser, x_train, x_test, y_train, y_test = create_minimal_dataset(64, 123)

    cfg = create_minimal_config()
    new_cfg = cfg.with_update(
        output = {
            "model_dir": str(tmp_path),
            "version": 1,
            "save_model": True,
            "save_plots": False,
        }
    )

    manager = DNNManager.from_config(
        new_cfg.model,
        input_shape=(x_train.shape[1],),
        normaliser=normaliser,
    )
    trainer = DNNTrainer(manager, new_cfg)

    trainer.train_models(
        input_features=x_train,
        input_labels=y_train,
        test_features=x_test,
        test_labels=y_test,
    )

    saved_model = tmp_path / f"{trainer.model_name}_v{trainer.version}_model.keras"
    saved_history = tmp_path / f"{trainer.model_name}_v{trainer.version}_history.pkl"

    assert saved_model.exists()
    assert saved_history.exists()

    loaded_manager = DNNManager.from_previous_model(
        model_dir=tmp_path,
        model_name=trainer.model_name,
        version=trainer.version,
    )

    x = x_test.to_numpy(dtype=np.float32)

    np.testing.assert_allclose(
        trainer.best_model.predict(x, verbose=0),
        loaded_manager.model.predict(x, verbose=0),
        err_msg="Predictions from the original and loaded models do not match.",
    )
