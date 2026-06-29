"""
Unit tests for the DNNTrainer class in the seawrd.trainer module. These tests cover various functionalities of the
DNNTrainer, including rounding to significant figures, generating callbacks, evaluating models, and printing
architecture performance.
"""
import numpy as np
import pandas as pd
import keras

from seawrd.config import SEAWRDConfig
from seawrd.model import DNNManager
from seawrd.trainer import DNNTrainer


class DummyModel:
    """
    A dummy model class for testing purposes. It simulates a model that always predicts a fixed output.
    """
    def predict(self, x):
        """
        Simulate predictions by returning a fixed array of values.

        Parameters
        ----------
        x : array-like
            The input features for which predictions are to be made.

        Returns
        -------
        np.ndarray
            The fixed predicted values.
        """
        return np.array([[1.0], [2.0], [3.0]])


def trainer_with_default_config():
    """
    Create a DNNTrainer instance with a default configuration for testing purposes.

    Returns
    -------
    DNNTrainer
        A DNNTrainer instance with a default configuration.
    """
    cfg = SEAWRDConfig.from_dict({
        "training": {"num_epochs": 1, "batch_size": 2, "num_models": 1},
        "output": {"save_model": False},
        "model": {"use_normalisation": False},
    })
    manager = DNNManager.from_config(cfg.model, input_shape=(2,))
    return DNNTrainer(manager, cfg)


def test_round_to_sig_figs():
    """
    Test the _round_to_n_sig_figs method of DNNTrainer to ensure it rounds numbers to the specified number of
    significant figures.
    """
    trainer = trainer_with_default_config()

    assert trainer._round_to_n_sig_figs(1234.567, 3) == 1230
    assert trainer._round_to_n_sig_figs(0, 3) == 0.0


def test_generate_callbacks_respects_disabled_flags():
    """
    Test that the _generate_callbacks method of DNNTrainer respects the flags for disabling learning rate scheduler
    and early stopping, and does not include those callbacks when the flags are set to False.
    """
    trainer = trainer_with_default_config()

    callbacks = trainer._generate_callbacks(
        use_lr_scheduler=False,
        use_early_stopping=False,
    )

    assert not any(isinstance(cb, keras.callbacks.ReduceLROnPlateau) for cb in callbacks)
    assert not any(isinstance(cb, keras.callbacks.EarlyStopping) for cb in callbacks)


def test_evaluate_model_returns_actual_minus_prediction():
    """
    Test that the _evaluate_model method of DNNTrainer returns the difference between actual and predicted values.
    """
    trainer = trainer_with_default_config()
    features = pd.DataFrame({"a": [0, 0, 0]})
    labels = pd.Series([2.0, 2.0, 2.0])

    error = trainer._evaluate_model(DummyModel(), features, labels)

    np.testing.assert_allclose(error, np.array([1.0, 0.0, -1.0]))


def test_print_architecture_requires_training():
    """
    Test that the print_architecture_performance method of DNNTrainer raises a ValueError if the model has not been
    trained yet.
    """
    trainer = trainer_with_default_config()

    try:
        trainer.print_architecture_performance()
    except ValueError as e:
        assert str(e) == "The model has not been trained yet. Please train the model before printing performance " \
        + "metrics."
