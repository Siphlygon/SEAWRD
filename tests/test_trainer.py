"""
Unit tests for the DNNTrainer class in the seawrd.trainer module. These tests cover various functionalities of the
DNNTrainer, including rounding to significant figures, generating callbacks, evaluating models, and printing
architecture performance.
"""
import os

import numpy as np
import pandas as pd
import keras
import pytest
import matplotlib

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
        "model": {
            "use_normalisation": False  # Disable normalization for speed in tests
        },
        "output": {
            "save_model": False, 
            "save_plots": False 
        }
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


def test_outputs_requires_training():
    """
    Test that the print_architecture_performance and plot_loss_curve methods of DNNTrainer raise a ValueError if the
    model has not been trained yet.
    """
    trainer = trainer_with_default_config()

    with pytest.raises(AssertionError):
        trainer.print_architecture_performance()

    with pytest.raises(AssertionError):
        trainer.plot_loss_curve()


def test_print_architecture_performance_prints_expected_output(capsys):
    """
    Test that the print_architecture_performance method of DNNTrainer prints the expected output when the model has
    been trained. This test captures the printed output and checks for specific substrings.
    """
    trainer = trainer_with_default_config()
    DNNManager.compile_from_config(trainer.model_manager.model, trainer.compile_config)  # Compile the model to avoid errors
    trainer._trained = True  # Simulate that the model has been trained

    # Give some dummy values to the architecture performance attributes for testing
    trainer.best_model = trainer.model_manager.model
    trainer.losses = np.array([0.5, 0.4, 0.3])
    trainer.val_losses = np.array([0.6, 0.5, 0.4])
    trainer.list_num_epoch = np.array([100, 200, 300])

    trainer.print_architecture_performance()

    captured = capsys.readouterr()
    assert "name" in captured.out
    assert "num_param" in captured.out
    assert "num_epoch_mean" in captured.out
    assert "loss_min" in captured.out
    assert "loss_max" in captured.out
    assert "loss_mean" in captured.out
    assert "loss_stdev" in captured.out
    assert "val_loss_min" in captured.out
    assert "val_loss_max" in captured.out
    assert "val_loss_mean" in captured.out
    assert "val_loss_stdev" in captured.out


def test_plot_loss_curve_does_not_raise_error():
    """
    Test that the plot_loss_curve method of DNNTrainer does not raise any errors when called after the model has
    been trained. This test ensures that the method can execute without issues.
    """
    trainer = trainer_with_default_config()
    DNNManager.compile_from_config(trainer.model_manager.model, trainer.compile_config)
    trainer._trained = True  # Simulate that the model has been trained

    # Give some dummy values to the architecture performance attributes for testing
    trainer.best_history.history = {
        "loss": [0.5, 0.4, 0.3],
        "val_loss": [0.6, 0.5, 0.4],
    }

    # Stop the plot from displaying during tests by using a non-interactive backend
    matplotlib.use("Agg")

    try:
        trainer.plot_loss_curve()
    except Exception as e:
        pytest.fail(f"plot_loss_curve raised an exception: {e}")


def test_plot_loss_curve_saves_file(tmp_path):
    """
    Test that the plot_loss_curve method of DNNTrainer saves the loss curve plot to a file when the save_plots flag is
    set to True and does not when save_plots is set to False in the output configuration.
    """
    trainer = trainer_with_default_config()
    trainer.output_config = trainer.output_config.with_update(save_plots=True)
    DNNManager.compile_from_config(trainer.model_manager.model, trainer.compile_config)
    trainer._trained = True  # Simulate that the model has been trained

    # Give some dummy values to the architecture performance attributes for testing
    trainer.best_history.history = {
        "loss": [0.5, 0.4, 0.3],
        "val_loss": [0.6, 0.5, 0.4],
    }

    # Stop the plot from displaying during tests by using a non-interactive backend
    matplotlib.use("Agg")

    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    trainer.plot_loss_curve()
    expected_file = tmp_path / f"{trainer.best_model.name}_plot_loss.png"
    assert expected_file.exists(), f"Expected plot file {expected_file} does not exist."

    # delete the file to test the case when save_plots is False
    expected_file.unlink()

    # Now test with save_plots set to False
    trainer.output_config = trainer.output_config.with_update(save_plots=False)
    trainer.plot_loss_curve()
    assert not expected_file.exists(), f"Plot file {expected_file} should not exist when save_plots is False."

    # Restore the original working directory
    os.chdir(original_cwd)


def test_validation_split_created_properly():
    """
    Test that the _create_validation_split method of DNNTrainer creates a validation split correctly based on the
    specified validation split ratio. This test checks that the resulting training and validation sets have the
    expected sizes.
    """
    trainer = trainer_with_default_config()  # default validation_split is 0.2
    features = pd.DataFrame({"a": np.arange(100)})
    labels = pd.Series(np.arange(100))

    train_features, train_labels, val_features, val_labels = trainer._create_validation_split(
        features, labels
    )

    assert len(train_features) == 80
    assert len(val_features) == 20
    assert len(train_labels) == 80
    assert len(val_labels) == 20
