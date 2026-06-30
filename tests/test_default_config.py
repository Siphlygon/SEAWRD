"""
Tests for the default configuration packaged with the seawrd module. These tests ensure that the default configuration
can be loaded correctly and that it contains valid settings.    
"""

from importlib.resources import files
from seawrd.config_manager import ConfigManager


def test_packaged_default_config_loads():
    """
    Test that the default configuration packaged with the seawrd module can be loaded into a ConfigManager.
    """
    path = files("seawrd") / "seawrd_default.toml"
    manager = ConfigManager.from_toml(path)

    assert manager.config.training.num_epochs > 0
    assert manager.config.compile.optimiser == "adam"


def test_packaged_default_config_has_valid_settings():
    """
    Test that the default configuration packaged with the seawrd module contains valid settings.
    """
    path = files("seawrd") / "seawrd_default.toml"
    manager = ConfigManager.from_toml(path)

    # Check that the model configuration has valid values
    assert manager.config.model.num_layers > 0
    assert manager.config.model.num_neurons > 0
    assert manager.config.model.activation in ["relu", "sigmoid", "tanh"]

    # Check that the training configuration has valid values
    assert manager.config.training.num_epochs > 0
    assert manager.config.training.batch_size > 0
    assert 0 < manager.config.training.validation_split < 1

    # Check that the compile configuration has valid values
    assert manager.config.compile.optimiser in ["adam", "sgd"]
    assert manager.config.compile.loss in ["mean_squared_error", "categorical_crossentropy"]

    # Check that the callbacks configuration has valid values
    assert isinstance(manager.config.callbacks.reduce_lr, bool)
    assert isinstance(manager.config.callbacks.early_stopping, bool)

    # Check that the device configuration has valid values
    assert manager.config.device.mode in ["auto", "cpu", "gpu"]

    # Check that the output configuration has valid values
    assert isinstance(manager.config.output.save_model, bool)
