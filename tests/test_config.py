"""
Unit tests for the seawrd.config module, which contains dataclasses for configuration settings used in the SEAWRD
package. These tests ensure that the configuration classes behave as expected, including validation of input values and
proper handling of default settings.
"""

import pytest
from seawrd.config import (
    ModelConfig, TrainingConfig, CompileConfig,
    CallbackConfig, DeviceConfig, OutputConfig, SEAWRDConfig
)


# ---------- Tests for invalid values in configuration dataclasses ----------
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"num_layers": -1}, "num_layers"),
        ({"num_neurons": 0}, "num_neurons"),
        ({"num_outputs": 0}, "num_outputs"),
        ({"activation": "not_a_real_activation"}, "activation"),
    ],
)
def test_model_config_rejects_invalid_values(kwargs : dict, message : str):
    """
    Test that ModelConfig raises ValueError for invalid parameter values.

    Parameters
    ----------
    kwargs : dict
        The keyword arguments to pass to the ModelConfig constructor.
    message : str
        The expected error message.
    """
    with pytest.raises((ValueError, NotImplementedError), match=message):
        ModelConfig(**kwargs)
        ModelConfig.from_dict(kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_epochs": 0},
        {"batch_size": 0},
        {"validation_split": 0},
        {"validation_split": 1},
        {"num_models": 0},
    ],
)
def test_training_config_rejects_invalid_values(kwargs : dict):
    """
    Test that TrainingConfig raises ValueError for invalid parameter values.

    Parameters
    ----------
    kwargs : dict
        The keyword arguments to pass to the TrainingConfig constructor.
    """
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)
        TrainingConfig.from_dict(kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"loss": "not_a_real_loss"},
        {"optimiser": "sdg"},  # as we currently only support adam
        {"learning_rate": -0.01},
        {"metrics": ["not_a_real_metric"]},
        {"steps_per_execution": 0},
        {"jit_compile": "not_a_boolean"},
    ],
)
def test_compile_config_rejects_invalid_values(kwargs : dict):
    """
    Test that CompileConfig raises ValueError for invalid parameter values.

    Parameters
    ----------
    kwargs : dict
        The keyword arguments to pass to the CompileConfig constructor.
    """
    with pytest.raises((ValueError, NotImplementedError)):
        CompileConfig(**kwargs)
        CompileConfig.from_dict(kwargs)

def test_compile_metrics_list_becomes_tuple():
    """
    Test that the metrics list in CompileConfig becomes a tuple.

    Parameters
    ----------
    kwargs : dict
        The keyword arguments to pass to the CompileConfig constructor.
    """
    cfg = CompileConfig.from_dict({"metrics": ["mean_squared_error"]})
    assert cfg.metrics == ("mean_squared_error",)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reduce_lr": "not_a_boolean"},
        # {"reduce_lr_monitor": "val_not_a_metric"},  # valid monitors depends on what is being tracked in compile
        {"reduce_lr_factor": 0},
        {"reduce_lr_factor": 1},
        {"reduce_lr_patience": -20},
        {"min_lr": -1e-6},
        {"early_stopping": "not_a_boolean"},
        # {"early_stopping_monitor": "val_not_a_metric"},  # valid monitors depends on what is being tracked in compile
        {"early_stopping_patience": -50},
    ],
)
def test_callback_config_rejects_invalid_values(kwargs : dict):
    """
    Test that CallbackConfig raises ValueError for invalid parameter values.

    Parameters
    ----------
    kwargs : dict
        The keyword arguments to pass to the CallbackConfig constructor.
    """
    with pytest.raises(ValueError):
        CallbackConfig(**kwargs)
        CallbackConfig.from_dict(kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "invalid_mode"},
        {"min_gpu_speedup": 1.0},  # Should be > 1
    ],
)
def test_device_config_rejects_invalid_values(kwargs : dict):
    """
    Test that DeviceConfig raises ValueError for invalid parameter values.

    Parameters
    ----------
    kwargs : dict
        The keyword arguments to pass to the DeviceConfig constructor.
    """
    with pytest.raises(ValueError):
        DeviceConfig(**kwargs)
        DeviceConfig.from_dict(kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"version": -0.2},
    ],
)
def test_output_config_rejects_invalid_values(kwargs : dict):
    """
    Test that OutputConfig raises ValueError for invalid parameter values.

    Parameters
    ----------
    kwargs : dict
        The keyword arguments to pass to the OutputConfig constructor.
    """
    with pytest.raises(ValueError):
        OutputConfig(**kwargs)
        OutputConfig.from_dict(kwargs)


def test_with_update_for_config_sections():
    """
    Test that the with_update method of any ConfigSection correctly updates the configuration.
    """
    cfg = ModelConfig.from_dict({})
    updated_cfg = cfg.with_update(num_layers=5)

    assert updated_cfg.num_layers == 5
    assert updated_cfg.num_neurons == cfg.num_neurons  # Unchanged


# ---------- Tests for SEAWRDConfig validation ----------
def test_default_config_is_valid():
    """
    Test that the default SEAWRDConfig is valid and has expected default values.
    """
    cfg = SEAWRDConfig.from_dict({})
    assert cfg.model.num_layers == 4
    assert cfg.training.validation_split == 0.2
    assert cfg.compile.metrics == ("mean_squared_error",)


def test_unknown_config_section_rejected():
    """
    Test that SEAWRDConfig raises ValueError when an unknown config section is provided.
    """
    with pytest.raises(ValueError, match="Unknown config sections"):
        SEAWRDConfig.from_dict({"nonsense": {}})


def test_callback_monitor_must_be_known_metric():
    """
    Test that SEAWRDConfig raises ValueError when an unknown callback monitor is provided.
    """
    with pytest.raises(ValueError, match="Unknown callback monitor"):
        SEAWRDConfig.from_dict({
            "compile": {"metrics": ["mean_squared_error"]},
            "callbacks": {"early_stopping_monitor": "accuracy"},
        })


def test_seawrd_with_update():
    """
    Test that the with_update method of SEAWRDConfig correctly updates the configuration.
    """
    cfg = SEAWRDConfig.from_dict({})
    updated_cfg = cfg.with_update(model={"num_layers": 5}, training={"batch_size": 64})

    assert updated_cfg.model.num_layers == 5
    assert updated_cfg.training.batch_size == 64
    assert updated_cfg.compile.metrics == cfg.compile.metrics  # Unchanged


def test_seawrd_with_update_section():
    """
    Test that the with_update_section method of SEAWRDConfig correctly updates a specific section of the configuration.
    """
    cfg = SEAWRDConfig.from_dict({})
    updated_cfg = cfg.with_update_section(section_name="model", **{"num_layers": 5})

    assert updated_cfg.model.num_layers == 5
    assert updated_cfg.training.batch_size == cfg.training.batch_size  # Unchanged
