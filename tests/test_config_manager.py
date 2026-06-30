"""
Unit tests for the ConfigManager class, which is responsible for loading and validating the SEAWRD configuration from a
TOML file. These tests ensure that the ConfigManager correctly loads configurations, applies overrides, and handles
errors appropriately.
"""

from pathlib import Path

import pytest

from seawrd.config import SEAWRDConfig
from seawrd.config_manager import ConfigManager


def test_load_minimal_toml(tmp_path : Path):
    """
    Test that a minimal TOML configuration file can be loaded into a ConfigManager.
    """
    path = tmp_path / "config.toml"
    path.write_text("""
        [training]
        batch_size = 16

        [model]
        num_layers = 2
        """
    )

    manager = ConfigManager.from_toml(path)

    assert manager.config.training.batch_size == 16
    assert manager.config.model.num_layers == 2


def test_with_override_returns_new_manager():
    """
    Test that with_override returns a new ConfigManager with the updated value, while the original ConfigManager remains
    unchanged.
    """
    manager = ConfigManager(SEAWRDConfig.from_dict({}))
    updated = manager.with_override("training.batch_size", 32)

    assert manager.config.training.batch_size == 1024
    assert updated.config.training.batch_size == 32


def test_with_override_rejects_unknown_section():
    """
    Test that with_override raises KeyError when an unknown configuration section is provided.
    """
    manager = ConfigManager(SEAWRDConfig.from_dict({}))

    with pytest.raises(KeyError, match="Unknown config section"):
        manager.with_override("bad.batch_size", 32)


def test_with_override_rejects_unknown_field():
    """
    Test that with_override raises KeyError when an unknown configuration field is provided.
    """
    manager = ConfigManager(SEAWRDConfig.from_dict({}))

    with pytest.raises(KeyError, match="Unknown config field"):
        manager.with_override("training.not_a_field", 32)


def test_with_override_rejects_invalid_value():
    """
    Test that with_override raises ValueError when an invalid value is provided for a configuration field.
    """
    manager = ConfigManager(SEAWRDConfig.from_dict({}))

    with pytest.raises(ValueError):
        manager.with_override("training.batch_size", -1)


def test_with_overrides_applies_multiple_changes():
    """
    Test that with_overrides applies multiple changes to the configuration and returns a new ConfigManager.
    """
    manager = ConfigManager(SEAWRDConfig.from_dict({}))
    overrides = {
        "training.batch_size": 64,
        "model.num_layers": 3,
    }
    updated = manager.with_overrides(overrides)

    assert manager.config.training.batch_size == 1024
    assert manager.config.model.num_layers == 4
    assert updated.config.training.batch_size == 64
    assert updated.config.model.num_layers == 3


def test_config_manager_rejects_unknown_toml_section(tmp_path: Path):
    """
    Test that ConfigManager raises ValueError when an unknown section is present in the TOML configuration file.
    
    Parameters
    ----------
    tmp_path : pathlib.Path
        The path to the temporary directory where the TOML file will be created.
    """
    path = tmp_path / "config.toml"
    path.write_text("""
        [not_a_real_section]
        thing = 1
        """
    )
    with pytest.raises(ValueError, match="Unknown config sections"):
        ConfigManager.from_toml(path)
