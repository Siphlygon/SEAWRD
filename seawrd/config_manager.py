from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from typing import Any

try:
    import tomllib  # Python 3.11+, tomlib is part of the standard library
except ModuleNotFoundError:
    import tomli as tomllib  # Python <=3.10 fallback, will require the `tomli` package to be installed

from .config import SEAWRDConfig


class ConfigManager:
    """
    A class to manage the loading and validation of the SEAWRD configuration from a TOML file.
    """
    def __init__(self, config : SEAWRDConfig, config_path: str | Path | None = None) -> None:
        """
        Initialise the ConfigManager with a path to the configuration file and a pre-loaded configuration.

        Parameters
        ----------
        config : SEAWRDConfig
            An instance of the SEAWRDConfig dataclass containing the configuration settings.
        config_path : str | Path | None
            The path to the TOML configuration file.
        """
        self.config = config
        self.config_path = Path(config_path) if config_path is not None else None


    @classmethod
    def from_toml(cls, path: str | Path) -> ConfigManager:
        """
        Create a ConfigManager instance from a TOML configuration file.

        Parameters
        ----------
        path : str | Path
            The path to the TOML configuration file.

        Returns
        -------
        ConfigManager
            An instance of ConfigManager initialized with the loaded configuration.
        """
        path = Path(path)

        with path.open("rb") as f:
            raw = tomllib.load(f)

        config = cls._from_dict(raw)
        return cls(config=config, config_path=path)


    @staticmethod
    def _from_dict(raw: dict[str, Any]) -> SEAWRDConfig:
        """
        Create a SEAWRDConfig instance from a dictionary representation.

        Parameters
        ----------
        raw : dict[str, Any]
            The raw dictionary representation of the configuration.

        Returns
        -------
        SEAWRDConfig
            An instance of the SEAWRDConfig dataclass initialized with the provided configuration.
        """
        return SEAWRDConfig.from_dict(raw)


    def to_dict(self) -> dict[str, Any]:
        """
        Convert the current configuration to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            The dictionary representation of the configuration.
        """
        return self.config.to_dict()


    def with_override(self, dotted_key: str, value: Any) -> "ConfigManager":
        """
        Return a new ConfigManager with an updated value for a specific configuration field.

        Example:
            cfg2 = cfg.with_override("training.batch_size", 2048)
            cfg3 = cfg.with_override("model.num_neurons", 64)

        Parameters
        ----------
        dotted_key : str
            The dotted key representing the configuration field to update. In format "section.field", e.g.,
            "training.batch_size".
        value : Any
            The new value for the configuration field.

        Returns
        -------
        ConfigManager
            A new ConfigManager instance with the updated value.

        Raises
        ------
        KeyError
            If the specified section or field is not found in the configuration.
        """
        section_name, field_name = dotted_key.split(".", maxsplit=1)

        if not hasattr(self.config, section_name):
            raise KeyError(f"Unknown config section: {section_name}")

        old_section = getattr(self.config, section_name)

        if not hasattr(old_section, field_name):
            raise KeyError(f"Unknown config field: {dotted_key}")

        new_section = replace(old_section, **{field_name: value})
        new_config = replace(self.config, **{section_name: new_section})

        return ConfigManager(config=new_config, config_path=self.config_path)


    def with_overrides(self, overrides: dict[str, Any]) -> "ConfigManager":
        """
        Return a new ConfigManager with multiple updated values.

        Parameters
        ----------
        overrides : dict[str, Any]
            A dictionary of dotted keys and their corresponding values to override.

        Returns
        -------
        ConfigManager
            A new ConfigManager instance with the updated values.
        """
        manager = self

        for dotted_key, value in overrides.items():
            manager = manager.with_override(dotted_key, value)

        return manager
