from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import keras

try:
    import tomllib  # Python 3.11+, tomlib is part of the standard library
except ModuleNotFoundError:
    import tomli as tomllib  # Python <=3.10 fallback, will require the `tomli` package to be installed


@dataclass(frozen=True)
class ModelConfig:
    """
    Configuration for the model architecture.
    """
    model_name: str = ""
    num_layers: int = 4
    num_neurons: int = 8
    num_outputs: int = 1
    activation: str = "relu"
    use_normalisation: bool = True

    def __post_init__(self):
        if self.num_layers < 0:
            raise ValueError("model.num_layers must be >= 0")
        if self.num_neurons <= 0:
            raise ValueError("model.num_neurons must be > 0")
        if self.num_outputs <= 0:
            raise ValueError("model.num_outputs must be > 0")

        # Check if the activation function exists in keras.activations
        if not hasattr(keras.activations, self.activation):
            raise ValueError(f"model.activation '{self.activation}' is not a valid Keras activation function.")


@dataclass(frozen=True)
class TrainingConfig:
    """
    Configuration for the training process.
    """
    num_epochs: int = 200
    batch_size: int = 1024
    validation_split: float = 0.2
    num_models: int = 5
    shuffle: bool = True

    def __post_init__(self):
        if self.num_epochs <= 0:
            raise ValueError("training.num_epochs must be > 0")
        if self.batch_size <= 0:
            raise ValueError("training.batch_size must be > 0")
        if not 0 < self.validation_split < 1:
            raise ValueError("training.validation_split must be between 0 and 1")
        if self.num_models <= 0:
            raise ValueError("training.num_models must be > 0")


@dataclass(frozen=True)
class CompileConfig:
    """
    Configuration for compiling the Keras model.
    """
    loss: str = "mean_squared_error"
    optimiser: Literal["adam"] = "adam"
    learning_rate: float = 0.005
    _metrics: tuple[str, ...] = ("mean_squared_error",)
    steps_per_execution: int | str = "auto"
    jit_compile: bool = False

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("compile.learning_rate must be > 0")
        if self.optimiser != "adam":
            raise NotImplementedError("Only optimiser='adam' is currently supported.")

        # Check if steps_per_execution is either "auto" or a positive integer
        if not (self.steps_per_execution == "auto" or (isinstance(self.steps_per_execution, int) and self.steps_per_execution > 0)):
            raise ValueError("compile.steps_per_execution must be 'auto' or a positive integer")

        # Check if the loss function exists in keras.losses
        if not hasattr(keras.losses, self.loss):
            raise ValueError(f"compile.loss '{self.loss}' is not a valid Keras loss function.")

        # Convert list from TOML into tuple, since the dataclass is frozen.
        if isinstance(self._metrics, list):
            object.__setattr__(self, "metrics", tuple(self._metrics))

        # Check if all metrics exist in keras.metrics
        for metric in self._metrics:
            if not hasattr(keras.metrics, metric):
                raise ValueError(f"compile.metrics '{metric}' is not a valid Keras metric function.")

        # # Check if the optimizer exists in keras.optimizers
        # if not hasattr(keras.optimizers, self.optimiser.capitalize()):
        #     raise ValueError(f"compile.optimiser '{self.optimiser}' is not a valid Keras optimizer.")

    @property
    def metrics(self) -> list[str]:
        """
        Return the metrics as a list. This is useful for Keras model compilation, which expects a list of metrics.

        Returns
        -------
        list[str]
            The metrics as a list.
        """
        return list(self._metrics)


@dataclass(frozen=True)
class CallbackConfig:
    """
    Configuration for Keras callbacks during training.
    """
    reduce_lr: bool = True
    reduce_lr_monitor: str = "val_loss"
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 20
    min_lr: float = 1e-6

    early_stopping: bool = True
    early_stopping_monitor: str = "val_loss"
    early_stopping_patience: int = 50
    restore_best_weights: bool = True

    def __post_init__(self):
        # Only validate these parameters if the corresponding callbacks are enabled
        if self.reduce_lr:
            # Check if the monitor metric exists in keras.metrics
            if not hasattr(keras.metrics, self.reduce_lr_monitor):
                raise ValueError(f"callbacks.reduce_lr_monitor '{self.reduce_lr_monitor}' is not a valid Keras metric function.")

            if not 0 < self.reduce_lr_factor < 1:
                raise ValueError("callbacks.reduce_lr_factor must be between 0 and 1")

            if self.reduce_lr_patience < 0:
                raise ValueError("callbacks.reduce_lr_patience must be >= 0")

        if self.early_stopping:
            # Check if the monitor metric exists in keras.metrics
            if not hasattr(keras.metrics, self.early_stopping_monitor):
                raise ValueError(f"callbacks.early_stopping_monitor '{self.early_stopping_monitor}' is not a valid Keras metric function.")

            if self.early_stopping_patience < 0:
                raise ValueError("callbacks.early_stopping_patience must be >= 0")


@dataclass(frozen=True)
class DeviceConfig:
    """
    Configuration for the device on which to run the training (CPU or GPU).
    """
    mode: Literal["auto", "cpu", "gpu"] = "auto"
    benchmark_device: bool = False
    min_gpu_speedup: float = 1.2

    def __post_init__(self):
        if self.mode not in {"auto", "cpu", "gpu"}:
            raise ValueError("device.mode must be 'auto', 'cpu', or 'gpu'")
        if self.min_gpu_speedup <= 1:
            raise ValueError("device.min_gpu_speedup should be > 1")


@dataclass(frozen=True)
class OutputConfig:
    """
    Configuration for output settings, including model saving and plot generation.
    """
    model_dir: str = "models/"
    version: int = 1
    save_model: bool = True
    save_plots: bool = True

    def __post_init__(self):
        if self.version < 0:
            raise ValueError("output.version must be >= 0")


@dataclass(frozen=True)
class SEAWRDConfig:
    """
    Main configuration class for the SEAWRD framework. This class aggregates all the individual configuration sections
    into a single, unified configuration object.
    """
    model: ModelConfig
    training: TrainingConfig
    compile: CompileConfig
    callbacks: CallbackConfig
    device: DeviceConfig
    output: OutputConfig

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the SEAWRDConfig dataclass into a dictionary representation.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the SEAWRDConfig dataclass, including all nested configurations.
        """
        return asdict(self)


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
        return SEAWRDConfig(
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            compile=CompileConfig(**raw.get("compile", {})),
            callbacks=CallbackConfig(**raw.get("callbacks", {})),
            device=DeviceConfig(**raw.get("device", {})),
            output=OutputConfig(**raw.get("output", {})),
        )


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
