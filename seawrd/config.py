from __future__ import annotations

from dataclasses import asdict, dataclass, replace, fields
from typing import Any, Literal, Mapping
from typing_extensions import Self

import keras


class ConfigSection:
    """
    Base helper for simple dataclass config sections, with potential for future extensions.
    """

    @classmethod
    def from_dict(cls: type[Self], data: Mapping[str, Any] | None = None) -> Self:
        """
        Load a dataclass instance from a dictionary, ensuring that only known fields are used for initialisation.

        This allows for creation of dataclass instances (i.e., ConfigSections) from dictionaries while validating that
        the provided keys match the expected fields of the dataclass.

        Parameters
        ----------
        cls : type[Self]
            The dataclass type to instantiate.
        data : Mapping[str, Any] | None, optional
            The dictionary containing the configuration data, by default None

        Returns
        -------
        Self
            An instance of the dataclass initialized with the provided configuration.

        Raises
        ------
        ValueError
            If there are unknown fields in the data dictionary that do not correspond to any fields in the dataclass.
        """
        if data is None:
            data = {}

        # Filter the provided data to only include known fields for the dataclass
        valid_fields = {field.name for field in fields(cls) if field.init}
        unknown_fields = set(data) - valid_fields

        if unknown_fields:
            raise ValueError(
                f"Unknown fields for {cls.__name__}: {sorted(unknown_fields)}"
            )

        return cls(**dict(data))

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the dataclass instance to a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the dataclass instance, including all fields and their values.
        """
        return asdict(self)

    def with_update(self: Self, **updates: Any) -> Self:
        """
        Create a new instance of the dataclass with updated fields.

        Parameters
        ----------
        self : Self
            The current instance of the dataclass.

        Returns
        -------
        Self
            A new instance of the dataclass with the specified fields updated.

        Raises
        ------
        ValueError
            If there are unknown fields in the updates that do not correspond to any fields in the dataclass.
        """
        valid_fields = {field.name for field in fields(self) if field.init}
        unknown_fields = set(updates) - valid_fields

        if unknown_fields:
            raise ValueError(
                f"Unknown fields for {type(self).__name__}: {sorted(unknown_fields)}"
            )

        return replace(self, **updates)


@dataclass(frozen=True)
class ModelConfig(ConfigSection):
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
class TrainingConfig(ConfigSection):
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
class CompileConfig(ConfigSection):
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
class CallbackConfig(ConfigSection):
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
class DeviceConfig(ConfigSection):
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
class OutputConfig(ConfigSection):
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


    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None = None) -> "SEAWRDConfig":
        """
        Create a SEAWRDConfig instance from a dictionary representation.

        Parameters
        ----------
        raw : Mapping[str, Any] | None, optional
            The dictionary representation of the configuration, by default None

        Returns
        -------
        SEAWRDConfig
            The initialized SEAWRDConfig instance

        Raises
        ------
        ValueError
            If there are unknown sections in the provided dictionary that do not correspond to any of the expected
            configuration sections.
        """
        raw = raw or {}

        valid_sections = {"model", "training", "compile", "callbacks", "device", "output"}
        unknown_sections = set(raw) - valid_sections

        if unknown_sections:
            raise ValueError(f"Unknown config sections: {sorted(unknown_sections)}")

        return cls(
            model=ModelConfig.from_dict(raw.get("model")),
            training=TrainingConfig.from_dict(raw.get("training")),
            compile=CompileConfig.from_dict(raw.get("compile")),
            callbacks=CallbackConfig.from_dict(raw.get("callbacks")),
            device=DeviceConfig.from_dict(raw.get("device")),
            output=OutputConfig.from_dict(raw.get("output")),
        )
