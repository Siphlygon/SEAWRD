from __future__ import annotations

from dataclasses import asdict, dataclass, replace, fields
from typing import Any, Literal, Mapping
from typing_extensions import Self

import keras


def validate_keras_field(field_name: str, field_type: str):
    """
    Validate that Keras can resolve a given field identifier.

    Parameters
    ----------
    field_name : str
        The name of the field to validate (e.g., 'loss', 'metric', 'activation').
    field_type : str
        The name of the Keras module to use for validation (e.g., losses, metrics, activations).
    
    Raises
    ------    
    ValueError
        If the field identifier or type is unknown or unsupported.
    """
    match field_type:
        case "losses":
            keras_module = keras.losses
        case "metrics":
            keras_module = keras.metrics
        case "activations":
            keras_module = keras.activations
        case _:
            raise ValueError(f"Unknown or unsupported Keras field type: {field_type!r}")

    try:
        keras_module.get(field_name)
    except Exception as exc:
        raise ValueError(f"Unknown or unsupported Keras {field_type}: {field_name!r}") from exc


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
        validate_keras_field(self.activation, "activations")


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
    metrics: tuple[str, ...] = ("mean_squared_error",)
    steps_per_execution: int = 1
    jit_compile: str | bool = "auto"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None = None) -> "CompileConfig":
        """
        Create a CompileConfig instance from a dictionary representation.
        
        Overrides the base from_dict method to ensure that the 'metrics' field is converted to a tuple if it is provided
        as a list.

        Parameters
        ----------
        data : Mapping[str, Any] | None, optional
            The dictionary representation of the configuration, by default None

        Returns
        -------
        CompileConfig
            The created CompileConfig instance
        """
        data = dict(data or {})

        if "metrics" in data:
            data["metrics"] = tuple(data["metrics"])

        return super().from_dict(data)

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("compile.learning_rate must be > 0")
        if self.optimiser != "adam":
            raise NotImplementedError("Only optimiser='adam' is currently supported.")

        # Check if steps_per_execution is either "auto" or a positive integer
        if self.steps_per_execution <= 0:
            raise ValueError("compile.steps_per_execution must be a positive integer")

        # Check if jit_compile is either "auto" or a boolean
        if not (self.jit_compile == "auto" or isinstance(self.jit_compile, bool)):
            raise ValueError("compile.jit_compile must be 'auto' or a boolean value")

        # Check if the loss function exists in keras.losses
        validate_keras_field(self.loss, "losses")

        # Convert list from TOML into tuple, since the dataclass is frozen.
        if isinstance(self.metrics, list):
            object.__setattr__(self, "metrics", tuple(self.metrics))

        # Check if all metrics exist in keras.metrics
        for metric in self.metrics:
            validate_keras_field(metric, "metrics")


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

    def __post_init__(self):
        # Check boolean fields for reduce_lr and early_stopping
        if not isinstance(self.reduce_lr, bool):
            raise ValueError("callbacks.reduce_lr must be a boolean value")
        if not isinstance(self.early_stopping, bool):
            raise ValueError("callbacks.early_stopping must be a boolean value")
        
        # Only validate these parameters if the corresponding callbacks are enabled
        # note - not checking for correct metrics here is intentional, see SEAWRDConfig.__post_init__ for why
        if self.reduce_lr:
            if not 0 < self.reduce_lr_factor < 1:
                raise ValueError("callbacks.reduce_lr_factor must be between 0 and 1")

            if self.reduce_lr_patience < 0:
                raise ValueError("callbacks.reduce_lr_patience must be >= 0")
            
            if self.min_lr < 0:
                raise ValueError("callbacks.min_lr must be >= 0")

        if self.early_stopping:
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
    
    @staticmethod
    def _validate_monitor_name(monitor: str,
                               metrics: tuple[str, ...],
                               has_validation: bool = True):
        """
        Validate that a monitor name is valid for callbacks based on the provided metrics.
        
        Note that this requires information from CompileConfig to determine which metrics are being tracked, and whether
        validation metrics are included, and so is intended to only be ran for SEAWRDConfig instances.

        Parameters
        ----------
        monitor : str
            The monitor name to validate.
        metrics : tuple[str, ...]
            The identifiers of the metrics being tracked by the compiled model.
        has_validation : bool, optional
            Whether to include validation monitor names, by default True

        Raises
        ------
        ValueError
            The error message indicating the unknown monitor name.
        """
        aliases = {
            "mse": "mean_squared_error",
            "mae": "mean_absolute_error",
            "mape": "mean_absolute_percentage_error",
            "msle": "mean_squared_logarithmic_error",
            "acc": "accuracy",
        }
        allowed = {"loss"}

        if has_validation:
            allowed.add("val_loss")

        for metric in metrics:
            metric_name = aliases.get(metric, metric)

            allowed.add(metric_name)

            if has_validation:
                allowed.add(f"val_{metric_name}")

        if monitor not in allowed:
            raise ValueError(
                f"Unknown callback monitor {monitor!r}. "
                f"Expected one of: {sorted(allowed)}"
            )

    def __post_init__(self):
        # Validate that the metrics specified in the callbacks section are also present in the compile section
        self._validate_monitor_name(self.callbacks.reduce_lr_monitor, self.compile.metrics)
        self._validate_monitor_name(self.callbacks.early_stopping_monitor, self.compile.metrics)


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
