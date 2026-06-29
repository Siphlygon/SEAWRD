from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import Any

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np

from config import ModelConfig, CompileConfig


class DNNManager:
    """
    A class to manage the creation, saving, and loading of a Deep Neural Network (DNN) model using Keras.
    """
    def __init__(self,
                 model : keras.Sequential,
                 history : dict | None,
                 version : int,
                 model_name : str):
        """
        Initialises the DNN class. Depending on the parameters, it either loads an existing model or generates a new one
        and saves it. The model is then compiled and ready for training or evaluation.

        Parameters
        ----------
        model : keras.Sequential
            The Keras Sequential model to be managed.
        history : dict | None
            The training history of the model, if available. If None, it indicates that the model has not been trained yet.
        version : int
            The version number of the model. This is used for saving and loading different versions of the model.
        model_name : str
            The name of the model. This is used for saving and loading the model files.
        """
        self.model = model
        self.history = history
        self.version = version
        self.model_name = model_name


    @classmethod
    def from_config(cls,
                    model_config: ModelConfig,
                    input_shape: tuple[int, ...],
                    normaliser: keras.layers.Normalization | None = None,) -> "DNNManager":
        """
        Generates a Keras Sequential model based on the provided configuration and creates a DNNManager instance with
        the generated model, an empty history, and version 0.

        Parameters
        ----------
        model_config : ModelConfig
            An instance of the ModelConfig dataclass containing model parameters such as number of layers, number of
            neurons, input shape, number of outputs, etc.
        input_shape : tuple[int, ...]
            The shape of the input features.
        normaliser : keras.layers.Normalization | None, optional
            The normalisation layer to be applied to the input features, by default None

        Returns
        -------
        DNNManager
            The DNNManager instance with the generated model, an empty history, and version 0.

        Raises
        ------
        ValueError
            If normalisation is to be used but the normaliser is not provided.
        """
        # Check if normalisation is to be used and if the normaliser is provided
        if model_config.use_normalisation:
            if normaliser is None:
                raise ValueError("normaliser must be provided when model.use_normalisation=true")
            normalisation_layer = normaliser
        else:
            normalisation_layer = None

        model = keras.Sequential()

        # Add the input layer with the specified input shape
        model.add(keras.Input(shape=input_shape))

        if normalisation_layer is not None:
            model.add(normalisation_layer)

        for _ in range(int(model_config.num_layers)):
            model.add(keras.layers.Dense(
                    int(model_config.num_neurons),
                    activation=model_config.activation,
                )
            )

        # Add the output layer with the specified number of outputs
        model.add(keras.layers.Dense(int(model_config.num_outputs)))

        if model_config.model_name:
            model_name = model_config.model_name
        else:
            temp_manager = cls(model=model, history=None, version=0, model_name="")
            model_name = temp_manager.create_model_name()

        return cls(
            model=model,
            history=None,
            version=0,
            model_name=model_name,
        )

    @classmethod
    def from_previous_model(cls,
                           model_dir : Path | str,
                           model_name : str,
                           version : int | None = None) -> DNNManager:
        """
        Loads a previously saved Keras model from the specified path and creates a DNNManager instance with the loaded
        model, history, and version.

        Parameters
        ----------
        model_dir : Path | str
            The directory containing the model files.
        model_name : str
            The name of the model.
        version : int | None, optional
            The version number of the model to load, by default None (loads the latest version).

        Returns
        -------
        DNNManager
            The DNNManager instance with the loaded model, history, and version.
        """
        temp_manager = cls(model=keras.Sequential(), history=None, version=0, model_name=model_name)
        model, history, version = temp_manager.get_model_version(model_dir, model_name, version)
        return cls(model=model, history=history, version=version, model_name=model_name)


    # ---------- UTILITY FUNCTIONS ----------
    def _copy_normalisation_weights(self,
                                    source_model : keras.Sequential,
                                    target_model : keras.Sequential):
        """
        Copies the weights of normalisation layers from the source model to the target model. This is useful when
        cloning a model to ensure that the normalisation layers in the new model have the same weights as those in the
        original model.

        Parameters
        ----------
        source_model : keras.Sequential
            The original Keras Sequential model from which to copy the normalisation weights.
        target_model : keras.Sequential
            The target Keras Sequential model to which the normalisation weights will be copied.
        """
        for source_layer, target_layer in zip(source_model.layers, target_model.layers):
            if isinstance(source_layer, keras.layers.Normalization):
                target_layer.set_weights(source_layer.get_weights())

    def clone_model(self) -> keras.Sequential:
        """
        Creates a clone of the current model with the same architecture and normalisation weights. This is useful for
        creating a new instance of the model for further training or evaluation without affecting the original model.

        Returns
        -------
        keras.Sequential
            A new Keras Sequential model that is a clone of the current model.
        """
        # Clone the model and copy normalisation weights
        model = keras.models.clone_model(self.model)
        self._copy_normalisation_weights(self.model, model)

        return model

    @staticmethod
    def _compile_from_config(model : keras.Sequential, compile_config : CompileConfig) -> None:
        """
        Compiles the current model using the instance's CompileConfig. This allows for dynamic compilation of the model
        with different loss functions, optimisers, and metrics.

        Parameters
        ----------
        model : keras.Sequential
            The Keras model to be compiled.
        """
        optimiser_name = compile_config.optimiser

        if optimiser_name == "adam":
            optimiser = keras.optimizers.Adam(
                learning_rate=compile_config.learning_rate
            )
        else:
            raise NotImplementedError(f"Unsupported optimiser: {optimiser_name}")

        model.compile(
            loss=compile_config.loss,
            optimizer=optimiser,
            metrics=list(compile_config.metrics),
            steps_per_execution=compile_config.steps_per_execution, # type: ignore
            jit_compile=compile_config.jit_compile  # type: ignore
        )


    # ---------- MODEL INFORMATION ----------
    def get_model_info(self) -> dict[str, Any]:
        """
        Retrieves information about the model, including its name, number of parameters, and training history.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the model's name, number of parameters, and training history.
        """
        return {
            "name": self.model.name,
            "num_param": self.model.count_params(),
            "history": self.history
        }

    def create_model_name(self) -> str:
        """
        Creates a model name based on the architecture of the model.

        Returns
        -------
        str
            The generated model name in the format "R(num_layers x num_neurons)_{num_inputs}i_{num_outputs}o".
        """
        num_layers = len(self.model.layers) - 2  # Exclude input and output layers
        num_neurons = self.model.layers[1].units if num_layers > 0 else 0
        num_inputs = self.model.input_shape[-1] if self.model.input_shape else 0
        num_outputs = self.model.output_shape[-1] if self.model.output_shape else 0
        return f"R({num_layers}x{num_neurons}_{num_inputs}i_{num_outputs}o)"


    # ---------- MODEL SAVING AND LOADING ----------
    def _get_model_paths(self,
                        model_dir : Path | str,
                        model_name : str,
                        version : int) -> dict[str, Path]:
        """
        Get the paths for the model, history, and plots for a specific version.

        Parameters
        ----------
        model_dir : Path | str
            The directory containing the model files.
        model_name : str
            The name of the model.
        version : int
            The version number of the model.

        Returns
        -------
        dict[str, Path]
            A dictionary containing the paths for the model, history, and plots.
        """
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)

        base = model_dir / f"{model_name}_v{version}"

        return {
            "model": base.with_name(base.name + "_model.keras"),
            "history": base.with_name(base.name + "_history.pkl"),
            "plots": base.with_name(base.name + "_plots.png"),
        }

    def get_latest_version(self,
                       model_dir : Path | str,
                       model_name : str,
                       pattern : re.Pattern | str | None = None) -> int:
        """
        Finds the latest version of a model from the specified directory.

        Parameters
        ----------
        model_dir : Path | str
            The directory containing the model files.
        model_name : str
            The name of the model.
        pattern : re.Pattern | str | None, optional
            The regex pattern to match model files, by default None

        Returns
        -------
        int
            The latest version number of the model, or 0 if none exists.
        """
        if pattern is None:
            pattern = re.compile(rf"^{re.escape(model_name)}_v(\d+)_model\.keras$")

        if isinstance(pattern, str):
            pattern = re.compile(pattern)

        if isinstance(model_dir, str):
            model_dir = Path(model_dir)

        # Find all model files matching the pattern and extract their version numbers
        versions = []
        for path in model_dir.glob(f"{model_name}_v*_model.keras"):
            match = pattern.match(path.name)
            if match:
                versions.append(int(match.group(1)))

        return max(versions) if versions else 0

    def save_model_version(self,
                           model : keras.Model,
                           history : keras.callbacks.History,
                           model_dir : Path | str,
                           model_name : str,
                           version : int) -> dict[str, Path]:
        """
        Save a specific version of the model and its training history.

        Parameters
        ----------
        model : keras.Model
            The model to save.
        history : keras.callbacks.History
            The training history to save.
        model_dir : Path | str
            The directory to save the model in.
        model_name : str
            The name of the model.
        version : int
            The version number of the model.

        Returns
        -------
        dict[str, Path]
            A dictionary containing the paths for the saved files.
        """
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)

        # Tries to create a folder to save the model, if it already exists, it will not raise an error
        model_dir.mkdir(parents=True, exist_ok=True)

        # Get the paths for the model and history files
        paths = self._get_model_paths(model_dir, model_name, version)

        # Save the model and history
        model.save(paths["model"])
        with open(paths["history"], "wb") as f:
            pickle.dump(history.history, f)

        print(f"Saved model version v{version}")
        print(f"Model:   {paths['model']}")
        print(f"History: {paths['history']}")

        return paths

    def get_model_version(self,
                           model_dir : Path | str,
                           model_name : str,
                           version : int | None = None) -> tuple[keras.Model, dict | None, int]:
        """
        Retrieves a specific version of the model and its training history.

        Parameters
        ----------
        model_dir : Path | str
            The directory containing the saved model.
        model_name : str
            The name of the model.
        version : int | None, optional
            The version number of the model to load. If None, loads the latest version, by default None

        Returns
        -------
        tuple[keras.Model, dict | None, int]
            A tuple containing the loaded model, its training history (if available), and the version number.

        Raises
        ------
        FileNotFoundError
            If no saved model is found for the specified model name.
        """
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)

        # If no version is specified, find the latest version
        if version is None:
            version = self.get_latest_version(model_dir, model_name)

        # If the specified version does not exist, raise an error
        if version == 0:
            raise FileNotFoundError(f"No saved model found for '{model_name}'.")

        paths = self._get_model_paths(model_dir, model_name, version)
        model = keras.models.load_model(paths["model"])

        history = None
        if paths["history"].exists():
            with open(paths["history"], "rb") as f:
                history = pickle.load(f)

        print(f"Loaded model version v{version}")

        return model, history, version

    def load_model_version(self,
                           model_dir : Path | str,
                           model_name : str,
                           version : int | None = None):
        """
        Loads a specific version of the model and its training history into the DNNManager.

        Parameters
        ----------
        model_dir : Path | str
            The directory containing the saved model.
        model_name : str
            The name of the model.
        version : int | None, optional
            The version number of the model to load. If None, loads the latest version, by default None
        """
        model, history, version = self.get_model_version(model_dir, model_name, version)
        self.model = model
        self.history = history
        self.version = version


if __name__ == "__main__":
    # Example training features (replace with actual data)
    train_features = np.asarray([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]], dtype=np.float32)

    # Initialise and adapt the normaliser with the training features
    normaliser = keras.layers.Normalization(axis=-1)
    normaliser.adapt(np.array(train_features))
    print("Normaliser adapted with training features.")

    # Decide the outputs
    label_cols = ["R_p"]

    # Create a dummy config (load it from TOML normally)
    config = {
        "num_layers": 4,
        "num_neurons": 8,
        "activation": "relu",
        "use_normalisation": True,
        "num_outputs": len(label_cols),
    }
    config = ModelConfig.from_dict(config)
    print("Model configuration created.")

    # Generate and compile the model
    dnn_manager = DNNManager.from_config(
        model_config=config,
        input_shape=train_features.shape[1:],
        normaliser=normaliser
    )
    dnn_model = dnn_manager.model
    print("Model generated from configuration.")
    dnn_model.summary()
