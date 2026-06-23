import keras
import tensorflow as tf
import numpy as np
from pathlib import Path
import re
import pickle


class DNN:
    """
    A class to manage the creation, saving, and loading of a Deep Neural Network (DNN) model using Keras.
    """

    def __init__(self,
                 num_layers : int,
                 num_neurons : int,
                 num_epochs : int):
        """
        Initialises the DNN class with the specified number of layers, neurons, and epochs.

        Parameters
        ----------
        num_layers : int
            The number of hidden layers in the neural network.
        num_neurons : int
            The number of neurons in each hidden layer.
        num_epochs : int
            The number of epochs to train the model.
        """
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.num_epochs = num_epochs

    # ---------- MODEL ARCHITECTURE ----------
    def generate_model(self,
                       train_features : np.ndarray,
                       normalizer : keras.layers.Normalization,
                       labels_cols : list) -> keras.Sequential:
        """
        Generates a Keras Sequential model, using an input layer, normalisation, num_layers hidden dense layers of
        num_neurons each, and an output layer specified by labels_cols.

        Parameters
        ----------
        train_features : np.ndarray
            The training features.
        normalizer : keras.layers.Normalization
            The normalisation layer to be applied to the input features.
        labels_cols : list
            The list of column names for the labels.

        Returns
        -------
        keras.Sequential
            The constructed Keras Sequential model.
        """
        # Creates a new model
        dnn_model = keras.Sequential()

        # Add an input layer for features
        dnn_model.add(keras.Input(shape=(train_features.shape[1],)))
        dnn_model.add(normalizer)

        # Add hidden layers
        for _ in range(self.num_layers):
            dnn_model.add(keras.layers.Dense(self.num_neurons, activation='relu'))

        # Add output layer
        dnn_model.add(keras.layers.Dense(len(labels_cols)))

        return dnn_model


    # ---------- MODEL SAVING AND LOADING ----------
    def latest_version(self,
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

    def get_model_paths(self,
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
        paths = self.get_model_paths(model_dir, model_name, version)

        # Save the model and history
        model.save(paths["model"])
        with open(paths["history"], "wb") as f:
            pickle.dump(history.history, f)

        print(f"Saved model version v{version}")
        print(f"Model:   {paths['model']}")
        print(f"History: {paths['history']}")

        return paths


    def load_model_version(self,
                           model_dir : Path | str,
                           model_name : str,
                           version : int | None = None) -> tuple[keras.Model, dict | None, int]:
        """
        Load a specific version of the model and its training history.

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
            version = self.latest_version(model_dir, model_name)

        # If the specified version does not exist, raise an error
        if version == 0:
            raise FileNotFoundError(f"No saved model found for '{model_name}'.")

        paths = self.get_model_paths(model_dir, model_name, version)
        model = keras.models.load_model(paths["model"])

        history = None
        if paths["history"].exists():
            with open(paths["history"], "rb") as f:
                history = pickle.load(f)

        print(f"Loaded model version v{version}")

        return model, history, version


if __name__ == "__main__":
    # Example usage of the DNN class
    dnn_manager = DNN(num_layers=2, num_neurons=8, num_epochs=1000)

    # Example training features (replace with actual data)
    train_features = np.asarray([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]])

    # Initialise and adapt the normaliser with the training features
    normaliser = keras.layers.Normalization(axis=-1)
    normaliser.adapt(np.array(train_features))

    label_cols = ["R_p"]

    dnn = dnn_manager.generate_model(train_features, normaliser, label_cols)
    dnn.fit(train_features, label_cols)
