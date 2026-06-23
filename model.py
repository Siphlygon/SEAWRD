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
