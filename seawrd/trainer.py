import keras
import tensorflow_docs
from seawrd.model import DNNManager
import numpy as np
import pandas as pd


class DNNTrainer:
    """
    A class to manage the training of a deep neural network (DNN) model. It can either load an existing model or
    generate a new one based on the provided parameters. The class also provides methods to generate callbacks for
    training and to print the architecture performance of the model.
    """

    def __init__(self,
                 load_existing: bool = False,
                 model_dir: str = "models/",
                 model_name: str = "dnn_model",
                 version: int = 0,
                 num_epochs: int = 1000,
                 batch_size: int = 256,
                 validation_split: float = 0.2,
                 learning_rate: float = 5e-3,
                 **kwargs):
        """
        Initialise the DNNTrainer class. This class is responsible for managing the training of a deep neural network
        (DNN) model. It can either load an existing model or generate a new one based on the provided parameters.

        Parameters
        ----------
        load_existing : bool, optional
            Whether to load an existing model, by default False
        model_dir : str, optional
            The directory where the model is saved or will be saved, by default "models/"
        model_name : str, optional
            The name of the model, by default "dnn_model"
        version : int, optional
            The version of the model, by default 0
        num_epochs : int, optional
            The number of epochs to train the model, by default 1000
        batch_size : int, optional
            The number of samples per gradient update, by default 256
        validation_split : float, optional
            The fraction of the training data to be used as validation data, by default 0.2
        learning_rate : float, optional
            The learning rate for the optimizer, by default 5e-3
        **kwargs
            Additional keyword arguments to be passed to the DNNManager for model creation or loading.
        """

        if not load_existing:
            self.model_manager = DNNManager.from_new_model(**kwargs)
        else:
            self.model_manager = DNNManager.from_previous_model(**kwargs)

        # Model parameters
        self.model = self.model_manager.model
        self.model_dir = model_dir
        self.model_name = model_name
        self.version = version

        # Training hyperparameters
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def generate_callbacks(self,
                           monitor : str = 'val_loss',
                           lr_factor : float = 0.5,
                           lr_patience : int = 20,
                           min_lr : float = 1e-6,
                           early_stopping_patience : int = 50,
                           verbose : int = 1) -> list[keras.callbacks.Callback]:
        """
        Generate a list of callbacks for training a Keras model. These callbacks help in monitoring and controlling the
        training process.

        Parameters
        ----------
        monitor : str, optional
            The metric to monitor for improvement, by default 'val_loss'.
        lr_factor : float, optional
            The factor by which the learning rate will be reduced, by default 0.5.
        lr_patience : int, optional
            The number of epochs with no improvement after which learning rate will be reduced, by default 20
        min_lr : float, optional
            The lower bound on the learning rate, by default 1e-6.
        early_stopping_patience : int, optional
            The number of epochs with no improvement after which training will be stopped, by default 50.
        verbose : int, optional
            The verbosity mode, by default 1, meaning that messages will be printed when reducing learning rate or
            stopping early.

        Returns
        -------
        list[keras.callbacks.Callback]
            A list of Keras callbacks to be used during model training.
        """
        callbacks = [
            tensorflow_docs.modeling.EpochDots(),
            keras.callbacks.ReduceLROnPlateau( # reduce Learning Rate if the model does not improve
                monitor=monitor, # the metric to monitor for improvement
                factor=lr_factor, # the factor by which the learning rate will be reduced. new_lr = lr * factor
                patience=lr_patience, # number of epochs with no improvement after which learning rate will be reduced
                min_lr=min_lr, # lower bound on the learning rate
                verbose=verbose # verbosity mode, 1 = print messages when reducing learning rate
            ),
            keras.callbacks.EarlyStopping( # stop the training if the model does not improve
                monitor=monitor,
                patience=early_stopping_patience,
                restore_best_weights=True, # restore model weights from the epoch with the best value of the monitored quantity
                verbose=verbose
            )
        ]
        return callbacks


    def start_training(self,
                       num_models: int,
                       input_features: pd.DataFrame,
                       input_labels: pd.Series):
        # Add callbacks for learning rate adjustment and early stopping
        my_callbacks = self.generate_callbacks()



if __name__ == "__main__":
    # because our DNN are small in size, the final result depends on our initial (random) state. we train the same
    # multiple with N_models different initial conditions, and keep the best + some statistics.
    overwrite_existing = True
    current_version = 0

    # default values -- could be loaded from config?
    num_layers = 4
    num_neurons = 8
    num_epochs = 1000
    dnn_trainer = DNNTrainer(load_existing=False,
                             num_layers=num_layers,
                             num_neurons=num_neurons,
                             num_epochs=num_epochs)
    dnn_manager = dnn_trainer.model_manager
    dnn_model = dnn_manager.model
