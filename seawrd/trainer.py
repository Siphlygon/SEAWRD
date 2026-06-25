import keras
import numpy as np
import pandas as pd
import tensorflow_docs

from seawrd.model import DNNManager


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
            self.model_manager = DNNManager.from_previous_model(model_dir=model_dir,
                                                                model_name=model_name,
                                                                version=version,)

        # Model values
        self.model = self.model_manager.model
        self.model_dir = model_dir
        self.model_name = model_name
        self.version = version

        # Training hyperparameters
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Initialize the callbacks for training
        self.callbacks = self._generate_callbacks()

    def _generate_callbacks(self,
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

    def _evaluate_model(self,
                           test_features: pd.DataFrame,
                           test_labels: pd.Series):
        """
        Evaluate the model's performance on the provided test dataset. This method computes the predictions of the model
        on the test features and calculates the error by comparing the predictions with the actual test labels.

        Parameters
        ----------
        test_features : pd.DataFrame
            The features of the test dataset on which the model will be evaluated.
        test_labels : pd.Series
            The labels of the test dataset against which the model will be evaluated.

        Returns
        -------
        np.ndarray
            The error between the actual test labels and the model's predictions. This is calculated as the difference
            between the actual values and the predicted values.
        """
        predictions = self.model.predict(test_features)
        actual_values = test_labels.to_numpy()
        error = actual_values - predictions
        return error

    # ---------- MAIN TRAINING LOOP ----------
    def _train_single_model(self,
                           model : keras.Model,
                           seed : int,
                           callbacks : list[keras.callbacks.Callback],
                           input_features : pd.DataFrame,
                           input_labels : pd.Series) -> keras.callbacks.History:
        """
        Train the provided model using the specified random seed for reproducibility. The training process involves
        compiling the model, fitting it to the training data, and applying callbacks for learning rate adjustment and
        early stopping.

        Parameters
        ----------
        model : keras.Model
            The Keras model to be trained.
        seed : int
            The random seed for reproducibility.
        callbacks : list[keras.callbacks.Callback]
            A list of Keras callbacks to be applied during training, such as learning rate adjustment and early
            stopping.
        input_features : pd.DataFrame
            The features of the input dataset, which will be split into training and validation sets based on the
            validation_split parameter.
        input_labels : pd.Series
            The labels of the input dataset, which will be split into training and validation sets based on the
            validation_split parameter.

        Returns
        -------
        keras.callbacks.History
            The training history of the model.
        """
        # Set random seed for reproducibility and clear previous Keras session
        keras.utils.set_random_seed(seed)
        keras.backend.clear_session()

        # todo: consider customising metric?

        # Compile and fit the model
        model.compile(loss="mean_squared_error",
                      optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      metrics=["mean_squared_error"])

        history = model.fit(
            input_features,
            input_labels,
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=0,
            shuffle=True)

        return history

    def train_models(self,
                     input_features: pd.DataFrame,
                     input_labels: pd.Series,
                     num_models: int = 10):
        """
        Train multiple models with different random initializations and keep the best one based on validation loss. This
        method also collects statistics about each model's performance.

        Parameters
        ----------
        input_features : pd.DataFrame
            _description_
        input_labels : pd.Series
            _description_
        num_models : int, optional
            _description_, by default 10
        """
        # beginning of actual training
        best_model = None
        best_history = None
        best_val = np.inf
        rp_means = np.zeros(num_models)
        rp_stds = np.zeros(num_models)
        losses = np.zeros(num_models)
        val_losses = np.zeros(num_models)

        for seed in range(num_models):
            print(f"Training model {seed}/{num_models}:")
            history = self._train_single_model(model=self.model,
                                               seed=seed,
                                               callbacks=self.callbacks,
                                               input_features=input_features,
                                               input_labels=input_labels)

            # records the best
            val_min = min(history.history['val_loss'])
            loss_min = min(history.history['loss'])
            print(f"Final val_loss of model {seed}/{num_models}: {val_min}")
            if val_min < best_val:
                best_val = val_min
                best_model = self.model
                best_history = history

            # records statistics about each model
            rp_error = self._evaluate_model(test_features=None,
                                            test_labels=None)
            rp_means[seed] = np.mean(rp_error)
            rp_stds[seed] = np.std(rp_error)
            # list_num_epoch[seed] = float(len(history.history['loss']))
            val_losses[seed] = val_min
            losses[seed] = loss_min

        # once all N_models have been trained, saves the best model
        self.model_manager.save_model_version(best_model,
                                              best_history,
                                              self.model_dir,
                                              self.model_name,
                                              self.version)

        return rp_means, rp_stds, losses, val_losses


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


