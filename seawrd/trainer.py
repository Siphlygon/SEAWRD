import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import numpy as np
import pandas as pd
import tensorflow_docs.modeling
import matplotlib.pyplot as plt

from model import DNNManager


class DNNTrainer:
    """
    A class to manage the training of a deep neural network (DNN) model. It can either load an existing model or
    generate a new one based on the provided parameters. The class also provides methods to generate callbacks for
    training and to print the architecture performance of the model.
    """

    def __init__(self,
                 load_existing: bool = False,
                 model_dir: str = "models/",
                 model_name: str | None = None,
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
        assert not (load_existing and version != 0), "If load_existing is True, version must be provided."
        assert not (load_existing and model_name is not None), "If load_existing is True, model_name must be provided."

        if not load_existing:
            self.model_manager = DNNManager.from_new_model(**kwargs)
        else:
            self.model_manager = DNNManager.from_previous_model(model_dir=model_dir,
                                                                model_name=model_name,
                                                                version=version,)

        # Model values
        self.model = self.model_manager.model
        self.model_dir = model_dir
        self.model_name = self.model_manager.model_name if model_name is None else model_name
        self.version = version

        # Training hyperparameters
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Initialize the callbacks for training
        self.callbacks = self._generate_callbacks()

        # Initialise variables to keep track of the best model and its performance
        self.best_model = None
        self.best_history = None
        self.best_val = np.inf

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
            The number of epochs with no improvement after which learning rate will be reduced, by default 15
        min_lr : float, optional
            The lower bound on the learning rate, by default 1e-7.
        early_stopping_patience : int, optional
            The number of epochs with no improvement after which training will be stopped, by default 30.
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

    def _round_to_n_sig_figs(self, x: float, n: int) -> float:
        """
        Round a number to a specified number of significant figures.

        Parameters
        ----------
        x : float
            The number to be rounded.
        n : int
            The number of significant figures to round to.

        Returns
        -------
        float
            The number rounded to the specified number of significant figures.
        """
        if x == 0:
            return 0.0
        else:
            return round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))

    def print_architecture_performance(self):
        """
        Print the performance metrics of the model's architecture, including the model name, number of parameters,
        and statistics about the training and validation losses. This method provides a quick overview of the model's
        performance after training, allowing for easy comparison between different model architectures or training runs.
        """
        assert hasattr(self, 'losses'), "The model has not been trained yet. Please train the model before printing performance metrics."

        print(f"name:           {self.model.name}")
        print(f"num_param:      {self.model.count_params()}")
        print(f"num_epoch_mean: {self._round_to_n_sig_figs(np.mean(self.list_num_epoch), 5)}")
        print(f"loss_min:       {self._round_to_n_sig_figs(np.min(self.losses), 5)}")
        print(f"loss_max:       {self._round_to_n_sig_figs(np.max(self.losses), 5)}")
        print(f"loss_mean:      {self._round_to_n_sig_figs(np.mean(self.losses), 5)}")
        print(f"loss_stdev:     {self._round_to_n_sig_figs(np.std(self.losses), 5)}")
        print(f"val_loss_min:   {self._round_to_n_sig_figs(np.min(self.val_losses), 5)}")
        print(f"val_loss_max:   {self._round_to_n_sig_figs(np.max(self.val_losses), 5)}")
        print(f"val_loss_mean:  {self._round_to_n_sig_figs(np.mean(self.val_losses), 5)}")
        print(f"val_loss_stdev: {self._round_to_n_sig_figs(np.std(self.val_losses), 5)}")

    def plot_loss_curve(self,
                        log_y : bool = True,
                        log_x : bool = True,
                        max_y : float = None):
        """
        Plot the loss curve of the best model after training.
        
        This method plots the training and validation losses over the epochs, providing a visual representation of the
        model's learning process. The loss curve can help in diagnosing issues such as overfitting or underfitting and
        in understanding how well the model has learned from the training data.
        
        Parameters
        ----------
        log_y : bool, optional
            Whether to use a logarithmic scale for the y-axis (loss values), by default True. This can be useful for
            visualizing loss values that span several orders of magnitude.
        log_x : bool, optional
            Whether to use a logarithmic scale for the x-axis (epoch values), by default True. This can be useful for
            visualizing training progress over many epochs.
        max_y : float, optional
            The maximum value for the y-axis (loss values), by default None. This can be useful for setting a fixed
            upper limit for the y-axis.
        """
        assert hasattr(self, 'best_history'), "The model has not been trained yet. Please train the model before printing the loss curve."

        plt.plot(self.best_history.history["loss"], label="training set", alpha=0.5, color="indigo")
        plt.plot(self.best_history.history["val_loss"], label="validation set", alpha=0.5, color="seagreen")
        plt.legend()
        plt.title("Loss Function ($R_{\\oplus}$)")
        plt.xlabel("Epochs")
        plt.ylabel("Loss ($R_{\\oplus}$)")
        if log_x:
            plt.xscale("log")
        if log_y:
            plt.yscale("log")
        plt.grid()
        # ignore first 100 epochs for y-axis limit
        # plt.ylim(ymax=max(max(self.best_history.history["loss"][100:]), max(self.best_history.history["val_loss"][100:])))
        if max_y is not None:
            plt.ylim(ymax=max_y)

        plt.savefig(self.model.name+"_plot_loss.png", dpi=300)
        plt.show()
        plt.close()


    # ---------- MAIN TRAINING LOOP ----------
    def _train_single_model(self,
                           model : keras.Model,
                           callbacks : list[keras.callbacks.Callback],
                           train_features : pd.DataFrame,
                           train_labels : pd.Series,
                           val_features : pd.DataFrame,
                           val_labels : pd.Series) -> keras.callbacks.History:
        """
        Train the provided model using the specified random seed for reproducibility. The training process involves
        compiling the model, fitting it to the training data, and applying callbacks for learning rate adjustment and
        early stopping.

        Parameters
        ----------
        model : keras.Model
            The Keras model to be trained.
        callbacks : list[keras.callbacks.Callback]
            A list of Keras callbacks to be applied during training, such as learning rate adjustment and early
            stopping.
        train_features : pd.DataFrame
            The features of the training dataset.
        train_labels : pd.Series
            The labels of the training dataset.
        val_features : pd.DataFrame
            The features of the validation dataset.
        val_labels : pd.Series
            The labels of the validation dataset.

        Returns
        -------
        keras.callbacks.History
            The training history of the model.
        """
        # Compile and fit the model
        model.compile(loss="mean_squared_error",
                      optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      metrics=["mean_squared_error"])

        history = model.fit(
            train_features,
            train_labels,
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            validation_data=(val_features, val_labels),
            callbacks=callbacks,
            verbose=0,
            shuffle=True)

        return history

    def train_models(self,
                     input_features: pd.DataFrame,
                     input_labels: pd.Series,
                     test_features: pd.DataFrame,
                     test_labels: pd.Series,
                     num_models: int = 10):
        """
        Train multiple models with different random initializations and keep the best one based on validation loss. This
        method also collects statistics about each model's performance.

        Parameters
        ----------
        input_features : pd.DataFrame
            The features of the input dataset, which will be used for training and validation.
        input_labels : pd.Series
            The labels of the input dataset, which will be split into training and validation sets based on the
            validation_split parameter.
        test_features : pd.DataFrame
            The features of the test dataset.
        test_labels : pd.Series
            The labels of the test dataset.
        num_models : int, optional
            The number of models to train, by default 10
        """
        # First create the validation set
        n_val = int(len(input_features) * self.validation_split)

        x_train, x_val = input_features[:-n_val], input_features[-n_val:]
        y_train, y_val = input_labels[:-n_val], input_labels[-n_val:]

        # Initialize arrays to store statistics about each model
        rp_means = np.zeros(num_models)
        rp_stds = np.zeros(num_models)
        losses = np.zeros(num_models)
        val_losses = np.zeros(num_models)
        list_num_epoch = np.zeros(num_models)

        for seed in range(num_models):
            print(f"Training model {seed}/{num_models}:")
            
            # Clear the Keras session to free up resources and avoid clutter from old models and layers
            keras.backend.clear_session()

            # Create a new model for each seed to ensure different initializations
            keras.utils.set_random_seed(seed)
            new_model = keras.models.clone_model(self.model)

            history = self._train_single_model(model=new_model,
                                               callbacks=self.callbacks,
                                               train_features=x_train,
                                               train_labels=y_train,
                                               val_features=x_val,
                                               val_labels=y_val)

            # records the best
            val_min = min(history.history['val_loss'])
            loss_min = min(history.history['loss'])
            print(f"Final val_loss of model {seed}/{num_models}: {val_min}")
            if val_min < self.best_val:
                self.best_val = val_min
                self.best_model = new_model
                self.best_history = history

            # records statistics about each model
            rp_error = self._evaluate_model(test_features=test_features,
                                            test_labels=test_labels)
            rp_means[seed] = np.mean(rp_error)
            rp_stds[seed] = np.std(rp_error)
            list_num_epoch[seed] = float(len(history.history['loss']))
            val_losses[seed] = val_min
            losses[seed] = loss_min

        # once all N_models have been trained, saves the best model
        self.model_manager.save_model_version(self.best_model,
                                              self.best_history,
                                              self.model_dir,
                                              self.model_name,
                                              self.version)

        # saves the statistics about all models
        self.losses = losses
        self.val_losses = val_losses
        self.list_num_epoch = list_num_epoch
        return rp_means, rp_stds, losses, val_losses


if __name__ == "__main__":
    # because our DNN are small in size, the final result depends on our initial (random) state. we train the same
    # multiple with N_models different initial conditions, and keep the best + some statistics.

    # Create some dummy data for training
    NUM_SAMPLES = 800
    NUM_FEATURES = 10
    input_features = pd.DataFrame(np.random.rand(NUM_SAMPLES, NUM_FEATURES),
                                  columns=[f"feature_{i}" for i in range(NUM_FEATURES)])
    input_labels = pd.Series(np.random.rand(NUM_SAMPLES), name="target")

    TEST_SAMPLES = 200
    test_features = pd.DataFrame(np.random.rand(TEST_SAMPLES, NUM_FEATURES),
                                 columns=[f"feature_{i}" for i in range(NUM_FEATURES)])
    test_labels = pd.Series(np.random.rand(TEST_SAMPLES), name="target")

    # calibrate a normaliser
    normaliser = keras.layers.Normalization(axis=-1)
    normaliser.adapt(np.array(input_features))

    # default values -- could be loaded from config?
    NUM_LAYERS = 4
    NUM_NEURONS = 8
    NUM_EPOCHS = 1000
    dnn_trainer = DNNTrainer(load_existing=False,
                             num_layers=NUM_LAYERS,
                             num_neurons=NUM_NEURONS,
                             num_epochs=NUM_EPOCHS,
                             input_shape=(NUM_FEATURES,),
                             normaliser=normaliser,
                             num_outputs=1,
                             version=1,)

    # Train the models
    rp_means, rp_stds, losses, val_losses = dnn_trainer.train_models(input_features=input_features,
                                                                    input_labels=input_labels,
                                                                    test_features=test_features,
                                                                    test_labels=test_labels,
                                                                    num_models=5)

    # Print the architecture performance of the best model
    dnn_trainer.print_architecture_performance()
