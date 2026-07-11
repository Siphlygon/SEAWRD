import os
from typing import Sequence

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow_docs.modeling

from .config import SEAWRDConfig
from .config_manager import ConfigManager
from .model import DNNManager
from .utils import create_validation_split


class DNNTrainer:
    """
    A class to manage the training of a deep neural network (DNN) model. It can either load an existing model or
    generate a new one based on the provided parameters. The class also provides methods to generate callbacks for
    training and to print the architecture performance of the model.
    """

    def __init__(self,
                 model_manager: DNNManager,
                 config: SEAWRDConfig):
        """
        Initialise the DNNTrainer class. This class is responsible for managing the training of a deep neural network
        (DNN) model. It can either load an existing model or generate a new one based on the provided parameters.

        Parameters
        ----------
        model_manager : DNNManager
            An instance of the DNNManager class, which is responsible for creating and managing the DNN model.
        config : SEAWRDConfig
            An instance of the SEAWRDConfig class, which contains the full configuration parameters for the program.
        """
        self.model_manager = model_manager

        # readability aliases for the config sections
        self.model_config = config.model
        self.training_config = config.training
        self.compile_config = config.compile
        self.callbacks_config = config.callbacks
        self.device_config = config.device
        self.output_config = config.output

        # Model values
        self.model_dir = self.output_config.model_dir
        self.model_name = self.model_config.model_name or self.model_manager.model_name
        self.version = self.output_config.version

        # Initialise variables to keep track of the best model and its performance
        self.best_model = keras.Sequential()
        self.best_history = keras.callbacks.History()
        self.best_val = np.inf
        self.losses: np.ndarray = np.array([])
        self.val_losses: np.ndarray = np.array([])
        self.list_num_epoch: np.ndarray = np.array([])
        self._trained = False


    def _round_to_n_sig_figs(self, x: float | np.floating, n: int) -> float:
        """
        Round a number to a specified number of significant figures.

        Parameters
        ----------
        x : float | np.floating
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
        return float(round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1)))

    def print_architecture_performance(self):
        """
        Print the performance metrics of the model's architecture, including the model name, number of parameters,
        and statistics about the training and validation losses. This method provides a quick overview of the model's
        performance after training, allowing for easy comparison between different model architectures or training runs.
        """
        assert self._trained, (
            "The model has not been trained yet. Please train the model before printing performance metrics.")

        print(f"name:           {self.best_model.name}")
        print(f"num_param:      {self.best_model.count_params()}")
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
                        log_y: bool = True,
                        log_x: bool = True,
                        max_y: float | None = None):
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
        assert self._trained, (
            "The model has not been trained yet. Please train the model before printing the loss curve.")

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
        # plt.ylim(ymax=max(max(self.best_history.history["loss"][100:]),
        #   max(self.best_history.history["val_loss"][100:])))
        if max_y is not None:
            plt.ylim(ymax=max_y)

        if self.output_config.save_plots:
            plt.savefig(self.best_model.name+"_plot_loss.png", dpi=300)
        plt.show()
        plt.close()

    # ---------- MAIN TRAINING LOOP ----------
    def _generate_callbacks(self,
                            use_lr_scheduler: bool = True,
                            lr_monitor: str = 'val_loss',
                            lr_factor: float = 0.5,
                            lr_patience: int = 20,
                            min_lr: float = 1e-6,
                            use_early_stopping: bool = True,
                            early_stopping_monitor: str = 'val_loss',
                            early_stopping_patience: int = 50,
                            verbose: int = 1) -> list[keras.callbacks.Callback]:
        """
        Generate a list of callbacks for training a Keras model. These callbacks help in monitoring and controlling the
        training process.

        Parameters
        ----------
        use_lr_scheduler : bool, optional
            Whether to use learning rate scheduling, by default True.
        lr_monitor : str, optional
            The metric to monitor for learning rate scheduling, by default 'val_loss'.
        lr_factor : float, optional
            The factor by which the learning rate will be reduced, by default 0.5.
        lr_patience : int, optional
            The number of epochs with no improvement after which learning rate will be reduced, by default 15
        min_lr : float, optional
            The lower bound on the learning rate, by default 1e-7.
        use_early_stopping : bool, optional
            Whether to use early stopping, by default True.
        early_stopping_monitor : str, optional
            The metric to monitor for early stopping, by default 'val_loss'.
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
        callbacks = [tensorflow_docs.modeling.EpochDots()]

        if use_lr_scheduler:
            callbacks += [
                keras.callbacks.ReduceLROnPlateau( # reduce Learning Rate if the model does not improve
                    monitor=lr_monitor, # the metric to monitor for improvement
                    factor=lr_factor, # the factor by which the learning rate will be reduced. new_lr = lr * factor
                    patience=lr_patience, # number of epochs with no improvement after which lr will be reduced
                    min_lr=min_lr, # lower bound on the learning rate
                    verbose=verbose # verbosity mode, 1 = print messages when reducing learning rate
                )
            ]

        if use_early_stopping:
            callbacks += [
                keras.callbacks.EarlyStopping( # stop the training if the model does not improve
                    monitor=early_stopping_monitor,
                    patience=early_stopping_patience,
                    restore_best_weights=True, # restore model weights from the epoch with the best value of the monitor
                    verbose=verbose
                )
            ]

        return callbacks # type: ignore

    def _evaluate_model(self,
                        model: keras.Model,
                        test_features: np.ndarray | pd.DataFrame,
                        test_labels: np.ndarray | pd.Series) -> np.ndarray:
        """
        Evaluate a model's performance on the provided test dataset. This method computes the predictions of the model
        on the test features and calculates the error by comparing the predictions with the actual test labels.

        Parameters
        ----------
        model : keras.Model
            The model to be evaluated.
        test_features : np.ndarray | pd.DataFrame
            The features of the test dataset on which the model will be evaluated.
        test_labels : np.ndarray | pd.Series
            The labels of the test dataset against which the model will be evaluated.

        Returns
        -------
        np.ndarray
            The error between the actual test labels and the model's predictions. This is calculated as the difference
            between the actual values and the predicted values.
        """
        if isinstance(test_features, pd.DataFrame):
            test_features = test_features.to_numpy()
        if isinstance(test_labels, pd.Series):
            test_labels = test_labels.to_numpy()

        predictions = model.predict(test_features).reshape(-1)
        actual_values = test_labels.reshape(-1)
        errors = actual_values - predictions
        return errors

    def _train_single_model(self,
                            model: keras.Model,
                            train_features: np.ndarray | pd.DataFrame,
                            train_labels: np.ndarray | pd.Series,
                            val_features: np.ndarray | pd.DataFrame,
                            val_labels: np.ndarray | pd.Series) -> keras.callbacks.History:
        """
        Train the provided model using the specified random seed for reproducibility. The training process involves
        compiling the model, fitting it to the training data, and applying callbacks for learning rate adjustment and
        early stopping.

        Parameters
        ----------
        model : keras.Model
            The Keras model to be trained.
        train_features : np.ndarray | pd.DataFrame
            The features of the training dataset.
        train_labels : np.ndarray | pd.Series
            The labels of the training dataset.
        val_features : np.ndarray | pd.DataFrame
            The features of the validation dataset.
        val_labels : np.ndarray | pd.Series
            The labels of the validation dataset.

        Returns
        -------
        keras.callbacks.History
            The training history of the model.
        """
        # Generate callbacks for training
        callbacks = self._generate_callbacks(
            use_lr_scheduler=self.callbacks_config.reduce_lr,
            lr_monitor=self.callbacks_config.reduce_lr_monitor,
            lr_factor=self.callbacks_config.reduce_lr_factor,
            lr_patience=self.callbacks_config.reduce_lr_patience,
            min_lr=self.callbacks_config.min_lr,
            use_early_stopping=self.callbacks_config.early_stopping,
            early_stopping_monitor=self.callbacks_config.early_stopping_monitor,
            early_stopping_patience=self.callbacks_config.early_stopping_patience,
            verbose=1
        )

        # Compile and fit the model
        DNNManager.compile_from_config(model=model, compile_config=self.compile_config)

        history = model.fit(
            train_features,
            train_labels,
            batch_size=self.training_config.batch_size,
            epochs=self.training_config.num_epochs,
            validation_data=(val_features, val_labels),
            callbacks=callbacks,
            verbose=0,  # type: ignore
            shuffle=self.training_config.shuffle,)

        return history

    def train_models(self,
                     input_features: np.ndarray | pd.DataFrame,
                     input_labels: np.ndarray | pd.Series,
                     test_features: np.ndarray | pd.DataFrame,
                     test_labels: np.ndarray | pd.Series,
                     feature_names: Sequence[str] | None = None,
                     label_name: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Train multiple models with different random initialisations and keep the best one based on validation loss. This
        method also collects statistics about each model's performance.

        Parameters
        ----------
        input_features : np.ndarray | pd.DataFrame
            The features of the input dataset, which will be used for training and validation.
        input_labels : np.ndarray | pd.Series
            The labels of the input dataset, which will be split into training and validation sets based on the
            validation_split parameter.
        test_features : np.ndarray | pd.DataFrame
            The features of the test dataset.
        test_labels : np.ndarray | pd.Series
            The labels of the test dataset.
        feature_names : Sequence[str] | None, optional
            The names of the input features, in training order. When provided (or inferable from a DataFrame passed as
            ``input_features``), a manifest is saved alongside the best model so it can later be paired with new data
            for prediction. By default None.
        label_name : str | None, optional
            The name of the label column being predicted, recorded in the manifest. Inferred from a labelled
            ``input_labels`` Series when not given. By default None.

        Returns
        -------
        pred_means : np.ndarray
            The mean prediction errors for each model.
        pred_stds : np.ndarray
            The standard deviation of prediction errors for each model.
        losses : np.ndarray
            The training losses for each model.
        val_losses : np.ndarray
            The validation losses for each model.
        """
        # Infer the feature/label names from the inputs when they were not supplied explicitly, so a manifest can be
        # written even when the caller simply hands over a labelled DataFrame/Series (the common notebook workflow).
        if feature_names is None and isinstance(input_features, pd.DataFrame):
            feature_names = list(input_features.columns)
        if label_name is None and isinstance(input_labels, pd.Series):
            label_name = input_labels.name

        x_train, y_train, x_val, y_val = create_validation_split(input_features,
                                                                 input_labels,
                                                                 self.training_config.validation_split)

        # Initialize arrays to store statistics about each model
        num_models = self.training_config.num_models
        pred_means = np.zeros(num_models)
        pred_stds = np.zeros(num_models)
        losses = np.zeros(num_models)
        val_losses = np.zeros(num_models)
        list_num_epoch = np.zeros(num_models)

        for seed in range(num_models):
            print(f"Training model {seed}/{num_models}:")

            # Clear the Keras session to free up resources and avoid clutter from old models and layers
            keras.backend.clear_session()

            # Create a new model for each seed to ensure different initializations
            keras.utils.set_random_seed(seed)
            new_model = self.model_manager.clone_model()

            history = self._train_single_model(model=new_model,
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
            pred_error = self._evaluate_model(model=new_model,
                                            test_features=test_features,
                                            test_labels=test_labels)
            pred_means[seed] = np.mean(pred_error)
            pred_stds[seed] = np.std(pred_error)
            list_num_epoch[seed] = float(len(history.history['loss']))
            val_losses[seed] = val_min
            losses[seed] = loss_min

        # once all N_models have been trained, saves the best model
        if self.output_config.save_model:
            self.model_manager.save_model_version(self.best_model,
                                                self.best_history,
                                                self.model_dir,
                                                self.model_name,
                                                self.version,
                                                feature_names=feature_names,
                                                label_name=label_name)

        # saves the statistics about all models
        self.losses = losses
        self.val_losses = val_losses
        self.list_num_epoch = list_num_epoch
        self._trained = True
        return pred_means, pred_stds, losses, val_losses


if __name__ == "__main__":
    # because our DNN are small in size, the final result depends on our initial (random) state. we train the same
    # multiple with N_models different initial conditions, and keep the best + some statistics.

    # Create some dummy data for training
    print("Generating dummy data for training...")
    NUM_SAMPLES = 800
    NUM_FEATURES = 10
    i_features = pd.DataFrame(np.random.rand(NUM_SAMPLES, NUM_FEATURES),
                                  columns=[f"feature_{i}" for i in range(NUM_FEATURES)])
    i_labels = pd.Series(np.random.rand(NUM_SAMPLES), name="target")

    TEST_SAMPLES = 200
    t_features = pd.DataFrame(np.random.rand(TEST_SAMPLES, NUM_FEATURES),
                                 columns=[f"feature_{i}" for i in range(NUM_FEATURES)])
    t_labels = pd.Series(np.random.rand(TEST_SAMPLES), name="target")

    # calibrate a normaliser
    print("Calibrating normaliser...")
    normaliser = keras.layers.Normalization(axis=-1)
    normaliser.adapt(np.array(i_features))

    # it would take a fair amount of lines to create a dummy config; we'll just load the default config for now
    print("Loading default configuration...")
    cfgm = ConfigManager.from_toml("seawrd/seawrd_default.toml")
    cfg = cfgm.config

    # Create a DNNManager instance with the loaded configuration and the normaliser
    print("Creating DNNManager and DNNTrainer instances...")
    dnn_manager = DNNManager.from_config(model_config=cfg.model,
                                         input_shape=(NUM_FEATURES,),
                                         normaliser=normaliser)
    dnn_trainer = DNNTrainer(model_manager=dnn_manager, config=cfg)

    # Train the models
    print("Training models...")
    rp_means, rp_stds, rp_losses, rp_val_losses = dnn_trainer.train_models(
        input_features=i_features,
        input_labels=i_labels,
        test_features=t_features,
        test_labels=t_labels)

    # Print the architecture performance of the best model
    dnn_trainer.print_architecture_performance()
