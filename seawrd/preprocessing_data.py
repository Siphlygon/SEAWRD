from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union
import os

os.environ["KERAS_BACKEND"] = "tensorflow" 
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import DataFrame


class DataPreprocessor:
    """
    Preprocess planetary data for ML regression training.

    This class validates a pandas DataFrame, derives common planet-level
    features, filters poor-quality rows, and returns train/test splits
    ready for a Keras regression model.
    """

    DERIVED_COLUMNS = {
        "R_p": ("R_a", "R_b"),
        "M_p": ("M_a", "M_b"),
    }

    def __init__(
        self,
        df: pd.DataFrame, #Default expects a pandas DataFrame
        features: Optional[Sequence[str]] = None, #Default expects a list of strings
        label: Optional[str] = None, #Default expects a string, i.e. R_p
        test_size: float = 0.2, #Fraction of data to use for testing
        random_state: Optional[int] = 0, #Random state for reproducibility
        quality_column: str = "errcode", #Name of filter column
        quality_value=0, #The value we allow through the filter
        normalize: bool = True, #Whether or not to fit a Keras normalization layer
    ):
        """
        Initialize the DataPreprocessor with a DataFrame and optional parameters.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing planetary data.
        features : Optional[Sequence[str]], optional
            List of feature column names, by default None
        label : Optional[str], optional
            Name of the label column, by default None
        test_size : float, optional
            Fraction of data to use for testing, by default 0.2
        random_state : Optional[int], optional
            Random seed for reproducibility, by default 0
        quality_column : str, optional
            Column name used for filtering poor-quality rows, by default "errcode"
        quality_value : optional
            Value in the quality column that indicates a good-quality row, by default 0
        normalize : bool, optional
            Whether to fit a Keras normalization layer to the features, by default True
        """
        # Copies the dataframe to protect the original, or try converting df into a pandas DataFrame
        self.df = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
        self.features = list(features) if features is not None else None
        self.label = label
        self.test_size = test_size
        self.random_state = random_state
        self.quality_column = quality_column
        self.quality_value = quality_value
        self.normalize = normalize

        self.feature_names_: Optional[list[str]] = None
        self.label_name_: Optional[str] = None
        self.normalizer: Optional[keras.layers.Normalization] = None
        self._prepared = False


    # ---------- DATAFILE PREPARATION METHODS ----------
    def _validate_inputs(self) -> None:
        """
        Validate the input parameters for the DataPreprocessor.

        Raises
        ------
        TypeError
            If df is not a pandas DataFrame or if label is not a string.
        ValueError
            If test_size is not a float between 0 and 1, or if the label column is not found in the dataframe.
        """
        if not isinstance(self.df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be a float between 0 and 1")
        if self.label is not None and not isinstance(self.label, str):
            raise TypeError("label must be a string")

    def _filter_quality(self, df: DataFrame) -> DataFrame:
        """
        Filter rows based on quality column values.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame to filter.

        Returns
        -------
        DataFrame
            Filtered DataFrame.
        """
        if self.quality_column in df.columns:
            df = df[df[self.quality_column] == self.quality_value]
        return df.copy()

    def _derive_columns(self, df: DataFrame) -> DataFrame:
        """
        Derive new columns based on existing columns.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame to derive columns.

        Returns
        -------
        DataFrame
            DataFrame with derived columns.
        """
        for derived_name, source_columns in self.DERIVED_COLUMNS.items():
            if derived_name not in df.columns and all(col in df.columns for col in source_columns):
                df[derived_name] = df[list(source_columns)].sum(axis=1)
        return df

    def _infer_label(self, df: DataFrame) -> str:
        """
        Infer the label column name from the DataFrame.
        
        If the label is explicitly provided, it checks for its existence in the DataFrame. If not provided, it defaults
        to "R_p" if present, which is the planet radius and not contained explicitly within the dataset.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame to infer the label column from.

        Returns
        -------
        str
            Inferred label column name.

        Raises
        ------
        ValueError
            If the label column is not found in the dataframe.
        """
        if self.label is not None:
            if self.label not in df.columns:
                raise ValueError(f"Label '{self.label}' is not present in the dataframe")
            return self.label
        if "R_p" in df.columns:
            return "R_p"
        raise ValueError("Unable to infer a label column. Provide label explicitly.")

    def _infer_features(self, df: DataFrame, label: str) -> list[str]:
        """
        Infer feature column names from the DataFrame, excluding the label and quality columns.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame to infer feature columns from.
        label : str
            Label column name.

        Returns
        -------
        list[str]
            List of inferred feature column names.

        Raises
        ------
        ValueError
            If expected feature columns are not found within the provided DataFrame.
        """
        if self.features is not None:
            missing = [col for col in self.features if col not in df.columns]
            if missing:
                raise ValueError(f"Requested features not found: {missing}")
            return list(self.features)

        # We only consider numeric columns as features, excluding the label and quality columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        return [col for col in numeric_columns if col not in {label, self.quality_column}]

    def prepare(self) -> None:
        """
        Prepare the DataFrame for training by validating inputs, filtering poor-quality rows, deriving new columns, and
        inferring features and label.

        Raises
        ------
        ValueError
            If no feature columns are identified for training.
        """
        self._validate_inputs()

        df = self._filter_quality(self.df)
        df = self._derive_columns(df)

        self.label_name_ = self._infer_label(df)
        self.feature_names_ = self._infer_features(df, self.label_name_)

        if not self.feature_names_:
            raise ValueError("No feature columns were identified for training")

        self.df = df[self.feature_names_ + [self.label_name_]].dropna().reset_index(drop=True)
        self._prepared = True


    # ---------- DATA SPLITTING AND NORMALIZATION METHODS ----------
    def split(self) -> Tuple[DataFrame, DataFrame, pd.Series, pd.Series]:
        """
        Split the DataFrame into training and testing sets.

        Returns
        -------
        Tuple[DataFrame, DataFrame, pd.Series, pd.Series]
            Training and testing feature and label DataFrames/Series.
        """
        if not self._prepared:
            self.prepare()

        train_df = self.df.sample(frac=1.0 - self.test_size, random_state=self.random_state)
        test_df = self.df.drop(train_df.index)

        return (
            train_df[self.feature_names_].copy(),
            test_df[self.feature_names_].copy(),
            train_df[self.label_name_].copy(),
            test_df[self.label_name_].copy(),
        )

    def fit_normalizer(self, train_features: DataFrame) -> keras.layers.Normalization:
        """
        Fit a Keras Normalization layer to the training features.

        Parameters
        ----------
        train_features : DataFrame
            Training feature DataFrame to fit the normalizer.

        Returns
        -------
        keras.layers.Normalization
            Fitted Keras Normalization layer.
        """
        normalizer = keras.layers.Normalization(axis=-1, name="feature_normalizer")
        normalizer.adapt(train_features.to_numpy(dtype=np.float32))
        self.normalizer = normalizer
        return normalizer


    # ---------- PUBLIC METHODS ----------
    def get_training_data(self,
                          return_array: bool = False,
        ) -> Tuple[
                Optional[keras.layers.Normalization],
                Union[DataFrame, np.ndarray],
                Union[DataFrame, np.ndarray],
                Union[pd.Series, np.ndarray],
                Union[pd.Series, np.ndarray],
            ]:
        """
        Return train/test splits and an optional fitted normalizer.

        Parameters
        ----------
        return_array : bool, default=False
            If True, return NumPy arrays instead of DataFrames/Series.

        Returns
        -------
        Tuple[Optional[keras.layers.Normalization], Union[DataFrame, np.ndarray], Union[DataFrame, np.ndarray], Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]
            Fitted normalizer (if applicable) and training/testing feature/label data.
        """
        train_features, test_features, train_labels, test_labels = self.split()
        normalizer = self.fit_normalizer(train_features) if self.normalize else None

        if return_array:
            return (
                normalizer,
                train_features.to_numpy(dtype=np.float32),
                test_features.to_numpy(dtype=np.float32),
                train_labels.to_numpy(dtype=np.float32),
                test_labels.to_numpy(dtype=np.float32),
            )

        return normalizer, train_features, test_features, train_labels, test_labels

    def to_tf_dataset(
        self,
        features: Union[DataFrame, np.ndarray],
        labels: Union[pd.Series, np.ndarray],
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> tf.data.Dataset:
        """
        Convert features and labels into a TensorFlow Dataset.

        Parameters
        ----------
        features : Union[DataFrame, np.ndarray]
            The input features for the dataset.
        labels : Union[pd.Series, np.ndarray]
            The labels for the dataset.
        batch_size : int, optional
            The batch size for the dataset, by default 32
        shuffle : bool, optional
            Whether to shuffle the dataset, by default True

        Returns
        -------
        tf.data.Dataset
            The converted TensorFlow Dataset.
        """
        if isinstance(features, pd.DataFrame):
            features = features.to_numpy(dtype=np.float32)
        if isinstance(labels, pd.Series):
            labels = labels.to_numpy(dtype=np.float32)

        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(features), seed=self.random_state)
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    # Dummy data for testing the DataPreprocessor class
    data = {
        "R_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "R_b": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "M_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "M_b": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "errcode": [0, 0, 1, 0, 0, 0],
    }
    df = pd.DataFrame(data)
    preprocessor = DataPreprocessor(df, label="R_p", features=["R_a", "R_b", "M_a", "M_b"])
    preprocessor.prepare()

    print("Prepared DataFrame:")
    print(preprocessor.df)

    train_features, test_features, train_labels, test_labels = preprocessor.split()
    print("\nTrain Features:")
    print(train_features)
    print("\nTest Features:")
    print(test_features)
    print("\nTrain Labels:")
    print(train_labels)
    print("\nTest Labels:")
    print(test_labels)

    normalizer = preprocessor.fit_normalizer(train_features)
    print("\nFitted Normalizer Mean:")
    print(normalizer.mean.numpy())

    tf_dataset = preprocessor.to_tf_dataset(train_features, train_labels)
    print("\nTensorFlow Dataset:")
    for batch_features, batch_labels in tf_dataset.take(1):
        print("Batch Features:")
        print(batch_features.numpy())
        print("Batch Labels:")
        print(batch_labels.numpy())
