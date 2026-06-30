from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .utils import fit_normaliser

if TYPE_CHECKING:
    import keras


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
        quality_column: str | None = "errcode", #Name of filter column
        quality_value: int | None = 0, #The value we allow through the filter
        normalise: bool = True, #Whether or not to fit a Keras normalisation layer
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
        quality_column : str | None, optional
            Column name used for filtering poor-quality rows, by default None, which means no filtering is applied
        quality_value : int | None, optional
            Value in the quality column that indicates a good-quality row, by default 0
        normalise : bool, optional
            Whether to fit a Keras normalisation layer to the features, by default True
        """
        # Copies the dataframe to protect the original, or try converting df into a pandas DataFrame
        self.df = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
        self.features = list(features) if features is not None else None
        self.label = label
        self.test_size = test_size
        self.random_state = random_state
        self.quality_column = quality_column
        self.quality_value = quality_value
        self.normalise = normalise

        self.feature_names_: Optional[list[str]] = None
        self.label_name_: Optional[str] = None
        self.normaliser: Optional[keras.layers.Normalization] = None

        # Run the preparation step to validate inputs, filter data, and derive features/label
        self._prepare()


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

    def _filter_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter rows based on quality column values.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to filter.

        Returns
        -------
        pd.DataFrame
            Filtered pandas DataFrame.
        """
        if self.quality_column not in df.columns:
            raise ValueError(f"Quality column '{self.quality_column}' is not present in the dataframe")

        df = df[df[self.quality_column] == self.quality_value]
        return df.copy()

    def _derive_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive new columns based on existing columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input pandas DataFrame to derive columns.

        Returns
        -------
        pd.DataFrame
            pandas DataFrame with derived columns.
        """
        for derived_name, source_columns in self.DERIVED_COLUMNS.items():
            if derived_name not in df.columns and all(col in df.columns for col in source_columns):
                df[derived_name] = df[list(source_columns)].sum(axis=1)
        return df

    def _infer_label(self, df: pd.DataFrame) -> str:
        """
        Infer the label column name from the pandas DataFrame.
        
        If the label is explicitly provided, it checks for its existence in the DataFrame. If not provided, it defaults
        to "R_p" if present, which is the planet radius and not contained explicitly within the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Input pandas DataFrame to infer the label column from.

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

    def _infer_features(self, df: pd.DataFrame, label: str) -> list[str]:
        """
        Infer feature column names from the pandas DataFrame, excluding the label and quality columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input pandas DataFrame to infer feature columns from.
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

    def _prepare(self) -> None:
        """
        Prepare the DataFrame for training by validating inputs, filtering poor-quality rows, deriving new columns, and
        inferring features and label.

        Raises
        ------
        ValueError
            If no feature columns are identified for training.
        """
        self._validate_inputs()
        df = self.df.copy()

        if self.quality_column is not None:
            df = self._filter_quality(df)
        df = self._derive_columns(df)

        self.label_name_ = self._infer_label(df)
        self.feature_names_ = self._infer_features(df, self.label_name_)

        if not self.feature_names_:
            raise ValueError("No feature columns were identified for training")

        self.df = df[self.feature_names_ + [self.label_name_]].dropna().reset_index(drop=True)
        self._prepared = True


    # ---------- DATA SPLITTING AND NORMALIZATION METHODS ----------
    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the pandas DataFrame into training and testing sets.

        Returns
        -------
        trainn_df : pd.DataFrame
            Training features pandas DataFrame.
        test_df : pd.DataFrame
            Testing features pandas DataFrame.
        train_labels : pd.Series
            Training labels pandas Series.
        test_labels : pd.Series
            Testing labels pandas Series.
        """
        train_df = self.df.sample(frac=1.0 - self.test_size, random_state=self.random_state)
        test_df = self.df.drop(train_df.index)

        return (
            train_df[self.feature_names_].copy(),
            test_df[self.feature_names_].copy(),
            train_df[self.label_name_].copy(),
            test_df[self.label_name_].copy(),
        )


    # ---------- PUBLIC METHODS ----------
    def get_training_data(self,
                          return_array: bool = True,
        ) -> Tuple[
                Optional[keras.layers.Normalization],
                Union[pd.DataFrame, np.ndarray],
                Union[pd.DataFrame, np.ndarray],
                Union[pd.Series, np.ndarray],
                Union[pd.Series, np.ndarray],
            ]:
        """
        Return train/test splits and an optional fitted normaliser.

        Parameters
        ----------
        return_array : bool, default=True
            If True, return NumPy arrays instead of DataFrames/Series. These arrays will be of type float32, suitable
            for Keras models.

        Returns
        -------
        normaliser : Optional[keras.layers.Normalization]
            The fitted Keras Normalization layer if normalization is enabled, otherwise None.
        train_features : Union[pd.DataFrame, np.ndarray]
            Training features as a pandas DataFrame or NumPy array.
        test_features : Union[pd.DataFrame, np.ndarray]
            Testing features as a pandas DataFrame or NumPy array.
        train_labels : Union[pd.Series, np.ndarray]
            Training labels as a pandas Series or NumPy array.
        test_labels : Union[pd.Series, np.ndarray]
            Testing labels as a pandas Series or NumPy array.
        """
        train_features, test_features, train_labels, test_labels = self.split()
        if self.normalise:
            normaliser = fit_normaliser(train_features)
            self.normaliser = normaliser
        else:
            normaliser = None

        if return_array:
            return (
                normaliser,
                train_features.to_numpy(dtype=np.float32),
                test_features.to_numpy(dtype=np.float32),
                train_labels.to_numpy(dtype=np.float32),
                test_labels.to_numpy(dtype=np.float32),
            )

        return normaliser, train_features, test_features, train_labels, test_labels

    # def to_tf_dataset(
    #     self,
    #     batch_size: int = 32,
    #     shuffle: bool = True,
    # ) -> tf.data.Dataset:
    #     """
    #     Convert the prepared data into a TensorFlow Dataset.

    #     Parameters
    #     ----------
    #     batch_size : int, optional
    #         The batch size for the dataset, by default 32
    #     shuffle : bool, optional
    #         Whether to shuffle the dataset, by default True

    #     Returns
    #     -------
    #     tf.data.Dataset
    #         The converted TensorFlow Dataset.
    #     """
    #     # Get the training data as NumPy arrays
    #     _, features, _, labels, _ = self.get_training_data(return_array=True)

    #     dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    #     if shuffle:
    #         dataset = dataset.shuffle(buffer_size=len(features), seed=self.random_state)
    #     return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def save_to_npz(self, path: Union[str, os.PathLike]) -> None:
        """
        Save the prepared data to a .npz file.

        Parameters
        ----------
        path : Union[str, os.PathLike]
            The file path where the .npz file will be saved.
        """
        _, features, test_features, labels, test_labels = self.get_training_data(return_array=True)
        np.savez(
            path,
            input_features=features,
            test_features=test_features,
            input_labels=labels,
            test_labels=test_labels,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess planetary data for ML regression training.")
    parser.add_argument("input_path", type=str, help="Path to the input data file")
    parser.add_argument("--output-path", type=str, help="Path to save the preprocessed .npz file")
    args = parser.parse_args()

    # Load the input data
    input_data = pd.read_table(args.input_path, sep=r"\s+")
    features = [col for col in input_data.columns if col not in ["R_p", "M_p", "errcode"]]
    label = "R_p"

    # Initialise the DataPreprocessor with the input data
    preprocessor = DataPreprocessor(input_data, features, label)

    # If specified, save the preprocessed data to a .npz file
    if args.output_path:
        preprocessor.save_to_npz(args.output_path)
