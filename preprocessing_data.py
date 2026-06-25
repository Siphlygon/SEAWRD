from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame
from tensorflow import keras
from typing import Optional, Sequence, Tuple, Union


class DataPreprocessor:
    """Preprocess planetary data for ML regression training.

    This class validates a pandas DataFrame, derives common planet-level
    features, filters poor-quality rows, and returns train/test splits
    ready for a Keras regression model.
    """

    DEFAULT_COLUMN_RENAMES = {"x_core'": "x_core"}
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
        self.df = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df) #Copies the dataframe to protect the original, or try converting df into a pandas DataFrame
        self.features = list(features) if features is not None else None #Converts features to a list if it's not None, otherwise sets it to None
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

    def _validate_inputs(self) -> None:
        if not isinstance(self.df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be a float between 0 and 1")
        if self.label is not None and not isinstance(self.label, str):
            raise TypeError("label must be a string")

    def _rename_columns(self, df: DataFrame) -> DataFrame:
        return df.rename(columns=self.DEFAULT_COLUMN_RENAMES)

    def _filter_quality(self, df: DataFrame) -> DataFrame:
        if self.quality_column in df.columns:
            df = df[df[self.quality_column] == self.quality_value]
        return df.copy()

    def _derive_columns(self, df: DataFrame) -> DataFrame:
        for derived_name, source_columns in self.DERIVED_COLUMNS.items():
            if derived_name not in df.columns and all(col in df.columns for col in source_columns):
                df[derived_name] = df[list(source_columns)].sum(axis=1)
        return df

    def _infer_label(self, df: DataFrame) -> str:
        if self.label is not None:
            if self.label not in df.columns:
                raise ValueError(f"Label '{self.label}' is not present in the dataframe")
            return self.label
        if "R_p" in df.columns:
            return "R_p"
        raise ValueError("Unable to infer a label column. Provide label explicitly.")

    def _infer_features(self, df: DataFrame, label: str) -> list[str]:
        if self.features is not None:
            missing = [col for col in self.features if col not in df.columns]
            if missing:
                raise ValueError(f"Requested features not found: {missing}")
            return list(self.features)

        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        return [col for col in numeric_columns if col not in {label, self.quality_column}]

    def prepare(self) -> None:
        """Run preprocessing and infer feature/label names."""
        self._validate_inputs()

        df = self._rename_columns(self.df)
        df = self._filter_quality(df)
        df = self._derive_columns(df)

        self.label_name_ = self._infer_label(df)
        self.feature_names_ = self._infer_features(df, self.label_name_)

        if not self.feature_names_:
            raise ValueError("No feature columns were identified for training")

        self.df = df[self.feature_names_ + [self.label_name_]].dropna().reset_index(drop=True)
        self._prepared = True

    def split(self) -> Tuple[DataFrame, DataFrame, pd.Series, pd.Series]:
        """Return train/test feature and label splits."""
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
        """Fit and return a Keras normalization layer for training features."""
        normalizer = keras.layers.Normalization(axis=-1, name="feature_normalizer")
        normalizer.adapt(train_features.to_numpy(dtype=np.float32))
        self.normalizer = normalizer
        return normalizer

    def get_training_data(
        self,
        return_array: bool = False,
    ) -> Tuple[
        Optional[keras.layers.Normalization],
        Union[DataFrame, np.ndarray],
        Union[DataFrame, np.ndarray],
        Union[pd.Series, np.ndarray],
        Union[pd.Series, np.ndarray],
    ]:
        """Return train/test splits and an optional fitted normalizer."""
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
    ) -> "tf.data.Dataset":
        """Convert features and labels to a TensorFlow Dataset."""
        import tensorflow as tf

        if isinstance(features, pd.DataFrame):
            features = features.to_numpy(dtype=np.float32)
        if isinstance(labels, pd.Series):
            labels = labels.to_numpy(dtype=np.float32)

        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(features), seed=self.random_state)
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
