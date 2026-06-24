import numpy as np
import pandas as pd
from tensorflow import keras

class DataPreprocessor:
    def __init__(self, df: dataFrame, features: list, label: list):
        self.df = df
        self.features = features
        self.label = label

    def preprocessing(self):
        """ Data preprocesing
        Produce normalizer, test and train dataframes ready for input to keras

        Args:
        df (dataframe): pandas dataframe. Input data
        features (list): list of strings. List of features
        label (scalar): string. Label used as target

        Returns:
        tuple: normalizer, test and train dataframes 
        """
    
        df = self.df
        features = self.features
        label = self.label

        df = df[df["errcode"] == 0]
        print(f"Number of lines after quality cuts: {len(df)}")

        df = df.rename(columns={"x_core'": "x_core"})

        # Radius
        df["R_p"] = df["R_a"] + df["R_b"]

        # Mass
        df["M_p"] = df["M_a"] + df["M_b"]

        # train and test sample (80/20 split)
        train_dataset = df.sample(frac=0.8, random_state=0)
        test_dataset = df.drop(train_dataset.index)

        # features = what it knows/what it will predict on
        train_features = train_dataset[features].copy()
        test_features = test_dataset[features].copy()

        # labels = what it is predicting
        train_labels = train_dataset[label].copy()
        test_labels = test_dataset[label].copy()

        # The layer is now ready to normalize future data 
        normalizer = keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))

        return normalizer, train_features, test_features, train_labels, test_labels