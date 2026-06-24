from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

# # File path and name file
# data_path = Path('data')
# file_name = 'DNN_data_IOP_Aguichine2021.dat'

# # Reading the data
# df = pd.read_table(data_path / file_name, sep=r"\s+")
# print(f"Number of lines of the data : {len(df)}")

def preprocessing(df, features, label):
    """
    Input:
        df: pandas dataframe
        features: list of features
        label: label

    output:
        Normalizer: mean and variance of each feature column

    """
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