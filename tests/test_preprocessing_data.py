"""
Unit tests for the DataPreprocessor class in the seawrd.preprocessing_data module.
"""
import numpy as np
import pytest
import pandas as pd

from seawrd.preprocessing_data import DataPreprocessor


def test_prepare_derives_radius_and_filters_quality_rows(planet_df : pd.DataFrame):
    """
    Test that the data preprocessor derives the radius and filters quality rows.

    Parameters
    ----------
    planet_df : pandas.DataFrame
        The input DataFrame containing planetary data.
    """
    features = ["x_core'", "x_H2O", "T_irr", "T_b", "M_b", "M_a"]
    p = DataPreprocessor(
        planet_df,
        features=features,
        label="R_p",
        quality_column="errcode",
        quality_value=0,
    )

    assert len(p.df) == 5  # One row with errcode != 0 should be filtered out
    assert p.label_name_ == "R_p"
    assert p.feature_names_ == ["x_core'", "x_H2O", "T_irr", "T_b", "M_b", "M_a"]

    # get original planet_df without the row with errcode != 0
    filtered_planet_df = planet_df[planet_df["errcode"] == 0]
    np.testing.assert_allclose(p.df["R_p"], filtered_planet_df["R_a"] + filtered_planet_df["R_b"],
                               err_msg="Derived radius R_p does not match the sum of R_a and R_b.")


def test_save_to_npz_persists_feature_and_label_names(planet_df : pd.DataFrame, tmp_path):
    """
    Test that save_to_npz records the feature names and label name alongside the data arrays, so a downstream training
    run can write a prediction manifest.

    Parameters
    ----------
    planet_df : pandas.DataFrame
        The input DataFrame containing planetary data.
    tmp_path : pathlib.Path
        A temporary directory provided by pytest.
    """
    features = ["x_core'", "x_H2O", "T_irr", "T_b", "M_b", "M_a"]
    # normalise=False keeps this test free of any keras dependency
    p = DataPreprocessor(planet_df, features=features, label="R_p", normalise=False)

    path = tmp_path / "bundle.npz"
    p.save_to_npz(path)

    with np.load(path, allow_pickle=False) as bundle:
        assert [str(name) for name in bundle["feature_names"]] == features, "Feature names were not persisted in order"
        assert str(bundle["label_name"]) == "R_p", "Label name was not persisted"


def test_prepare_does_not_mutate_input_dataframe(planet_df : pd.DataFrame):
    """
    Test that the data preprocessor does not mutate the input DataFrame.

    Parameters
    ----------
    planet_df : pandas.DataFrame
        The input DataFrame containing planetary data.
    """
    original_columns = set(planet_df.columns)

    DataPreprocessor(planet_df, label="R_p")

    assert set(planet_df.columns) == original_columns
    assert "R_p" not in planet_df.columns


def test_missing_explicit_label_raises(planet_df : pd.DataFrame):
    """
    Test that the data preprocessor raises a ValueError when the specified label column is missing.

    Parameters
    ----------
    planet_df : pandas.DataFrame
        The input DataFrame containing planetary data.
    """
    with pytest.raises(ValueError, match="Label"):
        DataPreprocessor(planet_df, label="missing")


def test_missing_explicit_feature_raises(planet_df : pd.DataFrame):
    """
    Test that the data preprocessor raises a ValueError when the specified feature column is missing.

    Parameters
    ----------
    planet_df : pandas.DataFrame
        The input DataFrame containing planetary data.
    """

    with pytest.raises(ValueError, match="Requested features"):
        DataPreprocessor(planet_df, features=["not_a_column"], label="R_p")


def test_quality_filtering_removes_correct_rows(planet_df : pd.DataFrame):
    """
    Test that the data preprocessor correctly filters rows based on the quality column.

    Parameters
    ----------
    planet_df : pandas.DataFrame
        The input DataFrame containing planetary data.
    """
    p = DataPreprocessor(planet_df, label="R_p", quality_column="errcode", quality_value=0)

    assert len(p.df) == 5  # One row with errcode != 0 should be filtered out

    # Test with a different quality value
    p2 = DataPreprocessor(planet_df, label="R_p", quality_column="errcode", quality_value=1)

    assert len(p2.df) == 1  # Only one row with errcode == 1 should remain


@pytest.mark.parametrize("test_size", [0, 1, -0.1, 1.1])
def test_invalid_test_size_rejected(planet_df : pd.DataFrame, test_size : float):
    """
    Test that the data preprocessor rejects invalid test sizes.

    Parameters
    ----------
    planet_df : pandas.DataFrame
        The input DataFrame containing planetary data.
    test_size : float
        The size of the test set.
    """
    with pytest.raises(ValueError, match="test_size"):
        DataPreprocessor(planet_df, test_size=test_size)


def test_get_training_data_returns_float32_arrays(planet_df : pd.DataFrame):
    """
    Test that the data preprocessor returns float32 arrays.

    Parameters
    ----------
    planet_df : pandas.DataFrame
        The input DataFrame containing planetary data.
    """
    p = DataPreprocessor(
        planet_df,
        features=["x_core'", "x_H2O", "T_irr", "T_b", "M_b", "M_a"],
        label="R_p",
        test_size=0.25,
        random_state=123,
    )

    normaliser, x_train, x_test, y_train, y_test = p.get_training_data(return_array=True)

    assert normaliser is not None
    assert x_train.dtype == np.float32
    assert x_test.dtype == np.float32
    assert y_train.dtype == np.float32
    assert y_test.dtype == np.float32
    assert x_train.shape[1] == 6


def test_get_training_data_returns_dataframe(planet_df : pd.DataFrame):
    """
    Test that the data preprocessor returns DataFrames when return_array=False.

    Parameters
    ----------
    planet_df : pandas.DataFrame
        The input DataFrame containing planetary data.
    """
    p = DataPreprocessor(
        planet_df,
        features=["x_core'", "x_H2O", "T_irr", "T_b", "M_b", "M_a"],
        label="R_p",
        test_size=0.25,
        random_state=123,
    )

    normaliser, x_train, x_test, y_train, y_test = p.get_training_data(return_array=False)

    assert normaliser is not None
    assert isinstance(x_train, pd.DataFrame)
    assert isinstance(x_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)


def test_get_training_data_splits_correctly(planet_df : pd.DataFrame):
    """
    Test that the data preprocessor splits the data correctly into training and test sets.

    Parameters
    ----------
    planet_df : pandas.DataFrame
        The input DataFrame containing planetary data.
    """
    p = DataPreprocessor(
        planet_df,
        features=["x_core'", "x_H2O", "T_irr", "T_b", "M_b", "M_a"],
        label="R_p",
        test_size=0.25,
        random_state=123,
    )

    _, x_train, x_test, y_train, y_test = p.get_training_data(return_array=True)

    assert len(x_train) == 4  # 75% of 5 rows after filtering
    assert len(x_test) == 1   # 25% of 5 rows after filtering


def test_get_training_data_with_no_quality_filter(planet_df : pd.DataFrame):
    """
    Test that the data preprocessor works correctly when no quality filter is applied.

    Parameters
    ----------
    planet_df : pandas.DataFrame
        The input DataFrame containing planetary data.
    """
    p = DataPreprocessor(
        planet_df,
        features=["x_core'", "x_H2O", "T_irr", "T_b", "M_b", "M_a"],
        quality_column=None,
        label="R_p",
        test_size=0.25,
        random_state=123,
    )

    _, x_train, x_test, y_train, y_test = p.get_training_data(return_array=True)

    assert len(x_train) == 4  # 75% of 6 rows without filtering
    assert len(x_test) == 2   # 25% of 6 rows without filtering


def test_get_training_data_with_different_random_states(planet_df : pd.DataFrame):
    """
    Test that the data preprocessor produces different splits with different random states.

    Parameters
    ----------
    planet_df : pandas.DataFrame
        The input DataFrame containing planetary data.
    """
    p1 = DataPreprocessor(
        planet_df,
        features=["x_core'", "x_H2O", "T_irr", "T_b", "M_b", "M_a"],
        label="R_p",
        test_size=0.25,
        random_state=123,
    )

    p2 = DataPreprocessor(
        planet_df,
        features=["x_core'", "x_H2O", "T_irr", "T_b", "M_b", "M_a"],
        label="R_p",
        test_size=0.25,
        random_state=456,
    )

    _, x_train1, x_test1, y_train1, y_test1 = p1.get_training_data(return_array=True)
    _, x_train2, x_test2, y_train2, y_test2 = p2.get_training_data(return_array=True)

    # Check that the training and test sets are different for different random states
    assert not np.array_equal(x_train1, x_train2)
    assert not np.array_equal(x_test1, x_test2)


def test_k_fold_splits_yields_correct_number_of_folds(planet_df : pd.DataFrame):
    """
    Test that k_fold_splits yields exactly n_splits folds.

    Parameters
    ----------
    planet_df : pandas.DataFrame
        The input DataFrame containing planetary data.
    """
    p = DataPreprocessor(
        planet_df,
        features=["x_core'", "x_H2O", "T_irr", "T_b", "M_b", "M_a"],
        label="R_p",
        normalise=False,
    )

    folds = list(p.k_fold_splits(n_splits=5, random_state=0))
    assert len(folds) == 5  # 5 rows remain after quality filtering


def test_k_fold_splits_folds_are_disjoint_and_cover_all_rows(planet_df : pd.DataFrame):
    """
    Test that each fold's validation set is disjoint from its training set, and that the validation sets across all
    folds together cover every row of the prepared data exactly once.

    Parameters
    ----------
    planet_df : pandas.DataFrame
        The input DataFrame containing planetary data.
    """
    p = DataPreprocessor(
        planet_df,
        features=["x_core'", "x_H2O", "T_irr", "T_b", "M_b", "M_a"],
        label="R_p",
        normalise=False,
    )

    all_val_indices = []
    for train_features, val_features, train_labels, val_labels in p.k_fold_splits(n_splits=5, random_state=0):
        assert set(train_features.index).isdisjoint(set(val_features.index)), (
            "Training and validation rows should never overlap within a fold")
        assert list(train_features.index) == list(train_labels.index)
        assert list(val_features.index) == list(val_labels.index)
        all_val_indices.extend(val_features.index)

    assert sorted(all_val_indices) == sorted(p.df.index), (
        "Every row should appear as a validation row in exactly one fold")


def test_k_fold_splits_raises_for_n_splits_less_than_two(planet_df : pd.DataFrame):
    """
    Test that k_fold_splits raises a ValueError when n_splits is less than 2.

    Parameters
    ----------
    planet_df : pandas.DataFrame
        The input DataFrame containing planetary data.
    """
    p = DataPreprocessor(planet_df, label="R_p", normalise=False)

    with pytest.raises(ValueError, match="n_splits"):
        next(p.k_fold_splits(n_splits=1))


def test_k_fold_splits_raises_when_n_splits_exceeds_row_count(planet_df : pd.DataFrame):
    """
    Test that k_fold_splits raises a ValueError when n_splits exceeds the number of available rows.

    Parameters
    ----------
    planet_df : pandas.DataFrame
        The input DataFrame containing planetary data.
    """
    p = DataPreprocessor(planet_df, label="R_p", normalise=False)  # 5 rows after filtering

    with pytest.raises(ValueError, match="n_splits"):
        next(p.k_fold_splits(n_splits=10))


def test_k_fold_splits_reproducible_with_same_random_state(planet_df : pd.DataFrame):
    """
    Test that k_fold_splits produces identical folds when called twice with the same random_state.

    Parameters
    ----------
    planet_df : pandas.DataFrame
        The input DataFrame containing planetary data.
    """
    p = DataPreprocessor(
        planet_df,
        features=["x_core'", "x_H2O", "T_irr", "T_b", "M_b", "M_a"],
        label="R_p",
        normalise=False,
    )

    folds_a = [val.index.tolist() for _, val, _, _ in p.k_fold_splits(n_splits=3, random_state=42)]
    folds_b = [val.index.tolist() for _, val, _, _ in p.k_fold_splits(n_splits=3, random_state=42)]

    assert folds_a == folds_b


# @pytest.mark.tf
# def test_correct_tf_dataset_creation(planet_df : pd.DataFrame):
#     """
#     Test that the data preprocessor creates TensorFlow datasets correctly.

#     Parameters
#     ----------
#     planet_df : pandas.DataFrame
#         The input DataFrame containing planetary data.
#     """
#     p = DataPreprocessor(
#         planet_df,
#         features=["x_core'", "x_H2O", "T_irr", "T_b", "M_b", "M_a"],
#         label="R_p",
#         test_size=0.25,
#         random_state=123,
#     )

#     tf_dataset = p.to_tf_dataset(
#         batch_size=32,
#         shuffle=True,
#     )

#     # Check that the datasets are not empty
#     assert len(list(tf_dataset)) > 0

#     # Check that the batch size is correct
#     for x_batch, y_batch in tf_dataset:
#         assert x_batch.shape[0] <= 32  # Last batch may be smaller
#         assert y_batch.shape[0] <= 32
