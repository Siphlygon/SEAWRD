"""
Unit tests for the seawrd.cross_validation module: CrossValidationResult and KFoldCrossValidator.
"""
import numpy as np
import pandas as pd
import pytest

from seawrd.config import SEAWRDConfig
from seawrd.cross_validation import CrossValidationResult, KFoldCrossValidator
from seawrd.preprocessing_data import DataPreprocessor


# ----------------- Tests for CrossValidationResult -----------------
def test_cross_validation_result_to_dataframe():
    """
    Test that CrossValidationResult.to_dataframe returns one row per fold with the recorded metric values.
    """
    result = CrossValidationResult(fold_metrics=[
        {"rmse": 1.0, "mae": 0.5},
        {"rmse": 3.0, "mae": 1.5},
    ])

    df = result.to_dataframe()

    assert list(df["rmse"]) == [1.0, 3.0]
    assert list(df["mae"]) == [0.5, 1.5]


def test_cross_validation_result_summary_computes_mean_and_std():
    """
    Test that CrossValidationResult.summary reports the mean and standard deviation of each metric across folds.
    """
    result = CrossValidationResult(fold_metrics=[
        {"rmse": 1.0},
        {"rmse": 3.0},
    ])

    summary = result.summary()

    assert set(summary["rmse"]) == {"mean", "std"}
    assert summary["rmse"]["mean"] == pytest.approx(2.0)
    assert summary["rmse"]["std"] == pytest.approx(np.std([1.0, 3.0], ddof=1))


# ----------------- Tests for KFoldCrossValidator construction -----------------
def test_kfold_cross_validator_raises_for_n_splits_less_than_two():
    """
    Test that KFoldCrossValidator raises a ValueError at construction when n_splits is less than 2.
    """
    cfg = SEAWRDConfig.from_dict({})

    with pytest.raises(ValueError, match="n_splits"):
        KFoldCrossValidator(cfg, n_splits=1)


def synthetic_linear_dataframe(n: int = 40, random_state: int = 0) -> pd.DataFrame:
    """
    Build a small synthetic planetary-style DataFrame whose label is a simple linear function of two features.

    Parameters
    ----------
    n : int, optional
        The number of rows to generate, by default 40.
    random_state : int, optional
        Random seed for reproducibility, by default 0.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns "a", "b", "R_p", and "errcode" (all zero).
    """
    rng = np.random.default_rng(random_state)
    a = rng.uniform(0, 5, n)
    b = rng.uniform(0, 5, n)
    return pd.DataFrame({
        "a": a,
        "b": b,
        "R_p": a + b,
        "errcode": np.zeros(n, dtype=int),
    })


def quick_cv_config(save_model: bool = True, save_ensemble: bool = True, model_dir: str | None = None) -> SEAWRDConfig:
    """
    Build a small, fast SEAWRDConfig for cross-validation tests.

    Parameters
    ----------
    save_model : bool, optional
        The output.save_model value to request (KFoldCrossValidator should override this to False regardless), by
        default True.
    save_ensemble : bool, optional
        The output.save_ensemble value to request (should also be overridden to False), by default True.
    model_dir : str | None, optional
        The output.model_dir to use, by default None (falls back to the config default).

    Returns
    -------
    SEAWRDConfig
        A small, fast configuration suitable for cross-validation tests.
    """
    output = {"save_model": save_model, "save_plots": True, "save_ensemble": save_ensemble}
    if model_dir is not None:
        output["model_dir"] = model_dir

    return SEAWRDConfig.from_dict({
        "model": {"use_normalisation": False, "num_layers": 1, "num_neurons": 4},
        "training": {"num_epochs": 2, "num_models": 1, "batch_size": 8, "validation_split": 0.2},
        "output": output,
    })


@pytest.mark.tf
@pytest.mark.slow
def test_kfold_cross_validator_run_returns_metrics_per_fold():
    """
    Test that KFoldCrossValidator.run trains and evaluates once per fold, returning one metrics dict and history per
    fold with the expected regression metric keys.
    """
    preprocessor = DataPreprocessor(synthetic_linear_dataframe(), features=["a", "b"], label="R_p", normalise=False)

    cv = KFoldCrossValidator(quick_cv_config(), n_splits=3, random_state=0)
    result = cv.run(preprocessor)

    assert len(result.fold_metrics) == 3
    assert len(result.fold_histories) == 3
    for metrics in result.fold_metrics:
        assert set(metrics) == {"rmse", "mae", "r2", "mape", "bias", "max_error"}

    summary = result.summary()
    assert "rmse" in summary
    assert set(summary["rmse"]) == {"mean", "std"}


@pytest.mark.tf
@pytest.mark.slow
def test_kfold_cross_validator_never_saves_models(tmp_path):
    """
    Test that KFoldCrossValidator.run never writes model/plot/ensemble files to disk, even when the passed-in config
    requests them -- cross-validation folds are transient by design.

    Parameters
    ----------
    tmp_path : pathlib.Path
        A temporary directory provided by pytest.
    """
    preprocessor = DataPreprocessor(synthetic_linear_dataframe(), features=["a", "b"], label="R_p", normalise=False)

    cfg = quick_cv_config(save_model=True, save_ensemble=True, model_dir=str(tmp_path))
    cv = KFoldCrossValidator(cfg, n_splits=2, random_state=0)
    cv.run(preprocessor)

    assert list(tmp_path.iterdir()) == [], "No files should be written to model_dir during cross-validation"
