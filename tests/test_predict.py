"""
Unit tests for the predict module entrypoint, which loads a saved model and runs predictions on new data.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from seawrd.cli import predict
from seawrd.config import ModelConfig
from seawrd.model import DNNManager


def test_arg_parser_has_correct_arguments():
    """
    Test that the argument parser in the predict module has the expected arguments.
    """
    parser = predict._build_argument_parser()
    args = [action.dest for action in parser._actions]

    assert "input_path" in args
    assert "model_name" in args
    assert "model_dir" in args
    assert "version" in args
    assert "output_path" in args
    assert "prediction_column" in args
    assert "ensemble" in args
    assert "uncertainty_column" in args


def test_arg_parser_requires_model_name():
    """
    Test that the argument parser requires --model-name to be supplied.
    """
    parser = predict._build_argument_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["input.dat"])


def test_arg_parser_defaults():
    """
    Test that the argument parser applies the expected defaults for optional arguments.
    """
    parser = predict._build_argument_parser()
    args = parser.parse_args(["input.dat", "--model-name", "demo"])

    assert args.model_dir == Path("models/")
    assert args.version is None
    assert args.output_path is None
    assert args.prediction_column is None
    assert args.ensemble is False
    assert args.uncertainty_column is None


@pytest.mark.tf
@pytest.mark.slow
def test_run_prediction_uses_manifest_to_align_columns(tmp_path: Path):
    """
    Test that run_prediction loads a saved model's manifest and correctly aligns a DataFrame whose columns are supplied
    out of order, producing a named prediction column.

    Parameters
    ----------
    tmp_path : pathlib.Path
        A temporary directory provided by pytest.
    """
    cfg = ModelConfig(use_normalisation=False, num_layers=1, num_neurons=4)
    manager = DNNManager.from_config(cfg, input_shape=(3,))
    manager.save_model_version(
        model=manager.model,
        history=manager.history,
        model_dir=tmp_path,
        model_name="demo",
        version=1,
        feature_names=["a", "b", "c"],
        label_name="R_p",
    )

    # Columns deliberately shuffled relative to training order
    input_df = pd.DataFrame({"c": [3.0], "a": [1.0], "b": [2.0]})

    result = predict.run_prediction(input_df, model_dir=tmp_path, model_name="demo", version=1)

    assert "R_p_pred" in result.columns, "Expected a prediction column named after the manifest's label"
    assert len(result) == 1


@pytest.mark.tf
@pytest.mark.slow
def test_run_prediction_without_manifest_uses_column_order_as_given(tmp_path: Path):
    """
    Test that run_prediction falls back to using all input columns in the order given when no manifest is present, and
    that the output column defaults to 'prediction'.

    Parameters
    ----------
    tmp_path : pathlib.Path
        A temporary directory provided by pytest.
    """
    cfg = ModelConfig(use_normalisation=False, num_layers=1, num_neurons=4)
    manager = DNNManager.from_config(cfg, input_shape=(2,))
    # Saved without feature_names, so no manifest is written
    manager.save_model_version(
        model=manager.model,
        history=manager.history,
        model_dir=tmp_path,
        model_name="demo",
        version=1,
    )

    input_df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    result = predict.run_prediction(input_df, model_dir=tmp_path, model_name="demo", version=1)

    assert "prediction" in result.columns


@pytest.mark.tf
@pytest.mark.slow
def test_main_writes_predictions_to_output_path(tmp_path: Path):
    """
    Test that the predict CLI entrypoint reads an input table, predicts using a saved model, and writes the results to
    the requested output path.

    Parameters
    ----------
    tmp_path : pathlib.Path
        A temporary directory provided by pytest.
    """
    model_dir = tmp_path / "models"
    cfg = ModelConfig(use_normalisation=False, num_layers=1, num_neurons=4)
    manager = DNNManager.from_config(cfg, input_shape=(2,))
    manager.save_model_version(
        model=manager.model,
        history=manager.history,
        model_dir=model_dir,
        model_name="demo",
        version=1,
        feature_names=["a", "b"],
        label_name="R_p",
    )

    input_path = tmp_path / "input.dat"
    pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}).to_csv(input_path, sep=" ", index=False)

    output_path = tmp_path / "output.dat"
    exit_code = predict.main([
        str(input_path),
        "--model-name", "demo",
        "--model-dir", str(model_dir),
        "--version", "1",
        "--output-path", str(output_path),
    ])

    assert exit_code == 0
    assert output_path.exists()

    result = pd.read_table(output_path, sep=r"\s+")
    assert "R_p_pred" in result.columns
    assert len(result) == 2


def _save_demo_ensemble(model_dir: Path, num_members: int = 2) -> None:
    """
    Save a small demo ensemble (mirroring what DNNTrainer.train_models writes with output.save_ensemble=True) for
    use in CLI ensemble tests.

    Parameters
    ----------
    model_dir : Path
        The directory to save the ensemble into.
    num_members : int, optional
        The number of ensemble members to save, by default 2.
    """
    from seawrd.config import ModelConfig
    from seawrd.model import DNNManager

    cfg = ModelConfig(use_normalisation=False, num_layers=1, num_neurons=4)
    feature_names = ["a", "b"]
    member_names = [f"demo_member{i}" for i in range(num_members)]

    manager = None
    for member_name in member_names:
        manager = DNNManager.from_config(cfg, input_shape=(2,))
        manager.save_model_version(
            model=manager.model, history=manager.history, model_dir=model_dir,
            model_name=member_name, version=1, feature_names=feature_names, label_name="R_p",
        )

    # The "best" model, saved under the ensemble's parent name, is what version resolution looks for
    manager.save_model_version(
        model=manager.model, history=manager.history, model_dir=model_dir,
        model_name="demo", version=1, feature_names=feature_names, label_name="R_p",
    )
    manager.save_ensemble_manifest(
        model_dir, "demo", version=1, member_names=member_names,
        feature_names=feature_names, label_name="R_p",
    )


@pytest.mark.tf
@pytest.mark.slow
def test_run_prediction_ensemble_adds_uncertainty_column(tmp_path: Path):
    """
    Test that run_prediction with ensemble=True loads all ensemble members and adds both a prediction and an
    uncertainty column.

    Parameters
    ----------
    tmp_path : pathlib.Path
        A temporary directory provided by pytest.
    """
    _save_demo_ensemble(tmp_path, num_members=3)

    input_df = pd.DataFrame({"b": [2.0], "a": [1.0]})  # deliberately shuffled
    result = predict.run_prediction(input_df, model_dir=tmp_path, model_name="demo", version=1, ensemble=True)

    assert "R_p_pred" in result.columns
    assert "R_p_std" in result.columns
    assert result["R_p_std"].iloc[0] >= 0.0


@pytest.mark.tf
@pytest.mark.slow
def test_main_ensemble_flag_writes_uncertainty_column(tmp_path: Path):
    """
    Test that the predict CLI entrypoint, given --ensemble, writes both prediction and uncertainty columns to the
    output file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        A temporary directory provided by pytest.
    """
    model_dir = tmp_path / "models"
    _save_demo_ensemble(model_dir, num_members=2)

    input_path = tmp_path / "input.dat"
    pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}).to_csv(input_path, sep=" ", index=False)

    output_path = tmp_path / "output.dat"
    exit_code = predict.main([
        str(input_path),
        "--model-name", "demo",
        "--model-dir", str(model_dir),
        "--version", "1",
        "--output-path", str(output_path),
        "--ensemble",
    ])

    assert exit_code == 0

    result = pd.read_table(output_path, sep=r"\s+")
    assert "R_p_pred" in result.columns
    assert "R_p_std" in result.columns
    assert len(result) == 2
