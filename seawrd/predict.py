"""
Prediction entrypoint for SEAWRD.

Loads a trained model (and its manifest, if present) and runs it on a table of new planetary data via
seawrd.predictor.Predictor. This is the inference counterpart to seawrd.train.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import pandas as pd

os.environ["KERAS_BACKEND"] = "tensorflow"

from .utils import get_logger

logger = get_logger("seawrd.predict")


def run_prediction(input_df: pd.DataFrame,
                   model_dir: Path | str,
                   model_name: str,
                   version: int | None = None,
                   prediction_column: str | None = None) -> pd.DataFrame:
    """
    Load a saved model and run predictions on a DataFrame of input features.

    Parameters
    ----------
    input_df : pd.DataFrame
        The input features to predict on. If the saved model has a manifest, columns are selected and reordered to
        match the training feature order; otherwise all columns are used as-is, in the order given.
    model_dir : Path | str
        The directory containing the saved model.
    model_name : str
        The name of the saved model to load.
    version : int | None, optional
        The version of the model to load. If None, the latest available version is loaded. By default None.
    prediction_column : str | None, optional
        The name of the column to store predictions in. Defaults to ``"{label_name}_pred"`` when the model's manifest
        records a label name, otherwise ``"prediction"``. By default None.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with a prediction column appended.
    """
    # Only now do we import keras by importing predictor
    from .predictor import Predictor

    predictor = Predictor.from_saved(model_dir, model_name, version)

    if predictor.feature_names is None:
        logger.warning(
            "No manifest found for model '%s'; using all input columns in the order given. "
            "Predictions will be unreliable if this does not match the training feature order.",
            model_name,
        )

    return predictor.predict_dataframe(input_df, prediction_column=prediction_column)


def _build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser for the SEAWRD prediction script.

    Returns
    -------
    argparse.ArgumentParser
        The CLI argument parser for the SEAWRD prediction script.
    """
    parser = argparse.ArgumentParser(description="Run predictions with a trained SEAWRD model.")
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a whitespace-delimited table of input features to predict on.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the saved model to load.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/"),
        help="Directory containing the saved model (default: models/).",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Model version to load. If omitted, the latest available version is loaded.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path to write predictions to as a whitespace-delimited table. If omitted, predictions are printed to "
             "stdout.",
    )
    parser.add_argument(
        "--prediction-column",
        type=str,
        default=None,
        help="Name of the output prediction column (default: '{label}_pred' if known, otherwise 'prediction').",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    CLI entrypoint for the SEAWRD prediction script. This function handles command-line arguments, loads the input
    data and saved model, runs predictions, and writes or prints the results.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        The command-line arguments, by default None
    """
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    input_df = pd.read_table(args.input_path, sep=r"\s+")
    logger.info("Loaded %d rows from %s.", len(input_df), args.input_path)

    result = run_prediction(
        input_df=input_df,
        model_dir=args.model_dir,
        model_name=args.model_name,
        version=args.version,
        prediction_column=args.prediction_column,
    )

    if args.output_path is not None:
        result.to_csv(args.output_path, sep=" ", index=False)
        logger.info("Wrote predictions to %s.", args.output_path)
    else:
        print(result.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
