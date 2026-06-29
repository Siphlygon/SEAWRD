"""
A set of helper functions for SEAWRD tests.
"""
from typing import Any

import keras
import numpy as np


# ----------------- Tests for Kerass objects -----------------
def _normalize_config(obj : Any) -> Any:
    """
    Convert tuples to lists recursively so Keras configs compare stably.
    
    While the default Keras config for input_shape specifies a tuple, under serialisation and deserialisation, this may
    be converted to a list. This does not change the functionality of the model but will cause an assertion of the two
    configs to fail. This function recursively converts tuples to lists so that the configs can be compared stably.

    Parameters
    ----------
    obj : Any
        The object to normalize. This can be a tuple, list, dict, or any other type.

    Returns
    -------
    Any
        The normalized object.
    """
    if isinstance(obj, tuple):
        return [_normalize_config(x) for x in obj]  # convert tuple to list and normalize elements
    if isinstance(obj, list):
        return [_normalize_config(x) for x in obj]  # normalize elements of the list
    if isinstance(obj, dict):
        return {k: _normalize_config(v) for k, v in obj.items()}
    return obj


def assert_same_keras_model(model_a : keras.Model,
                            model_b : keras.Model,
                            same_weights : bool = True):
    """
    Assert that two Keras models have the same architecture and (optionally) weights.

    Parameters
    ----------
    model_a : keras.Model
        The first Keras model to compare.
    model_b : keras.Model
        The second Keras model to compare.
    same_weights : bool, optional
        Whether to perform a comparison of weights, by default True
    test_normalisation : bool, optional
        Whether to perform a comparison of normalisation layer configurations, by default False
    """
    # Assert that the two models have the exact same architecture (layers)
    config_a = _normalize_config(model_a.get_config())
    config_b = _normalize_config(model_b.get_config())
    assert config_a == config_b, f"Model architectures differ:\n{config_a}\n{config_b}"

    # Assert that the two models have the same number of weight arrays
    weights_a = model_a.get_weights()
    weights_b = model_b.get_weights()
    assert len(weights_a) == len(weights_b), f"Number of weight arrays differ: {len(weights_a)} != {len(weights_b)}"

    # Assert that the two models have the same weight values
    for i, (wa, wb) in enumerate(zip(weights_a, weights_b)):
        assert wa.shape == wb.shape, f"Weight {i} shape differs: {wa.shape} != {wb.shape}"

        if same_weights:  # If exact comparison is required, check for exact equality
            assert np.array_equal(wa, wb), f"Weight {i} values differ"


def assert_same_keras_normalisation(model_a : keras.Model,
                                    model_b : keras.Model):
    """
    Assert that two Keras models have the same normalisation layer configuration.
    
    This is used in clone_model where we do not care what weights any other hidden layers have, but require two models
    to have the same normalisation layer configuration and weights.
    
    Parameters
    ----------
    model_a : keras.Model
        The first Keras model to compare.
    model_b : keras.Model
        The second Keras model to compare.
    """
    # Extract the normalisation layers from both models
    norm_a = next((layer for layer in model_a.layers if isinstance(layer, keras.layers.Normalization)), None)
    norm_b = next((layer for layer in model_b.layers if isinstance(layer, keras.layers.Normalization)), None)

    assert norm_a is not None, "Model A does not have a normalisation layer"
    assert norm_b is not None, "Model B does not have a normalisation layer"

    for original_weight, cloned_weight in zip(norm_a.get_weights(), norm_b.get_weights()):
        np.testing.assert_allclose(original_weight, cloned_weight)


def assert_same_history(history_a : keras.callbacks.History,
                        history_b : keras.callbacks.History,
                        rtol : float = 1e-7,
                        atol : float = 1e-8):
    """
    Assert that two Keras History objects have the same keys and values within a specified tolerance.

    Parameters
    ----------
    history_a : keras.callbacks.History
        The first Keras History object to compare.
    history_b : keras.callbacks.History
        The second Keras History object to compare.
    rtol : float, optional
        The relative tolerance for comparison, by default 1e-7
    atol : float, optional
        The absolute tolerance for comparison, by default 1e-8
    """
    # Extract the history dictionaries from the History objects
    hist_a = history_a.history
    hist_b = history_b.history

    assert hist_a.keys() == hist_b.keys(), f"History keys differ: {hist_a.keys()} != {hist_b.keys()}"

    for key in hist_a:
        values_a = np.asarray(hist_a[key])
        values_b = np.asarray(hist_b[key])

        assert values_a.shape == values_b.shape, (
            f"History entry {key!r} has different shape: "
            f"{values_a.shape} != {values_b.shape}"
        )

        np.testing.assert_allclose(
            values_a,
            values_b,
            rtol=rtol,
            atol=atol,
            err_msg=f"History entry {key!r} differs",
        )
