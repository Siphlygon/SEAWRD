import os
os.environ["KERAS_BACKEND"] = "tensorflow"


def validate_keras_field(field_name: str, field_type: str):
    """
    Validate that Keras can resolve a given field identifier.
    
    Keras provides a mechanism to retrieve various components (like losses, metrics, and activations) using string identifiers. This function checks if the provided field identifier can be resolved by Keras for the specified field type.

    Parameters
    ----------
    field_name : str
        The name of the field to validate (e.g., 'loss', 'metric', 'activation').
    field_type : str
        The name of the Keras module to use for validation (e.g., losses, metrics, activations).
    
    Raises
    ------    
    ValueError
        If the field identifier or type is unknown or unsupported.
    """
    import keras  # use a local import to avoid unnecessary dependencies when importing the package

    match field_type:
        case "losses":
            keras_module = keras.losses
        case "metrics":
            keras_module = keras.metrics
        case "activations":
            keras_module = keras.activations
        case _:
            raise ValueError(f"Unknown or unsupported Keras field type: {field_type!r}")

    try:
        keras_module.get(field_name)
    except Exception as exc:
        raise ValueError(f"Unknown or unsupported Keras {field_type}: {field_name!r}") from exc

