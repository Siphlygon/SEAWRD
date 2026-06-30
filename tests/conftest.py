import pandas as pd
import pytest

@pytest.fixture
def planet_df():
    """
    Create a sample DataFrame with planet data.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing sample planet data.
    """
    return pd.DataFrame({
        "x_core'": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "x_H2O": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "T_irr": [100, 200, 300, 400, 500, 600],
        "T_b": [150, 250, 350, 450, 550, 650],
        "M_b": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "M_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "R_b": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "R_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "errcode": [0, 0, 1, 0, 0, 0],
    })
