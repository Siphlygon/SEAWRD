"""
A very simple smoke test to ensure that the package and its submodules can be imported without errors.
"""

def test_package_imports():
    """
    Test that the seawrd package and its submodules can be imported without errors.
    """
    import seawrd
    import seawrd.config
    import seawrd.config_manager
    import seawrd.model
    import seawrd.preprocessing_data
    import seawrd.trainer
