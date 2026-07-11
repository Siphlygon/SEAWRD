Useful Commands
===============

Here are some useful commands for working with the SEAWRD project:

Testing with Pytest
-------------------
To run the tests for the SEAWRD project, you can use the following command:

.. code:: bash

    pytest --cov=seawrd

You can also increase the verbosity (i.e., see more information) by adding the `-v` flag; see prints/outputs in the code by adding the `-s` flag; or see what statements are missing coverage by adding the `--cov-report term-missing` flag. You can combine these flags as needed.

Building Documentation with Sphinx
----------------------------------
In order to build the documentation for the SEAWRD project, you can use the following command:

.. code:: bash

    python -m sphinx -M html docs docs/_build


Building SEAWRD with Twine
--------------------------
To build the SEAWRD project and upload it to PyPI using Twine, you can use the following commands:

.. code:: bash

    pip install twine build
    python -m build
    python -m twine upload dist/*