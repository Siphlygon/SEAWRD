Useful Commands
===============

Here are some useful commands for working with the SEAWRD project:

Testing
-------
To run the tests for the SEAWRD project, you can use the following command:

.. code:: bash

    pytest --cov=seawrd

You can also increase the verbosity (i.e., see more information) by adding the `-v` flag; see prints/outputs in the code by adding the `-s` flag; or see what statements are missing coverage by adding the `--cov-report term-missing` flag. You can combine these flags as needed.

Sphinx
------
In order to build the documentation for the SEAWRD project, you can use the following command:

.. code:: bash

    python -m sphinx -M html docs docs/_build
