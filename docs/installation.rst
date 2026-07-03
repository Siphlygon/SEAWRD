Usage Guide
===========

Here is how you can use this project.

Installation
------------
Run ``pip install seawrd`` to get started.


Usage
-----
To use the SEAWRD package, first have a data file or use the example data file `DNN_data_IOP_Aguichine2021.dat`. Create a npz file from the data file using the ``DatasetPreprocessor`` class from `preprocessing_data.py` using the following command:

.. code-block:: bash

    python -m seawrd.preprocessing_data DNN_data_IOP_Aguichine2021.dat --output_file data.npz

Then consider changing the default configuration file `seawrd_default.toml` to your needs, or provide a custom configuration file. Finally, run the training script with the following command:

.. code-block:: bash

    python -m seawrd.train data.npz

or with a custom configuration file:

.. code-block:: bash

    python -m seawrd.train data.npz --config custom_config.toml
