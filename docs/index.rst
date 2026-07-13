SEAWRD
======

Welcome to ``SEAWRD``'s documentation! This project is a Python package for training and evaluating surrogate models, primarily for simulating expensive exoplanet simulations. It provides tools for preprocessing data, training surrogate models, and evaluating their performance. The package is designed to be user-friendly and flexible, allowing researchers to easily integrate it into their workflows.

``SEAWRD`` was created during `Code/Astro <https://semaphorep.github.io/codeastro/>`_ 2026, and is still under active development. We welcome contributions from the community, and encourage users to report issues and submit pull requests on our `GitHub repository <https://github.com/Siphlygon/SEAWRD>`_. For example usage, please view the example usage Jupyter notebook `seawrd.ipynb`.


Accreditation
+++++++++++++
Please cite the `DOI <https://doi.org/10.5281/zenodo.20869608>`_ if you make use of this software in your research.

Acknowledgements
++++++++++++++++
The data source file "DNN_data_IOP_Aguichine2021.dat" used in the example usage Jupyter notebook is from Aguichine et al. (2021), and you are encouraged to read their paper found here:

A. Aguichine, O. Mousis, M. Deleuil, and E. Marcq, “Mass–Radius relationships for irradiated ocean planets,” The Astrophysical Journal, vol. 914, no. 2, p. 84, Jun. 2021, doi: `10.3847/1538-4357/abfa99 <https://doi.org/10.3847/1538-4357/abfa99>`_.


User Guides
+++++++++++
.. toctree::
   :maxdepth: 2

   installation
   cuda
   useful_commands
   usage_scripts


Public API
++++++++++

.. toctree::
   :maxdepth: 2

   config
   config_manager
   preprocessing_data
   model
   trainer
   cross_validation
   evaluation
   device_selection
   device_benchmark_worker
   bootstrap
   utils



Indices and tables
++++++++++++++++++

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

