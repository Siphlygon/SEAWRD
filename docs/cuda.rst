Enabling CUDA
=============

In order to use a GPU for training with ``tensorflow``, you need to have CUDA installed on your system. This was not trivial for our project, and here are some steps to help get CUDA working with ``tensorflow``. Note that this is not a complete guide, and you may need to adapt these steps to your specific system and configuration.


WSL2
++++
1. Install the NVIDIA driver for WSL2 on your Windows machine, **NOT YOUR WS2 LINUX!**. You can download it from the official `NVIDIA website <https://developer.nvidia.com/cuda/wsl>`_.
2. Install the CUDA toolkit for WSL2. You can follow the official NVIDIA `guide <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>`_ or find the CUDA Toolkit download and instructions `here <https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local>`_.
3. ``tensorflow`` requires cuDNN to also run. Install cuDNN for WSL2. You can download it from the official `NVIDIA website <https://developer.nvidia.com/cudnn>`_.
4. Set up environment variables for CUDA and cuDNN. Add the following lines to your `~/.bashrc` or `~/.zshrc` file: 

.. code:: bash

    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

5. Test to see if CUDA is working by running the following command in your WSL2 terminal:
.. code:: bash

    nvidia-smi

This should display information about your NVIDIA GPU and its current usage.

6. You will also likely need to download appropriate CUDA Python libraries, although note this will require 2GB+ of space and take a long while to download. You can do this by running the following command in your WSL2 terminal:
.. code:: bash
    pip install "tensorflow[and-cuda]"


7. Test if TensorFlow can access the GPU by running the following Python code:
.. code:: python
    
    import tensorflow as tf
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

If the output shows a number greater than 0, then TensorFlow is successfully using your GPU. Otherwise, and especially if `nvidia-smi` works, it is likely that you are having issues with [s]