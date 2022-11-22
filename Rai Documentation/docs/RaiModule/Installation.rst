.. _installation:

================
**Installation**
================

**Windows 10**
--------------

We recommend using `Visual studio code <https://code.visualstudio.com>`_ for Windows users for easier installation of Python packages and required libraries. For this RAI project you need an environment with Python version 3.9.

Some packages uses Visual C++ 14.0 BuildTools. You can also install the build tools from `Microsoft Visual studio code <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_ . The build tools do not come with Visual Studio Code by default.


- **Setting up the Documentation sources**.

.. tabs::

   .. group-tab:: VS code

      https://code.visualstudio.com/download

   .. group-tab:: Python Version

      3.9.13- https://www.python.org/downloads/windows/

   .. group-tab:: Pip Version

      22.3

   .. group-tab:: Clone Git-Repo
   
      https://github.com/cisco-open/ResponsibleAI.git

   .. group-tab:: Redis

     downloaded using the .msi file at https://github.com/microsoftarchive/redis/releases/tag/win-3.2.100


 
.. note::

   ``NumPy 1.23.4`` latest version is not compatible with python 3.10 version.


**Install a package locally and run**.
--------------------------------------

Here is a quick demo of how to install a package locally and run in the environment:

- Install packages in requirement.txt file.


.. code-block:: bash

   Run pip install ``-r requirements.txt``.



.. warning:: If you run any Error.
   For instance:Package could not install ``plotly``.
   Install ``plotly`` separately with the following command 
   ``python pip install 'plotly' --user``.

- Try installing packages in requirements.txt again
- pip install -r requirements.txt.

All packages are successfully installed.

- RAI can then be installed using.

.. code-block:: bash

  pip install ``--editable``


**Description**: when you are developing it on your system any changes to the original package would reflect directly in your environment.

