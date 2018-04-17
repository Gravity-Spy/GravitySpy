.. GravitySpy documentation master file, created by
   sphinx-quickstart on Thu Apr 21 14:05:08 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GravitySpy's documentation!
======================================

`Gravity Spy <https://gravityspy.org>`_ is an innovative citizen-science meets Machine Learning meets gravitational wave physics project. This repository is meant to faciliate the creation of new similar citizen science projects on `Zooniverse <https://zooniverse.org>`_
For full details see `Zevin et al. (Classical and Quantum Gravity, 2016) <https://iopscience.iop.org/article/10.1088/1361-6382/aa5cea>`_

Installing GravitySpy
---------------------

The easiest method to install gravityspy is using `pip <https://pip.pypa.io/en/stable/>`_ directly from the `GitHub repository <https://github.com/Gravity-Spy/GravitySpy.git>`_:

.. code-block:: bash

   $ pip install git+https://github.com/Gravity-Spy/GravitySpy.git

How to run GravitySpy
---------------------

The main product of this package is the command-line executable `wscan`, which takes an excess noise time makes an omega scan of the event and classifies the image.

To run an analysis:

.. code-block:: bash

   $ wscan --inifile my-wini-config-file.ini --eventTime <eventTime> --outDir ./public_html/GravitySpy/Test/ --uniqueID --ID 123abc1234 --HDF5 --runML --pathToModel ../bin

where ``<eventtime>`` is the GPS time stamp assosciated with an Omicron trigger or other time of known excess noise, and ``./my-wini-config-file.ini`` is the path of your configuration file. In the folder Production, is an example of an ini file.

For a full list of command-line argument and options, run

.. code-block:: bash

   $ wscan --help

For more details see :ref:`command-line`.

The many databases of GravitySpy
--------------------------------

For more details see :ref:`DBs`

Package documentation
---------------------

Please consult these pages for more details on using GravitySpy:

.. toctree::
   :maxdepth: 1

   command-line/index
   examples/index
   DBs/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
