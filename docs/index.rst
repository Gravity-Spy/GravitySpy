.. GravitySpy documentation master file, created by
   sphinx-quickstart on Thu Apr 21 14:05:08 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GravitySpy's documentation!
=================================

`Gravity Spy <https://gravityspy.org>`_ is an innovative citizen-science meets Machine Learning meets gravitational wave physics project. This repository is meant to faciliate the creation of new similar citizen science projects on `Zooniverse <https://zooniverse.org>`_
For full details see `Zevin et al. (Classical and Quantum Gravity, 2016) <https://iopscience.iop.org/article/10.1088/1361-6382/aa5cea>`_

Installing GravitySpy
----------------

The easiest method to install pyomega is using `pip <https://pip.pypa.io/en/stable/>`_ directly from the `GitHub repository <https://github.com/Gravity-Spy/GravitySpy.git>`_:

.. code-block:: bash

   $ pip install git+https://github.com/Gravity-Spy/GravitySpy.git

How to run GravitySpy
----------------

The main product of this package is the command-line executable `pyomega`, which runs an end-to-end search for statistical coincidences, and produces a list of viable data-quality flags that can be used as vetoes in a search, as well as an HTML summary.

To run an analysis:

.. code-block:: bash

   $ pyomega <gpsstart> <gpsend> --config-file ./my-config-file.ini

where ``<gpsstart>`` and ``<gpsend>`` are the start and stop GPS times of the analysis period, and ``./my-config-file.ini`` is the path of your configuration file. Strictly speaking the configuration file is optional, but is highly recommend if you want to have any control over your analysis.

For a full list of command-line argument and options, run

.. code-block:: bash

   $ pyomega --help

For more details see :ref:`command-line`.

Package documentation
---------------------

Please consult these pages for more details on using GravitySpy:

.. toctree::
   :maxdepth: 1

   command-line/index
   api/pyomega.config

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
