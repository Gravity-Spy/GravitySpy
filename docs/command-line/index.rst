.. _command-line:

#####################################
Running ``hveto`` on the command line
#####################################

As described on the home page, the main interface to the hveto algorithm is the command-line executable ``hveto``.

Basic instructions
==================

Basic instructions on running ``hveto`` can be gained by looking at the ``--help`` menu:

.. command-output:: hveto --help

Detailed instructions
=====================

A few of the command-line options require special formatting, use the reference below for more detailed info.

``gpsstart`` and ``gpsend``
---------------------------

Each of the required GPS start and stop times can be given as GPS integers, or as date strings, e.g.

.. code-block:: bash

   hveto 1135641617 1135728017 ...

will produce the same analysis as

.. code-block:: bash

   hveto "Jan 1 2016" "Jan 2 2016" ...

.. note::

   The quote marks used in the date string example are important so that the shell passes them to ``python`` as a single argument

   

``-j/--nproc``
--------------

The majority of the processing time for ``hveto`` is taken up by two things

- reading event files for the auxiliary channels
- determining the most-significant channel for a given round

Each of these can be sped up by using multiple CPUs available on the host machine by supplying ``--nproc X`` where X is any integer.

.. warning::

   No restrictions are placed on the number of parallel processes that can be given on the command line, users should be aware that using too-high a number could overload the machine, or otherwise cause problems for themselves and other users.

   You can check how many processors are available on the machine with the following unix command

   .. code-block:: bash

      cat /proc/cpuinfo | grep processor | wc -l

``-p/--primary-cache``
----------------------

This should receive a LAL-format cache file (see :mod:`glue.lal` for details)

``-a/--auxiliary-cache``
------------------------

This should receive a LAL-format cache file (similarly to ``--primary-cache``), but with a specific format:

- each contained trigger file should contain events for only a single channel
- the contained files should map to their channel as follows; if a channel is named ``X1:SYS-SUBSYS_NAME``, each filename should start ``X1-SYS_SUBSYS_NAME`` according to the `T050017 <https://dcc.ligo.org/LIGO-T050017>`_ convention. Additional underscore-delimited components can be given, but will not be used to map to the channel.

.. note::

   If the ``channels`` option is not given in the ``[auxiliary]`` section of the configuration, it will be populated from the unique list of channels parsed as above from the ``--auxiliary-cache``
