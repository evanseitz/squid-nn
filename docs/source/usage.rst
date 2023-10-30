Usage
=====

.. _installation:

Installation
------------

To use SQUID, first install it using pip:

.. code-block:: console

   (.venv) $ pip install squid-nn

Dependencies
----------------

    (.venv) $ pip install logomaker
    (.venv) $ pip install mavenn
    (.venv) $ pip install mavenn --upgrade


For example:

>>> from squid import utils, mave, mutagenizer, surrogate_zoo, predictor, impress
>>> import logomaker
>>> import mavenn