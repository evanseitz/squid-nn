.. _installation:

Installation Instructions
=========================

SQUID
-----

To install SQUID, use the ``pip`` package manager via the command line: ::

    $ pip install squid-nn

Alternatively, you can clone SQUID from
`GitHub <https://github.com/evanseitz/squid-nn>`_ 
using the command line: ::

    $ cd appropriate_directory
    $ git clone https://github.com/evanseitz/squid-nn.git

where ``appropriate_directory`` is the absolute path to where you would like
SQUID to reside. Then add the following to the top of any Python file in
which you use SQUID: ::

    # Insert local path to SQUID at beginning of Python's path
    import sys
    sys.path.insert(0, 'appropriate_directory/squid')

    #Load squid
    import squid


MAVE-NN
-------

MAVE-NN models and related visualization tools require additional dependencies: ::

    $ pip install logomaker 
    $ pip install mavenn
    $ pip install mavenn --upgrade

Please see the `MAVENN <https://mavenn.readthedocs.io>`_ documentation for more information.

For older DNNs that require inference via Tensorflow 1.x, Python 2.x is required which is not supported by MAVE-NN. 
Users will need to create separate environments in this case:

    1.  Tensorflow 1.x and Python 2.x environment for generating *in silico* MAVE data
    2.  Tensorflow 2.x and Python 3.x environment for training MAVE-NN models