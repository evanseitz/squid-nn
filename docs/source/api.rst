.. _api:

API Reference
==============

Examples
--------

The script ``run_squid.py`` can be modified and run as a demo for using the
SQUID framework in conjunction with two previously-published DNNs (DeepSTARR
and ResidualBind-32) to model a genomic locus using additive and pairwise effects,
respectively. 


Generate MAVE
-------------
The ``squid.mave`` class orchestrates generation of an *in silico*
MAVE dataset. First, ``squid.mutagenizer`` is called to apply random
mutagenesis on an input sequence-of-interest to create an ensemble
of ``num_sim`` mutagenized sequences. Next, ``squid.predictor`` is called
to run inference for each sequence in the ensemble, altogether forming
a MAVE dataset containing ``num_sim`` sequence–function pairs.

.. autoclass:: squid.mutagenizer
    :members: apply_mut_by_seq_index
.. autoclass:: squid.predictor
    :members: prediction_in_batches, profile_sum, profile_pca

Mutagenesis
-----------

The ``squid.mutagenizer`` class contains functions to generate an *in silico*
dataset by randomly mutating an input sequence-of-interest.

.. autoclass:: squid.mutagenizer
    :members: apply_mut_by_seq_index


Inference
---------

The ``squid.predictor`` class...



