.. _api:

API Reference
==============

Generate MAVE
-------------
The ``squid.mave`` class orchestrates generation of an *in silico*
MAVE dataset. First, ``squid.mutagenizer`` is called to apply random
mutagenesis on an input sequence-of-interest to create an ensemble
of ``num_sim`` mutagenized sequences. Next, ``squid.predictor`` is called
to run inference for each sequence in the ensemble, altogether forming
a MAVE dataset containing ``num_sim`` sequence–function pairs.


Mutagenesis
-----------

The ``squid.mutagenizer`` class contains functions to generate an *in silico*
dataset by randomly mutating an input sequence-of-interest.


Inference
---------

The ``squid.predictor`` class... testing...

.. autoclass:: squid.impress
    :members: plot_y_hist, plot_performance, plot_additive_logo, 
        plot_pairwise_matrix, plot_y_vs_yhat, plot_y_vs_phi,
        plot_eig_vals, plot_eig_vecs
