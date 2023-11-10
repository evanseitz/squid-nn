import pandas as pd
import numpy as np


def arr2pd(x, alphabet=['A','C','G','T']):
    """Function to convert a Numpy array to Pandas dataframe with proper column headings.

    Parameters
    ----------
    x : numpy.ndarray
        One-hot encoding or attribution map (shape : (L,C)).
    alphabet : list
        The alphabet used to determine the C characters in the logo such that
        each entry is a string; e.g., ['A','C','G','T'] for DNA.

    Returns
    -------
    x : pandas.dataframe
        Dataframe corresponding to the input array.
    """
    labels = {}
    idx = 0
    for i in alphabet:
        labels[i] = x[:,idx]
        idx += 1
    x = pd.DataFrame.from_dict(labels, orient='index').T
    
    return x


def oh2seq(one_hot, alphabet=['A','C','G','T']):
    """Function to convert one-hot encoding to a sequence.

    Parameters
    ----------
    one_hot : numpy.ndarray
        Input one-hot encoding of sequence (shape : (L,C))
    alphabet : list
        The alphabet used to determine the C characters in the logo such that
        each entry is a string; e.g., ['A','C','G','T'] for DNA.

    Returns
    -------
    seq : string
        Input sequence with length L.
    """
    seq = []
    for i in range(np.shape(one_hot)[0]):
        for j in range(len(alphabet)):
            if one_hot[i][j] == 1:
                seq.append(alphabet[j])
    seq = ''.join(seq)
    return seq


def seq2oh(seq, alphabet=['A','C','G','T']):
    """Function to convert a sequence to one-hot encoding.

    Parameters
    ----------
    seq : string
        Input sequence with length L
    alphabet : list
        The alphabet used to determine the C characters in the logo such that
        each entry is a string; e.g., ['A','C','G','T'] for DNA.

    Returns
    -------
    one_hot : numpy.ndarray
        One-hot encoding corresponding to input sequence (shape : (L,C)).
    """
    L = len(seq)
    one_hot = np.zeros(shape=(L,len(alphabet)), dtype=np.float32)
    for idx, i in enumerate(seq):
        for jdx, j in enumerate(alphabet):
            if i == j:
                one_hot[idx,jdx] = 1
    return one_hot


def fix_gauge(x, gauge, wt=None, r=0.1):
    """Function to fix the gauge for an attribution matrix.

    Parameters
    ----------
    x : numpy.ndarray
        Attribution scores for a sequence-of-interest (shape : (L,C)).
    gauge : gauge mode used to fix model parameters.
        See https://mavenn.readthedocs.io/en/latest/math.html for more info.
        'uniform'   :   hierarchical gauge using a uniform sequence distribution over
                        the characters at each position observed in the training set
                        (unobserved characters are assigned probability 0).
        'empirical' :   uses an empirical distribution computed from the training data.
        'consensus' :   wild-type gauge using the training data consensus sequence.
        'default'   :   default gauge (no change).
    OH_wt : numpy.ndarray
        Wild-type sequence (one-hot encoding) for 'wildtype' or 'empirical' gauge (shape : (L,C)).
    r : float
        For 'empirical gauge', the probability of mutation used during generation of
        in silico MAVE dataset (should match user-defined 'mut_rate').

    Returns
    -------
    OH : numpy.ndarray
        Gauge-fixed one-hot encoding corresponding to input sequence (shape : (L,C)).
    """
    x1 = x.copy()

    if gauge == 'empirical':
        L = wt.shape[0] #length of sequence
        wt_argmax = np.argmax(wt, axis=1) #index of each wild-type in the one-hot encoding

        p_lc = np.ones(shape=wt.shape) #empirical probability matrix
        p_lc = p_lc*(r/3.)

        for l in range(L):
            p_lc[l,wt_argmax[l]] = (1-r)

        for l in range(L):
            weighted_avg = np.average(x[l,:], weights=p_lc[l,:])
            for c in range(4):
                x1[l,c] -= weighted_avg

    elif gauge == 'wildtype':
        L = wt.shape[0]
        wt_argmax = np.argmax(wt, axis=1)
        for l in range(L):
            wt_val = x[l, wt_argmax[l]]
            x1[l,:] -= wt_val

    elif gauge == 'hierarchical':
        for l in range(x.shape[0]):
            col_mean = np.mean(x[l,:])
            x1[l,:] -= col_mean

    elif gauge == 'default':
        pass

    return x1


    