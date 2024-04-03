import os
#os.environ["TQDM_DISABLE"] = "1"
import numpy as np
from tqdm import tqdm


class BaseMutagenesis:
    """
    Base class for in silico MAVE data generation for a given sequence.
    """
    def __call__(self, x, num_sim):
        """Return an in silico MAVE based on mutagenesis of 'x'.

        Parameters
        ----------
        x : torch.Tensor
            one-hot sequence (shape: (L, A)).
        num_sim : int
            Number of sequences to mutagenize.

        Returns
        -------
        torch.Tensor
            Batch of one-hot sequences with random augmentation applied.
        """
        raise NotImplementedError()


class RandomMutagenesis(BaseMutagenesis):
    """Module for performing random mutagenesis.

    Parameters
    ----------
    mut_rate : float, optional
        Mutation rate for random mutagenesis (defaults to 0.1).
    uniform : bool
        uniform (True), Poisson (False); sets the number of mutations per sequence.

    Returns
    -------
    numpy.ndarray
        Batch of one-hot sequences with random mutagenesis applied.
    """

    def __init__(self, mut_rate, uniform=False):
        self.mut_rate = mut_rate
        self.uniform = uniform

    def __call__(self, x, num_sim):

        L, A = x.shape
        avg_num_mut = int(np.ceil(self.mut_rate*L))

        # get indices of nucleotides
        x_index = np.argmax(x, axis=1)

        # sample number of mutations for each sequence
        if self.uniform:
            num_muts = int(avg_num_mut*np.ones((num_sim,), dtype=int))
        else:
            num_muts = np.random.poisson(avg_num_mut, (num_sim, 1))[:,0]
        num_muts = np.clip(num_muts, 0, L)
        one_hot = apply_mut_by_seq_index(x_index, (num_sim,L,A), num_muts)
        return one_hot
    

class CombinatorialMutagenesis():
    """Module for performing combinatorial mutagenesis.

    Returns
    ----------
    numpy.ndarray
        Batch of one-hot sequences with combinatorial mutagenesis applied,
        such that the number of sequences produced is the number of characters A
        in the alphabet raised to the length L of the 'mut_window'.
    """

    def __call__(self, x, num_sim): # 'num_sim' will be replaced by A**L

        L, A = x.shape

        def seq2oh(seq, alphabet=['A','C','G','T']):   
            L = len(seq)
            one_hot = np.zeros(shape=(L,len(alphabet)), dtype=np.float32)
            for idx, i in enumerate(seq):
                for jdx, j in enumerate(alphabet):
                    if i == j:
                        one_hot[idx,jdx] = 1
            return one_hot

        from itertools import product
        seqs = list(product(list(range(A)), repeat=L))
        one_hot = np.zeros(shape=(int(A**L), L, A))
        for i in tqdm(range(int(A**L)), desc="Mutagenesis"):
            one_hot[i,:,:] = seq2oh(seqs[i], alphabet=list(range(A)))

        return one_hot
    

class TwoHotMutagenesis(BaseMutagenesis):
    """Module to perform random mutagenesis using two-hot encoding.
    That is, encode each individual nucleotide at a given position
    using a one-hot encoding scheme, then represent the unphased
    diploid sequence as the sum of the two one-hot encoded nucleotides
    at each position. The sequence "AYCR", for example, would be encoded as:
    [[2, 0, 0, 0], [0, 1, 0, 1], [0, 2, 0, 0], [1, 0, 1, 0]].

    Returns
    ----------
    numpy.ndarray
        Batch of one-hot sequences with random mutagenesis applied, with alphabet:
        {A, C, G, T, R (A/G), Y (C/T), S (C/G), W (A/T), K (G/T), M (A/C)}, such that
        heterozygous positions are represented using the IUPAC ambiguity codes.
    """
    
    def __init__(self, mut_rate, uniform=False):
        self.mut_rate = mut_rate
        self.uniform = uniform

    def __call__(self, x, num_sim):
        from numpy.random import choice
        from numpy.random import poisson

        def swap_elements(x, t):
            """Per iteration, use numpy.random.choice to randomly select elements where
            replacements will occur in the original list. Then zip those indices
            against the values used for the substitution and apply the replacements.
            """
            new_x = x[:]
            for idx, value in zip(choice(range(len(x)), size=len(t), replace=False), t):
                new_x[idx] = value
            return new_x

        L, A = x.shape # ensure A=4 for this module
        alphabet_pool = ['A', 'C', 'G', 'T', 'R', 'Y', 'S', 'W', 'K', 'M'] # pool for selecting characters
        seq = twohot2seq(x)
        seq = [*seq]

        # set up number of mutations to sample for each sequence
        avg_num_mut = int(np.ceil(self.mut_rate*L))
        if self.uniform:
            num_muts = int(avg_num_mut*np.ones((num_sim,), dtype=int)) + 1
        else:
            num_muts = poisson(avg_num_mut, (num_sim, 1))[:,0] + 1

        # mutagenize each sequence based on number of mutations; i.e., samples from alphabet pool
        one_hot = np.zeros(shape=(num_sim, L, A))
        for i, num_mut in enumerate(tqdm(num_muts, desc="Mutagenesis")):
            if i == 0:
                one_hot[i,:,:] = seq2twohot(''.join(seq))
            else:
                options_list = choice(alphabet_pool, size=num_mut, replace=True) # sample 'num_mut' characters from alphabet_pool with replacement
                mut_seq = ''.join(swap_elements(seq, options_list))
                one_hot[i,:,:] = seq2twohot(mut_seq)

        return one_hot


"""
class CustomMutagenesis(BaseMutagenesis):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def __call__(self, x, num_sim):

        # code goes here
        return one_hot
"""


################################################################################
# useful functions
################################################################################


def apply_mut_by_seq_index(x_index, shape, num_muts):
    """Function to perform random mutagenesis.

    Parameters
    ----------
    x_index : np.ndarray
        Indices of wildtype sequence.
    shape : list
        Shape of MAVE array; i.e., (num_sim,L,A).
    num_muts : int
        Number of mutations per sequence.

    Returns
    -------
    torch.Tensor
        Batch of one-hot sequences with random mutagenesis applied.
    """

    num_sim, L, A = shape
    one_hot = np.zeros((num_sim, L, A))

    # loop through and generate random mutagenesis
    for i, num_mut in enumerate(tqdm(num_muts, desc="Mutagenesis")):

        if i == 0: # keep wild-type sequence
            one_hot[i,:,:] = np.eye(A)[x_index]
        else:
            # generate mutation index
            mut_index = np.random.choice(range(0, L), num_mut, replace=False)

            # sample alphabet
            mut = np.random.choice(range(1, A), (len(mut_index)))

            # loop through sequence and add mutation index (note: up to 3 is added which does not map to [0,3] alphabet)
            seq_index = np.copy(x_index)
            for j, m in zip(mut_index, mut):
                seq_index[j] += m

            # wrap non-sensical indices back to alphabet -- effectively makes it random mutation
            seq_index = np.mod(seq_index, A)

            # create one-hot from index
            one_hot[i,:,:] = np.eye(A)[seq_index]
    return one_hot.astype('uint8')


def twohot2seq(one_hot):
    """Function to convert two-hot encoding to a DNA sequence.

    Parameters
    ----------
    one_hot : numpy.ndarray
        Input one-hot encoding of sequence (shape : (L,C))

    Returns
    -------
    seq : string
        Input sequence with length L.
    """
    seq = []
    for i in range(one_hot.shape[0]):

        if np.array_equal(one_hot[i,:], np.array([2, 0, 0, 0])):
            seq.append('A')
        elif np.array_equal(one_hot[i,:], np.array([0, 2, 0, 0])):
            seq.append('C')
        elif np.array_equal(one_hot[i,:], np.array([0, 0, 2, 0])):
            seq.append('G')
        elif np.array_equal(one_hot[i,:], np.array([0, 0, 0, 2])):
            seq.append('T')
        elif np.array_equal(one_hot[i,:], np.array([0, 0, 0, 0])):
            seq.append('N')
        elif np.array_equal(one_hot[i,:], np.array([1, 1, 0, 0])):
            seq.append('M')
        elif np.array_equal(one_hot[i,:], np.array([1, 0, 1, 0])):
            seq.append('R')
        elif np.array_equal(one_hot[i,:],np.array([1, 0, 0, 1])):
            seq.append('W')
        elif np.array_equal(one_hot[i,:], np.array([0, 1, 1, 0])):
            seq.append('S')
        elif np.array_equal(one_hot[i,:], np.array([0, 1, 0, 1])):
            seq.append('Y')
        elif np.array_equal(one_hot[i,:], np.array([0, 0, 1, 1])):
            seq.append('K')
    seq = ''.join(seq)
    return seq


def seq2twohot(seq):
    """Function to convert heterozygous DNA sequence to two-hot encoding.

    Parameters
    ----------
    seq : string
        Input sequence with length L.

    Returns
    -------
    one_hot : numpy.ndarray
        Input one-hot encoding of sequence (shape : (L,C))
    """
    seq_list = list(seq.upper()) # get sequence into an array
    # one hot the sequence
    encoding = {
        "A": np.array([2, 0, 0, 0]),
        "C": np.array([0, 2, 0, 0]),
        "G": np.array([0, 0, 2, 0]),
        "T": np.array([0, 0, 0, 2]),
        "N": np.array([0, 0, 0, 0]),
        "M": np.array([1, 1, 0, 0]),
        "R": np.array([1, 0, 1, 0]),
        "W": np.array([1, 0, 0, 1]),
        "S": np.array([0, 1, 1, 0]),
        "Y": np.array([0, 1, 0, 1]),
        "K": np.array([0, 0, 1, 1]),
    }
    one_hot = [encoding.get(seq, seq) for seq in seq_list]
    one_hot = np.array(one_hot)
    return one_hot