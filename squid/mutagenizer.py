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
    torch.Tensor
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

        if i == 0: #keep wild-type sequence
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
    return one_hot