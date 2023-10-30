import numpy as np


class InSilicoMAVE():
    """Module for performing in silico MAVE.

    Parameters
    ----------
    mut_generator : class
        Module for performing random mutagenesis.
    pred_generator : class
        Module for inferring model predictions.
    seq_length : int
        Full length L of input sequence.
    mut_window : [int, int]
        Index of start and stop position along sequence to probe;
        i.e., [start, stop], where start < stop and both entries
        satisfy 0 <= int <= L.
    """
    def __init__(self, mut_generator, mut_predictor, seq_length, mut_window=None):
        self.mut_generator = mut_generator
        self.mut_predictor = mut_predictor
        self.seq_length = seq_length
        self.mut_window = mut_window
        if mut_window is not None:
            self.start_position = mut_window[0]
            self.stop_position = mut_window[1]
        else:
            self.start_position = 0
            self.stop_position = seq_length


    def generate(self, x, num_sim, seed=None, verbose=1):
        """Randomly mutate segments in a set of one-hot DNA sequences.

        Parameters
        ----------
        x : torch.Tensor
            One-hot sequence (shape: (L,A)).
        num_sim : int
            Number of sequences to mutagenize for in silico MAVE.
        seed : int, optional
            Sets the random number seed.

        Returns
        -------
        x_mut : numpy.ndarray
            Sequences simulated by mut_predictor.
        y_mut : numpy.ndarray
            Inferred predictions for sequences (shape: (N,1)).
        """
        if seed:
            np.random.seed(seed)
        if verbose:
            print('')
            print('Building in silico MAVE...')

        # generate in silico MAVE based on mutagenesis strategy
        if self.mut_window is not None:
            x_window = self.delimit_range(x, self.start_position, self.stop_position)
            x_mut = self.mut_generator(x_window, num_sim)
            x_mut_full = self.pad_seq(x_mut, x, self.start_position, self.stop_position)
            y_mut = self.mut_predictor(x_mut_full)
        else:
            x_mut = self.mut_generator(x, num_sim)
            y_mut = self.mut_predictor(x_mut)

        return x_mut, y_mut


    def pad_seq(self, x_mut, x, start_position, stop_position):
        """Function to pad mutated sequences on both sides with the surrounding unmutated region.
        
        Parameters
        ----------
        x_mut : numpy.ndarray
            Sequences with randomly mutated segments with length l < L
            defined by l = stop_position - start_position (shape: (N,l,C)).
        x : torch.Tensor
            Batch of one-hot sequences (shape: (L,A)).
        start_position : int
            Index of start position along sequence to probe.
        stop_position : int
            Index of stop position along sequence to probe.

        Returns
        -------
        numpy.ndarray
            Sequences with randomly mutated segments, padded to correct shape
            with random DNA (shape: (N,L,C)).
        """
        N = x_mut.shape[0]
        x = x[np.newaxis,:]
        x_start = np.tile(x[:,:start_position,:], (N,1,1))
        x_stop = np.tile(x[:,stop_position:,:], (N,1,1))
        return np.concatenate([x_start, x_mut, x_stop], axis=1)


    def delimit_range(self, x, start_position, stop_position):
        """Function to delimit sequence to a specific region.
        
        Parameters
        ----------
        x : torch.Tensor
            Batch of one-hot sequences (shape: (L,A)).
        start_position : int
            Index of start position along sequence to probe.
        stop_position : int
            Index of stop position along sequence to probe.

        Returns
        -------
        numpy.ndarray
            Delimited sequences with length l < L defined by
            l = stop_position - start_position (shape: (N,l,C)).
        """
        return x[start_position:stop_position,:]
    
