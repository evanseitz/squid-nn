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
    def __init__(self, mut_generator, mut_predictor, seq_length, mut_window=None, context_agnostic=False):
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
        self.context_agnostic = context_agnostic


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
            if self.context_agnostic:
                x_mut_full = self.pad_seq_random(x_mut, x, self.start_position, self.stop_position)
            else:
                x_mut_full = self.pad_seq(x_mut, x, self.start_position, self.stop_position)
            y_mut = self.mut_predictor(x_mut_full)
        else:
            x_mut = self.mut_generator(x, num_sim)
            y_mut = self.mut_predictor(x_mut)

        return x_mut, y_mut


    def pad_seq(self, x_mut, x, start_position, stop_position):
        """Function to pad mutated sequences on both sides with the original unmutated context.
        
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



    def pad_seq_random(self, x_mut, x, start_position, stop_position):
        """Function to pad mutated sequences on both sides with random DNA.
        
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
        x_shuffle = dinuc_shuffle(x,num_shufs= N)
        return np.concatenate([x_shuffle[:,:start_position,:], x_mut, x_shuffle[:,stop_position:,:]], axis=1)



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
    



#------------------------------------------------------------------------------------
# Useful shuffle function
#------------------------------------------------------------------------------------


# taken from https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py
def dinuc_shuffle(seq, num_shufs=None, rng=None):
    """
    Creates shuffles of the given sequence, in which dinucleotide frequencies
    are preserved.
    Arguments:
        `seq`: either a string of length L, or an L x D NumPy array of one-hot
            encodings
        `num_shufs`: the number of shuffles to create, N; if unspecified, only
            one shuffle will be created
        `rng`: a NumPy RandomState object, to use for performing shuffles
    If `seq` is a string, returns a list of N strings of length L, each one
    being a shuffled version of `seq`. If `seq` is a 2D NumPy array, then the
    result is an N x L x D NumPy array of shuffled versions of `seq`, also
    one-hot encoded. If `num_shufs` is not specified, then the first dimension
    of N will not be present (i.e. a single string will be returned, or an L x D
    array).
    """

    def string_to_char_array(seq):
        """
        Converts an ASCII string to a NumPy array of byte-long ASCII codes.
        e.g. "ACGT" becomes [65, 67, 71, 84].
        """
        return np.frombuffer(bytearray(seq, "utf8"), dtype=np.int8)


    def char_array_to_string(arr):
        """
        Converts a NumPy array of byte-long ASCII codes into an ASCII string.
        e.g. [65, 67, 71, 84] becomes "ACGT".
        """
        return arr.tostring().decode("ascii")


    def one_hot_to_tokens(one_hot):
        """
        Converts an L x D one-hot encoding into an L-vector of integers in the range
        [0, D], where the token D is used when the one-hot encoding is all 0. This
        assumes that the one-hot encoding is well-formed, with at most one 1 in each
        column (and 0s elsewhere).
        """
        tokens = np.tile(one_hot.shape[1], one_hot.shape[0])  # Vector of all D
        seq_inds, dim_inds = np.where(one_hot)
        tokens[seq_inds] = dim_inds
        return tokens


    def tokens_to_one_hot(tokens, one_hot_dim):
        """
        Converts an L-vector of integers in the range [0, D] to an L x D one-hot
        encoding. The value `D` must be provided as `one_hot_dim`. A token of D
        means the one-hot encoding is all 0s.
        """
        identity = np.identity(one_hot_dim + 1)[:, :-1]  # Last row is all 0s
        return identity[tokens]

    if type(seq) is str:
        arr = string_to_char_array(seq)
    elif type(seq) is np.ndarray and len(seq.shape) == 2:
        seq_len, one_hot_dim = seq.shape
        arr = one_hot_to_tokens(seq)
    else:
        raise ValueError("Expected string or one-hot encoded array")


    if not rng:
        rng = np.random.RandomState()
   
    # Get the set of all characters, and a mapping of which positions have which
    # characters; use `tokens`, which are integer representations of the
    # original characters
    chars, tokens = np.unique(arr, return_inverse=True)

    # For each token, get a list of indices of all the tokens that come after it
    shuf_next_inds = []
    for t in range(len(chars)):
        mask = tokens[:-1] == t  # Excluding last char
        inds = np.where(mask)[0]
        shuf_next_inds.append(inds + 1)  # Add 1 for next token
 
    if type(seq) is str:
        all_results = []
    else:
        all_results = np.empty(
            (num_shufs if num_shufs else 1, seq_len, one_hot_dim),
            dtype=seq.dtype
        )

    for i in range(num_shufs if num_shufs else 1):
        # Shuffle the next indices
        for t in range(len(chars)):
            inds = np.arange(len(shuf_next_inds[t]))
            inds[:-1] = rng.permutation(len(inds) - 1)  # Keep last index same
            shuf_next_inds[t] = shuf_next_inds[t][inds]

        counters = [0] * len(chars)
       
        # Build the resulting array
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]

        if type(seq) is str:
            all_results.append(char_array_to_string(chars[result]))
        else:
            all_results[i] = tokens_to_one_hot(chars[result], one_hot_dim)
    return all_results if num_shufs else all_results[0]

