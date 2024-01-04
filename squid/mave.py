import numpy as np
import random


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
    context_agnostic : boole
        Option for generating global neighborhoods, such that the
        sequence surrounding a conserved pattern of interest is
        randomly mutated across the in silico MAVE dataset
    inter_window : [int, int] or [[int, int], [int, int], ...]
        Index of start and stop position of each inter-site window,
        where each window defines the boundaries of the sequence
        in between two sites of interest (optional,  for 'context_agnostic')
    save_window : [int <= mut_window[0], int >= mut_window[1]]
        Window used for delimiting sequences that are exported in 'x_mut' array;
        if used, the 'save_window' interval must be equal to or larger 'mut_window',
        and if larger, 'save window' must contain the interval 'mut_window' entirely
    alphabet : list
        The alphabet used to determine the C characters in the logo such that
        each entry is a string; e.g., ['A','C','G','T'] for DNA.
    """
    def __init__(self, mut_generator, mut_predictor, seq_length, mut_window=None, context_agnostic=False, inter_window=None, save_window=None, alphabet=['A','C','G','T']):
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
            mut_window = [self.start_position, self.stop_position]
        self.context_agnostic = context_agnostic
        self.inter_window = inter_window
        if save_window is not None and mut_window is not None:
            if (save_window[0] > mut_window[0]) or (save_window[1] < mut_window[1]):
                save_window = None
                print("Conflict found in 'save_window' interval, setting to None.")
        self.save_window = save_window
        self.alphabet = alphabet


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
                x_mut = self.pad_seq_random(x_mut, x, self.start_position, self.stop_position)

                if self.save_window:
                    print("Variable 'save_window' not yet implemented for 'context_agnostic' mode.")

                # optional: perform global mutagenesis in between two (or more) sites of interest
                if self.inter_window is not None:
                    num_inter_windows = sum(isinstance(i, list) for i in self.inter_window) # count number of inter-site windows
                    if num_inter_windows == 0:
                        num_inter_windows += 1
                        self.inter_window = [self.inter_window]
                    for w in range(num_inter_windows):
                        w_start = self.inter_window[w][0]
                        w_stop = self.inter_window[w][1]
                        x_mut = self.pad_seq_random(x_mut, x, w_start, w_stop, inter=True)

            else:
                x_mut = self.pad_seq(x_mut, x, self.start_position, self.stop_position, self.save_window)

            if self.mut_predictor is None: # skip inference
                y_mut = None
            else: # required for surrogate modeling
                y_mut = self.mut_predictor(x_mut, x, self.save_window)

        else:
            x_mut = self.mut_generator(x, num_sim)
            if self.mut_predictor is None: # skip inference
                y_mut = None
            else: # required for surrogate modeling
                y_mut = self.mut_predictor(x_mut, x, self.save_window)

        return x_mut, y_mut


    def pad_seq(self, x_mut, x, start_position, stop_position, save_window):
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
        x = x[np.newaxis,:].astype('uint8')
        #x_start = np.tile(x[:,:start_position,:], (N,1,1)) # high memory use
        #x_stop = np.tile(x[:,stop_position:,:], (N,1,1)) # high memory use
        if save_window is None:
            x_start = np.broadcast_to(x[:,:start_position,:], (N,start_position,x.shape[2]))
            x_stop = np.broadcast_to(x[:,stop_position:,:], (N,x.shape[1]-stop_position,x.shape[2]))
        else:
            x_start = np.broadcast_to(x[:,save_window[0]:start_position,:], (N,start_position-save_window[0],x.shape[2]))
            x_stop = np.broadcast_to(x[:,stop_position:save_window[1],:], (N,save_window[1]-stop_position,x.shape[2]))

        return np.concatenate([x_start, x_mut, x_stop], axis=1)



    def pad_seq_random(self, x_mut, x, start_position, stop_position, dinuc=False, inter=False):
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
        dinuc : boole
            Perform mutagenesis by random shuffle (False) or dinucleotide shuffle (True).
        inter : boole
            Pad sequence to the left and right of 'mut_window' (False) or within 'inter_window' (True)

        Returns
        -------
        numpy.ndarray
            Sequences with randomly mutated segments, padded to correct shape
            with random DNA (shape: (N,L,C)).
        """
        x = x.astype('uint8')
        N = x_mut.shape[0]
        if dinuc is False:
            x_shuffle = random_shuffle(x, self.alphabet, num_shufs=N)
        else:
            x_shuffle = dinuc_shuffle(x, num_shufs=N)

        if inter is False:
            x_padded = np.concatenate([x_shuffle[:,:start_position,:], x_mut, x_shuffle[:,stop_position:,:]], axis=1)
        else:
            x_padded = np.concatenate([x_mut[:,:start_position,:], x_shuffle[:,start_position:stop_position,:], x_mut[:,stop_position:,:]], axis=1)

        return x_padded



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

def random_shuffle(seq, alphabet=['A','C','G','T'], num_shufs=None, rng=None):
    """Creates random shuffles with equiprobability of characters at each position
    
    Parameters
    ----------
    seq : ndarray
        one-hot encoding of sequence
    num_shufs : int
        the number of shuffles to create; if unspecified, only one shuffle will be created

    Returns
    -------
    ndarray
        ndarray of shuffled versions of 'seq' (shape=(N,L,D)), also one-hot encoded
        If 'num_shufs' is not specified, then the first dimension of N will not be present
        (i.e. a single string will be returned, or an LxD array).
    """
    def seq2oh(seq, alphabet=['A','C','G','T']):   
        L = len(seq)
        one_hot = np.zeros(shape=(L,len(alphabet)), dtype=np.float32)
        for idx, i in enumerate(seq):
            for jdx, j in enumerate(alphabet):
                if i == j:
                    one_hot[idx,jdx] = 1
        return one_hot
    
    if num_shufs is None:
        num_shufs = 1

    seqs = np.zeros(shape=(num_shufs,seq.shape[0],seq.shape[1]))
    for seq_idx in range(num_shufs):
        random_seq = ''.join(random.choices(str(''.join(alphabet)), k=seq.shape[0]))
        seqs[seq_idx,:,:] = seq2oh(random_seq, alphabet)
    return seqs



# taken from https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py
def dinuc_shuffle(seq, num_shufs=None, rng=None):
    """Creates shuffles of the given sequence, in which dinucleotide frequencies
    are preserved.

    Parameters
    ----------
    seq : str or ndarray
        either a string of length L, or an L x D NumPy array of one-hot encodings
    num_shufs : int
        the number of shuffles to create, N; if unspecified, only one shuffle will be created
        `rng`: a NumPy RandomState object, to use for performing shuffles

    Returns
    -------
    list (if 'seq' is string)
        List of N strings of length L, each one being a shuffled version of 'seq'
        
    ndarray (if 'seq' is ndarray)
        ndarray of shuffled versions of 'seq' (shape=(N,L,D)), also one-hot encoded
        If 'num_shufs' is not specified, then the first dimension of N will not be present
        (i.e. a single string will be returned, or an LxD array).
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

