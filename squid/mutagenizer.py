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
    seed : int, optional
        Random seed for reproducibility. If None, results will not be reproducible.
        (defaults to None)

    Returns
    -------
    numpy.ndarray
        Batch of one-hot sequences with random mutagenesis applied.
    """

    def __init__(self, mut_rate, uniform=False, seed=None):
        self.mut_rate = mut_rate
        self.uniform = uniform
        self.seed = seed

    def __call__(self, x, num_sim):
        if self.seed is not None:
            np.random.seed(self.seed)

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
    
    Parameters
    ----------
    max_order : int, optional
        Maximum order of mutations to generate. If -1, generates all possible combinations.
        If 1, generates only single mutations (all SNVs). If 2, generates single and double mutations, etc.
        Must be less than or equal to sequence length L, or -1 for all combinations.
        (defaults to -1)
    mut_window : [int, int], optional
        Index of start and stop position along sequence to probe for mutations.
        If provided, only generates mutations within this window (inclusive on both ends).
        For example, mut_window=[4,6] will generate mutations at positions 4, 5, and 6.
        (defaults to None, which means the entire sequence is considered)
    batch_size : int, optional
        Batch size for one-hot encoding conversion. If None, converts all at once.
        For large sequences, using a batch size can help manage memory usage.
        (defaults to None)
    seed : int, optional
        Random seed for reproducibility. If None, results will not be reproducible.
        (defaults to None)

    Returns
    ----------
    numpy.ndarray
        Batch of one-hot sequences with combinatorial mutagenesis applied.
        For max_order=-1: number of sequences is A^L
        For max_order=k: number of sequences is 1 + sum(n_choose_r * (A-1)^r) for r in 1..k
        where:
        - L is sequence length
        - A is alphabet size
        - n_choose_r is the binomial coefficient (L choose r)
        - The leading 1 accounts for the reference sequence
        
    Examples
    --------
    For L=4, A=4:
    - max_order=1: 1 + C(4,1)*(3^1) = 1 + 12 = 13 sequences
    - max_order=2: 1 + C(4,1)*(3^1) + C(4,2)*(3^2) = 1 + 12 + 54 = 67 sequences

    Raises
    ------
    ValueError
        If max_order is greater than sequence length L or less than -1
    """
    def __init__(self, max_order=-1, mut_window=None, batch_size=256, seed=None):
        if max_order < -1:
            raise ValueError("max_order must be -1 or a non-negative integer")
        self.max_order = max_order
        self.mut_window = mut_window
        self.batch_size = batch_size
        self.seed = seed

    def __call__(self, x, num_sim): # 'num_sim' will be ignored
        if self.seed is not None:
            np.random.seed(self.seed)
            
        L, A = x.shape
        
        # If mut_window is provided, we'll only consider positions within that window
        if self.mut_window is not None:
            start_pos, stop_pos = self.mut_window
            stop_pos = stop_pos + 1  # Make stop_pos exclusive to include the last position
            window_length = stop_pos - start_pos
            if window_length <= 0:
                raise ValueError("mut_window stop_pos must be greater than or equal to start_pos")
            if start_pos < 0 or stop_pos > L:
                raise ValueError(f"mut_window must be within sequence bounds [0, {L}]")
        else:
            start_pos, stop_pos = 0, L
            window_length = L
            
        if self.max_order > window_length:
            raise ValueError(f"max_order ({self.max_order}) cannot exceed window length ({window_length})")
            
        x_index = np.argmax(x, axis=1)  # Get reference sequence indices
        from itertools import combinations, product

        # If max_order is -1, set it to window_length for complete enumeration
        max_order = window_length if self.max_order == -1 else self.max_order
        
        # Pre-calculate total size and allocate array
        total_variants = 1 + sum(  # +1 for reference sequence
            len(list(combinations(range(start_pos, stop_pos), order))) * (A-1)**order 
            for order in range(1, max_order + 1)
        )
        all_variants = np.zeros((total_variants, L), dtype=np.int8)
        all_variants[0] = x_index  # Add reference sequence
        
        # Pre-compute alternative bases for each position
        alt_bases_lookup = {i: np.array([b for b in range(A) if b != base]) 
                           for i, base in enumerate(x_index[start_pos:stop_pos], start=start_pos)}
        
        current_idx = 1  # Start after reference sequence
        
        # Generate variants for each order up to max_order
        for order in range(1, max_order + 1):
            n_positions = len(list(combinations(range(start_pos, stop_pos), order)))
            n_variants = n_positions * (A-1)**order
            
            with tqdm(total=n_variants, desc=f"Order {order} mutations") as pbar:
                for pos in combinations(range(start_pos, stop_pos), order):
                    # Get pre-computed alternative bases for these positions
                    alt_bases_per_pos = [alt_bases_lookup[p] for p in pos]
                    
                    # Generate all combinations at once for this position set
                    alt_combos = np.array(list(product(*alt_bases_per_pos)))
                    n_combos = len(alt_combos)
                    
                    # Create variants for all combinations at once
                    new_seqs = np.tile(x_index, (n_combos, 1))
                    new_seqs[:, pos] = alt_combos
                    
                    # Add to pre-allocated array
                    all_variants[current_idx:current_idx + n_combos] = new_seqs
                    current_idx += n_combos
                    pbar.update(n_combos)
        
        print("Converting to one-hot encoding...")
        if self.batch_size is None:
            # Convert all at once
            one_hot = np.eye(A, dtype=np.int8)[np.ascontiguousarray(all_variants)]
        else:
            # Convert in batches
            n_sequences = len(all_variants)
            one_hot = np.zeros((n_sequences, L, A), dtype=np.int8)
            
            for i in tqdm(range(0, n_sequences, self.batch_size), desc="One-hot encoding"):
                batch_end = min(i + self.batch_size, n_sequences)
                one_hot[i:batch_end] = np.eye(A, dtype=np.int8)[all_variants[i:batch_end]]
        
        return one_hot
    

class TwoHotMutagenesis(BaseMutagenesis):
    """Module to perform random mutagenesis using two-hot encoding.
    That is, encode each individual nucleotide at a given position
    using a one-hot encoding scheme, then represent the unphased
    diploid sequence as the sum of the two one-hot encoded nucleotides
    at each position. The sequence "AYCR", for example, would be encoded as:
    [[2, 0, 0, 0], [0, 1, 0, 1], [0, 2, 0, 0], [1, 0, 1, 0]].

    Parameters
    ----------
    mut_rate : float
        Mutation rate for random mutagenesis.
    uniform : bool, optional
        uniform (True), Poisson (False); sets the number of mutations per sequence.
        (defaults to False)
    seed : int, optional
        Random seed for reproducibility. If None, results will not be reproducible.
        (defaults to None)

    Returns
    ----------
    numpy.ndarray
        Batch of one-hot sequences with random mutagenesis applied, with alphabet:
        {A, C, G, T, R (A/G), Y (C/T), S (C/G), W (A/T), K (G/T), M (A/C)}, such that
        heterozygous positions are represented using the IUPAC ambiguity codes.
    """
    
    def __init__(self, mut_rate, uniform=False, seed=None):
        self.mut_rate = mut_rate
        self.uniform = uniform
        self.seed = seed

    def __call__(self, x, num_sim):
        if self.seed is not None:
            np.random.seed(self.seed)
            
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


def get_alternative_bases(ref_base, A):
    """Get all possible alternative bases for a given reference base."""
    return [b for b in range(A) if b != ref_base]



if __name__ == "__main__":
    if 1:
        print("\nTesting CombinatorialMutagenesis:")
        L = 10  # Change this value to test different lengths
        A = 4  # Alphabet size (A,C,G,T)
        
        # Create one-hot encoding for sequence of all A's
        x = np.zeros((L, A))
        x[:, 0] = 1  # Set first position (A) to 1 for all positions
        
        # Test with different max_order values
        for max_order in [2]:
            mut = CombinatorialMutagenesis(max_order=max_order, mut_window=[4, 6])
            result = mut(x, num_sim=None)
            
            # Convert results back to sequences for easy viewing
            sequences = []
            nucleotides = ['A', 'C', 'G', 'T']
            for seq in result:
                seq_indices = np.argmax(seq, axis=1)
                sequences.append(''.join([nucleotides[idx] for idx in seq_indices]))
            
            print(f"\nmax_order = {max_order}:")
            print(f"Number of sequences generated: {len(sequences)}")
            if len(sequences) < 50:  # Only print sequences if there aren't too many
                print("Sequences:")
                for seq in sequences:
                    print(seq)

    else:
        print("\nTesting RandomMutagenesis:")
        L = 20  # sequence length
        A = 4   # alphabet size
        
        # Create one-hot encoding for sequence of all A's
        x = np.zeros((L, A))
        x[:, 0] = 1  # Set first position (A) to 1 for all positions
        
        # Test with Poisson mutations, 10% mutation rate
        mut = RandomMutagenesis(mut_rate=0.1, uniform=False, seed=42)
        result = mut(x, num_sim=10)
        
        # Convert results back to sequences for easy viewing
        sequences = []
        nucleotides = ['A', 'C', 'G', 'T']
        for seq in result:
            seq_indices = np.argmax(seq, axis=1)
            sequences.append(''.join([nucleotides[idx] for idx in seq_indices]))
        
        print(f"Input sequence: {'A' * L}")
        print("\nMutated sequences:")
        for seq in sequences:
            print(seq)