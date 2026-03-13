# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # convert to lists if not already, to allow for indexing and sampling
    seqs = list(seqs)
    labels = list(labels)

    # separate positive and negative examples
    pos = [s for s, l in zip(seqs, labels) if l]
    neg = [s for s, l in zip(seqs, labels) if not l]

    n_pos = len(pos) # number of positive examples
    n_neg = len(neg) # number of negative examples

    # if either class is empty
    if n_pos == 0 or n_neg == 0:
        # nothing to balance so return original
        return seqs, labels

    target = max(n_pos, n_neg) # target number of samples for each class to balance dataset

    # sample with replacement to reach target for both classes
    pos_sampled = list(np.random.choice(pos, size=target, replace=True))
    neg_sampled = list(np.random.choice(neg, size=target, replace=True))

    sampled_seqs = pos_sampled + neg_sampled # combine sampled sequences
    sampled_labels = [True] * target + [False] * target # corresponding labels for sampled sequences

    # shuffle combined
    perm = np.random.permutation(len(sampled_seqs)) # generate random permutation of indices
    sampled_seqs = [sampled_seqs[i] for i in perm] # apply permutation to sequences
    sampled_labels = [sampled_labels[i] for i in perm] # apply same permutation to labels to maintain correct mapping

    return sampled_seqs, sampled_labels

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    # if input is empty, return empty array
    if len(seq_arr) == 0:
        return np.array([])

    seq_arr = list(seq_arr) # ensure input is a list to allow for indexing and length checks
    L = len(seq_arr[0]) # length of first sequence to use as reference
    for s in seq_arr: # for each sequence, check that it has the same length as the first sequence
        if len(s) != L: # if any sequence has a different length, raise an error since we cannot one-hot encode sequences of different lengths
            raise ValueError("All sequences must have the same length to one-hot encode")

    # define mapping from nucleotides to one-hot vectors
    mapping = {
        'A': [1, 0, 0, 0],
        'T': [0, 1, 0, 0],
        'C': [0, 0, 1, 0],
        'G': [0, 0, 0, 1]
    }

    n = len(seq_arr) # number of sequences to encode
    encodings = np.zeros((n, 4 * L), dtype=float) # initialize array to hold encodings, with shape (number of sequences, 4 times sequence length)

    for i, seq in enumerate(seq_arr): # for each sequence in the input list
        seq = seq.upper() # convert to uppercase to ensure mapping works regardless of input case
        vec = [] # list to hold one-hot encoding for current sequence
        for ch in seq: # for each character in the sequence
            if ch in mapping: # if the character is a valid nucleotide
                vec.extend(mapping[ch]) # extend the encoding vector with the corresponding one-hot encoding
            else:
                # else if the character is not a valid nucleotide
                vec.extend([0, 0, 0, 0]) # encode it as all zeros
        encodings[i, :] = np.array(vec) # store the encoding for the current sequence in the encodings array

    return encodings