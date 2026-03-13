"""
BMI203: Biocomputing Algorithms - Winter 2023
Final project: neural networks
"""

__version__ = "1.0.0"

from .nn import NeuralNetwork
from .preprocess import sample_seqs, one_hot_encode_seqs
from .io import read_text_file, read_fasta_file

__all__ = [
    "NeuralNetwork",
    "sample_seqs",
    "one_hot_encode_seqs",
    "read_text_file",
    "read_fasta_file"
]