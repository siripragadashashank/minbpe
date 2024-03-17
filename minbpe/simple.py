"""
Minimal BPE tokenizer
"""
from .base import Tokenizer, get_stats, merge


class SimpleTokenizer(Tokenizer):
    """
    Extends base Tokenizer with simple encoding and decoding
    """
    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256, "Vocab size should be greater than 256"
        num_merges = vocab_size - 256
        




