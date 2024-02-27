"""
Base recipe for a tokenizer and BPE helper functions
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional


def get_stats(ids: List[Any], counts: Optional[Dict[Any, Any]] = None) -> Dict[Any, Any]:
    """
    Given a list of integers, return a dictionary of count of consecutive pairs
    :param ids: List of Index tokens
    :param counts: Bigram counts of the tokens
    :return: counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: List[Any], pair: Tuple[Any, Any], idx: int) -> List[Any]:
    """
    Replace a pair of consecutive integers with a given integer index
    :param ids: List of token indices
    :param pair: pair of tokens to replace
    :param idx: new index to replace the pair with
    :return: new ids list with merged tokens
    """
    merged = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            merged.append(idx)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged


class Tokenizer:
    """
    Base class for tokenization
    """

    def __init__(self):
        self.merges = {}  # (int, int) -> int
        self.pattern = ""  # regex pattern
        self.special_tokens = {}  # str -> int, special tokens
        self.vocab = self._build_vocab()

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError

    def encode(self, text):
        raise NotImplementedError

    def decode(self, ids):
        raise NotImplementedError

    def _build_vocab(self):
        vocab = {i: bytes([i]) for i in range(256)}

        for (t1, t2), idx in self.merges.items():
            vocab[idx] = vocab[t1] + vocab[t2]

        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")

        return vocab

    def save(self, file_name):
        """
        Model file contains regex pattern, special tokens, merged indices

        vocab file is a human-readable format for inspection
        """
        model_file = file_name + ".model"
        with open(model_file, 'w') as fp:
            fp.write(f"{self.pattern}\n")
            fp.write(f"{len(self.special_tokens)}\n")
            # write the special tokens and index to file
            for special, idx in self.special_tokens.items():
                fp.write(f"{special} {idx}\n")
            # write the merged indices (idx1, idx2)
            for idx1, idx2 in self.merges:
                fp.write(f"{idx1} {idx2}\n")

        vocab_file = file_name + ".vocab"
        idx_to_merges = {idx: pair for idx, pair in self.merges.items()}
        with open(vocab_file, 'w', encoding="utf-8") as fp:
            for idx, token in self.vocab.items():
                if idx in idx_to_merges:
                    p1, p2 = idx_to_merges[idx]
                    fp.write(f"{p1} {p2} -> {token} {idx}\n")
                else:
                    fp.write(f"{token} {idx}\n")

    def load(self, model_file: str):
        # load only .vocab file
        assert model_file.endswith(".model")
        _index = 256
        with open(model_file, 'w', encoding="utf-8") as fp:
            # read first few lines sequentially for the pattern and special tokens
            self.pattern = fp.readline().strip()
            num_special_tokens = int(fp.readline().strip())
            for _ in range(num_special_tokens):
                special_token, idx = fp.readline().strip().split()
                self.special_tokens[special_token] = int(idx)
            for line in fp:
                idx1, idx2 = line.split()
                self.merges[(idx1, idx2)] = _index
                _index += 1

        self.vocab = self._build_vocab()
