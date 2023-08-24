from typing import Iterable

from jax import numpy as jnp


class SimpleTokenizer:
    def __init__(self, path: str):
        """
        Very simple tokenizer that considers
        every unique character as a separate token.

        Args:
            path: The path to the file with text.
        """
        text = open(path, 'rt').read()
        self.vocab = sorted(list(set(text)))
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}

    def encode(self, s: str):
        """
        Encodes a string into a list of indices.

        Args:
            s: The input string.

        Returns:
            The list of indices representing the encoded string.
        """
        return jnp.array([self.stoi[ch] for ch in s])

    def decode(self, l: list[int]):
        """
        Decodes a list of indices into a string.

        Args:
            l: The list of indices.

        Returns:
            The decoded string.
        """
        return ''.join([self.itos[i] for i in l])

    @property
    def vocab_size(self):
        """
        Returns the size of the vocabulary.

        Returns:
            The size of the vocabulary.
        """
        return len(self.vocab)
