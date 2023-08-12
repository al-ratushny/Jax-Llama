from typing import Iterable

from jax import numpy as jnp


class SimpleTokenizer:
    def __init__(self, path: str):
        text = open(path, 'rt').read()
        self.vocab = sorted(list(set(text)))
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}

    def encode(self, s: str):
        return jnp.array([self.stoi[ch] for ch in s])

    def decode(self, l):
        return ''.join([self.itos[i] for i in l.tolist()])

    @property
    def vocab_size(self):
        return len(self.vocab)
