import jax
from jax import numpy as jnp

from jax_llama.config import BATCH_SIZE, CONTEXT_WINDOW, TRAIN_SIZE


class Dataset:
    def __init__(self, path: str, tokenizer):
        text = open(path, 'rt').read()
        self.dataset = jnp.array(tokenizer.encode(text))
        self.key = jax.random.PRNGKey(0)

        train_part = int(TRAIN_SIZE * len(self.dataset))
        self.train = self.dataset[:train_part]
        self.test = self.dataset[train_part:]

    @property
    def n_train_steps(self):
        return len(self.train) // BATCH_SIZE

    @property
    def n_test_steps(self):
        return len(self.test) // BATCH_SIZE

    def get_batch(
        self,
        split: str,
    ) -> tuple:
        if split == 'train':
            batch_data = self.train
        elif split == 'test':
            batch_data = self.test
        else:
            raise ValueError('wrong split value')

        self.key, subkey = jax.random.split(self.key)
        ix = jax.random.randint(
            key=subkey,
            shape=(BATCH_SIZE,),
            minval=0,
            maxval=batch_data.shape[0] - CONTEXT_WINDOW - 1,
        )
        x = jnp.stack([batch_data[i:i + CONTEXT_WINDOW] for i in ix])
        y = jnp.stack([batch_data[i + 1:i + CONTEXT_WINDOW + 1] for i in ix])
        return x, y
