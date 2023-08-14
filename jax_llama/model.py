from flax import linen as nn

from jax_llama.config import D_MODEL, VOCAB_SIZE


class SimpleModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Embed(VOCAB_SIZE, D_MODEL)(x)
        x = nn.RMSNorm()(x)
        x = nn.Dense(D_MODEL)(x)
        x = nn.relu(x)
        x = nn.Dense(VOCAB_SIZE)(x)
        return x
