from flax import linen as nn
from jax import numpy as jnp
from jax_llama.config import (
    CONTEXT_WINDOW, D_MODEL, VOCAB_SIZE,
    N_HEADS, N_LAYES,
)
import seaborn as sns
import matplotlib.pyplot as plt


class Llama(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Embed(VOCAB_SIZE, D_MODEL)(x)
        for _ in range(N_LAYES):
            x = LlamaBlock()(x)
        ff = nn.Sequential([
            nn.Dense(D_MODEL),
            SwiGLU(),
            nn.Dense(VOCAB_SIZE),
        ])
        x = ff(x)
        return x


class LlamaBlock(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.RMSNorm()(x)
        x += RoPEAttention()(x)
        x = nn.RMSNorm()(x)
        ff = nn.Sequential([nn.Dense(D_MODEL), SwiGLU()])
        x += ff(x)
        return x


class SwiGLU(nn.Module):
    @nn.compact
    def __call__(self, x, beta: float = 1.0):
        gate_dense = nn.Dense(D_MODEL)(x)
        # !!!element-wise multiplication
        gate = gate_dense * nn.sigmoid(beta * gate_dense)
        x = nn.Dense(D_MODEL)(x)
        return gate * x


class RoPEAttention(nn.Module):
    @nn.compact
    def __call__(self, x):
        heads = []

        R = _get_rotation_matrix()
        for _ in range(N_HEADS):
            query_i = nn.Dense(D_MODEL)(x)
            keys_i = nn.Dense(D_MODEL)(x)
            values_i = nn.Dense(D_MODEL)(x)
            query_i = jnp.einsum("...i, ...ij -> ...j", query_i, R)
            keys_i = jnp.einsum("...i, ...ij -> ...j", keys_i, R)
            values_i = jnp.einsum("...i, ...ij -> ...j", values_i, R)
            head = self.__compute_head(keys_i, query_i, values_i)
            heads.append(head)

        heads = jnp.concatenate(heads, axis=-1)
        output = nn.Dense(D_MODEL)(heads)
        return output

    def __compute_head(self, query, keys, values):
        casual_mask = jnp.tril(jnp.ones((CONTEXT_WINDOW, CONTEXT_WINDOW)))
        x = jnp.einsum("...ik, ...jk -> ...ij", query, keys) / jnp.sqrt(D_MODEL)
        x = jnp.where(casual_mask, x, -1e9)
        x = nn.softmax(x)
        x = jnp.matmul(x, values)
        return x


def _get_rotation_matrix():
    R = jnp.zeros((CONTEXT_WINDOW, D_MODEL, D_MODEL))
    for pos in range(CONTEXT_WINDOW):
        for i in range(D_MODEL // 2):
            theta = 10000.0 ** (-2.0 * (i - 1) / D_MODEL)
            m_theta = pos * theta
            j, k = 2 * i, 2 * i + 1
            R = R.at[pos, j, j].set(jnp.cos(m_theta))
            R = R.at[pos, j, k].set(-jnp.sin(m_theta))
            R = R.at[pos, k, j].set(jnp.sin(m_theta))
            R = R.at[pos, k, k].set(jnp.cos(m_theta))
    return R


def __plot_rotation_matrix():
    # view test for a rotation matrix,
    # since it's always useful to see
    # that it works as you expect it
    R = _get_rotation_matrix()  # type: ignore
    sns.heatmap(R[10, :, :])
    plt.savefig('rotation_matrix.png')


if __name__ == "__main__":
    __plot_rotation_matrix()
