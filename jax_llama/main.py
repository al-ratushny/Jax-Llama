import jax
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state
from jax import numpy as jnp

from jax_llama.config import DATA_FILE, LR, N_EPOCHS, VOCAB_SIZE
from jax_llama.data_utils import Dataset
from jax_llama.model import SimpleModel
from jax_llama.tokenizer import SimpleTokenizer


def cross_entropy_loss(logits, labels):
    one_hot_encoded_labels = jax.nn.one_hot(labels, VOCAB_SIZE)
    loss = optax.softmax_cross_entropy(
        logits=logits, labels=one_hot_encoded_labels
    ).mean()
    return loss


def compute_metrics(logits, labels):
    loss = jnp.mean(cross_entropy_loss(logits, labels))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {'loss': loss, 'accuracy': accuracy}


def init_train_state(
    model, rng, x,
) -> train_state.TrainState:
    params = model.init(rng, x)['params']
    optimizer = optax.adam(LR)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        tx=optimizer,
        params=params,
    )
    return state


@jax.jit
def train_step(state, xs, ys):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, xs)
        loss = cross_entropy_loss(logits, ys)
        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    logits = state.apply_fn({'params': state.params}, xs)
    metrics = compute_metrics(logits, ys)
    return state, metrics


@jax.jit
def test_step(state, xs, ys):
    logits = state.apply_fn({'params': state.params}, xs)
    return compute_metrics(logits, ys)


def accumulate_metrics(metrics):
    return {
        k: np.mean([metric[k] for metric in metrics]) for k in metrics[0]
    }


def train_and_validate(dataset: Dataset, state: train_state.TrainState):
    for epoch in range(N_EPOCHS):
        train_metrics = []
        for step in range(dataset.n_train_steps):
            xs, ys = dataset.get_batch('train')
            state, metrics = train_step(state, xs, ys)
            train_metrics.append(metrics)
            if step % 10 == 0:
                print(f'train; {step}/{dataset.n_train_steps}; {metrics}')

        test_metrics = []
        for step in range(dataset.n_test_steps):
            xs, ys = dataset.get_batch('test')
            metrics = test_step(state, xs, ys)
            test_metrics.append(metrics)
            if step % 10 == 0:
                print(f'test; {step}/{dataset.n_train_steps}; {metrics}')

    return state


tokenizer = SimpleTokenizer(DATA_FILE)
dataset = Dataset(DATA_FILE, tokenizer)
xs, _ = dataset.get_batch('train')
rng = jax.random.PRNGKey(0)
model = SimpleModel()
print(nn.tabulate(model, rng)(xs[0]))
state = init_train_state(model, rng, xs)
state = train_and_validate(dataset, state)