import jax
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state
from jax import jit
from jax import numpy as jnp
from tqdm import trange

from jax_llama.config import (CONTEXT_WINDOW, DATA_FILE, LR, N_EPOCHS,
                              TEST_STEPS, TRAIN_STEPS, VOCAB_SIZE)
from jax_llama.data_utils import Dataset
from jax_llama.model import Llama
from jax_llama.tokenizer import SimpleTokenizer


def cross_entropy_loss(logits, labels):
    """
    Computes the cross-entropy loss between the predicted logits and the true labels.

    Args:
        logits: The predicted logits, which are the output of the model
            before the softmax activation function is applied.
        labels: The true labels for the input data.
    """
    one_hot_encoded_labels = jax.nn.one_hot(labels, VOCAB_SIZE)
    loss = optax.softmax_cross_entropy(
        logits=logits, labels=one_hot_encoded_labels
    ).mean()
    return loss


def compute_metrics(logits, labels):
    """
    Computes the loss and accuracy metrics based on the predicted logits and true labels.

    Args:
        logits: The predicted logits, which are the output of the model
            before the softmax activation function is applied.
        labels: The true labels for the input data.
    """
    loss = jnp.mean(cross_entropy_loss(logits, labels))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {'loss': loss, 'accuracy': accuracy}


def init_train_state(
    model, rng, x,
) -> train_state.TrainState:
    """
    Initializes the training state for a given model,
    random number generator (RNG), and input data.

    Args:
        model: The model for which the training state is being initialized.
        rng: The random number generator used for initialization.
        x: The input data.
    """
    params = model.init(rng, x)['params']
    optimizer = optax.adam(LR)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        tx=optimizer,
        params=params,
    )
    return state


@jit
def train_step(state, xs, ys):
    """Performs a single training step using a given state."""
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


@jit
def test_step(state, xs, ys):
    """Performs a single test step using."""
    logits = state.apply_fn({'params': state.params}, xs)
    return compute_metrics(logits, ys)


def accumulate_metrics(metrics):
    """Computes mean value for all the metrics."""
    return {
        k: np.mean([metric[k] for metric in metrics]) for k in metrics[0]
    }


def train_and_validate(dataset: Dataset, state: train_state.TrainState):
    """Provides a training loop for a given number of epochs."""
    for epoch in range(N_EPOCHS):
        train_metrics = []
        for step in (pbar := trange(TRAIN_STEPS)):
            xs, ys = dataset.get_batch('train')
            state, metrics = train_step(state, xs, ys)
            train_metrics.append(metrics)
            if step % 10 == 0:
                pbar.set_description(
                    f'train; {epoch}/{N_EPOCHS}; {step}/{TRAIN_STEPS}; '
                    f'accuracy={metrics["accuracy"]}; loss={metrics["loss"]}',
                )
        print(accumulate_metrics(train_metrics[-100:]))
        train_metrics = []

        test_metrics = []
        for step in (pbar := trange(TEST_STEPS)):
            xs, ys = dataset.get_batch('test')
            metrics = test_step(state, xs, ys)
            test_metrics.append(metrics)
            if step % 10 == 0:
                pbar.set_description(
                    f'test; {epoch}/{N_EPOCHS}; {step}/{TEST_STEPS}; '
                    f'accuracy={metrics["accuracy"]}; loss={metrics["loss"]}',
                )
        print(accumulate_metrics(test_metrics[-100:]))
        test_metrics = []

    return state


def predict(state, xs, n_tokens=10):
    """Generates predictions for a given input sequence using a trained model."""
    for i in range(n_tokens):
        logits = state.apply_fn(
            {'params': state.params},
            np.expand_dims(xs[-CONTEXT_WINDOW:], 0),
        )
        last_prediction = logits[-1, -1]
        p = nn.softmax(last_prediction, axis=-1)
        xs = jnp.append(xs, jnp.argmax(p))
    return tokenizer.decode(xs.tolist())


if __name__ == '__main__':
    # init
    jax.config.update('jax_platform_name', 'cpu')
    tokenizer = SimpleTokenizer(DATA_FILE)
    dataset = Dataset(DATA_FILE, tokenizer)
    xs, _ = dataset.get_batch('train')
    rng = jax.random.PRNGKey(0)
    model = Llama()
    print(nn.tabulate(model, rng)(xs))
    state = init_train_state(model, rng, xs)

    # train
    state = train_and_validate(dataset, state)

    # predict
    prediction = predict(state, tokenizer.encode('You are all resolved'))
    print(prediction)
