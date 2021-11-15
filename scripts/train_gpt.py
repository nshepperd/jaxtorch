import jax
import jax.numpy as jnp
import random

from jaxtorch import Module, PRNG, Context
from jaxtorch import nn
from jaxtorch import cbor

import gpt

class SGD(object):
    def __init__(self, parameters):
        self.parameters = list(parameters)

    def step(self, px, grad, lr):
        new_values = px.clone()
        for p in self.parameters:
            new_values[p] = px[p] - grad[p] * lr
        return new_values

def sgd(params, grad, lr):
    return jax.tree_util.tree_map(lambda p, g: p - g * lr, params, grad)

def main():
    rng = PRNG(jax.random.PRNGKey(0))
    mconf = gpt.GPT1Config(256, 64,
                           n_layer = 3,
                           n_head = 2,
                           n_embd = 256)
    model = gpt.GPTLM(mconf)
    px = model.init_weights(rng.split())
    with open('gpt.py', 'rb') as fp:
        data = fp.read()

    def loss(px, seq, key):
        cx = Context(px, key)
        return model.loss(cx, seq)
    f_grad = jax.jit(jax.value_and_grad(loss))

    counter = 1
    while True:
        i = random.randint(0, len(data)-64)
        seq = jnp.array([list(data[i:i+64])])
        (v_loss, v_grad) = f_grad(px, seq, rng.split())
        px = sgd(px, v_grad, 0.001)
        print(counter, v_loss)

        if counter % 100 == 0:
            idx = jnp.array([[-1] * 64])
            idx = model.generate(Context(px, rng.split()), idx)
            print(bytes(idx.squeeze().tolist()).decode('utf-8', errors='replace'))

        counter += 1

if __name__ == '__main__':
    main()