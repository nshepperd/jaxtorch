import jax
import jax.numpy as jnp
import random

from jaxtorch import Module, PRNG, Context, ParamState
from jaxtorch import nn
from jaxtorch import cbor

import gpt

def test_layernorm():
    cx = Context(ParamState(), jax.random.PRNGKey(0))
    ln = nn.LayerNorm(cx, 5)
    x = jax.random.normal(shape=[2, 5], key=jax.random.PRNGKey(1))
    print(ln(cx, x))

def test_gpt():
    cx = Context(ParamState(), jax.random.PRNGKey(0))
    mconf = gpt.GPT1Config(10, 10)
    model = gpt.GPTLM(cx, mconf)
    # with open('mod.cbor', 'wb') as fp:
    #     cbor.dump(model.state_dict(px), fp)
    # with open('mod.cbor', 'rb') as fp:
    #     state_dict = cbor.load(fp)
    # model.load_state_dict(px, state_dict)
    print(model.state_dict(cx))
    x = jnp.array([[3, 2, 1]])
    print(model.loss(cx, x))

if __name__ == '__main__':
    test_gpt()