import math

import jax
import jax.numpy as jnp
import numpy as np

from jaxtorch import core


def zeros(*shape: int):
    shape = tuple(shape)
    return core.Param(shape, lambda key: jnp.zeros(shape))

def ones(*shape: int):
    shape = tuple(shape)
    return core.Param(shape, lambda key: jnp.ones(shape))

def normal(*shape: int, stddev: float | jax.Array=1.0):
    shape = tuple(shape)
    return core.Param(shape, lambda key: stddev * jax.random.normal(key, shape))

def const(tensor: jax.Array):
    shape = tuple(tensor.shape)
    return core.Param(shape, lambda key: tensor)

def glorot_normal(*shape: int):
    shape = tuple(shape)
    fan_out = shape[0] * np.prod(shape[2:])
    fan_in = shape[1] * np.prod(shape[2:])
    stddev = np.sqrt(2.0 / (fan_in + fan_out))
    return core.Param(shape, lambda key: stddev * jax.random.normal(key, shape))


def uniform(*shape: int, min: float | jax.Array = -1.0, max: float | jax.Array = 1.0):
    shape = tuple(shape)
    return core.Param(
        shape, lambda key: jax.random.uniform(key, shape, minval=min, maxval=max)
    )


def kaiming_uniform(*shape: int, a: float = 0):
    shape = tuple(shape)
    fan_in = np.prod(shape[1:])
    gain = math.sqrt(2.0 / (1 + a ** 2))
    bound = gain * math.sqrt(3.0 / fan_in)
    return uniform(*shape, min=-bound, max=bound)
