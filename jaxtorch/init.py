import math
import jax
import jax.numpy as jnp
import numpy as np
from jaxtorch import core


def zeros(*shape):
    shape = jax.core.canonicalize_shape(shape)
    return core.Param(shape, lambda key: jnp.zeros(shape))


def ones(*shape):
    shape = jax.core.canonicalize_shape(shape)
    return core.Param(shape, lambda key: jnp.ones(shape))


def normal(*shape, mean=0.0, stddev=1.0):
    shape = jax.core.canonicalize_shape(shape)
    return core.Param(shape, lambda key: mean + stddev * jax.random.normal(key, shape))


def const(tensor):
    shape = jax.core.canonicalize_shape(tensor.shape)
    return core.Param(shape, lambda key: tensor)


def full(*shape, value=1.0):
    shape = jax.core.canonicalize_shape(shape)
    return core.Param(shape, lambda key: jnp.full(shape, value))


def glorot_normal_t(key, *shape, gain=1.0):
    shape = jax.core.canonicalize_shape(shape)
    fan_out = shape[0] * np.prod(shape[2:])
    fan_in = shape[1] * np.prod(shape[2:])
    stddev = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return stddev * jax.random.normal(key, shape)


def glorot_normal(*shape, gain=1.0):
    return core.Param(shape, lambda key: glorot_normal_t(key, *shape, gain=gain))


def glorot_uniform(*shape, gain=1.0):
    shape = jax.core.canonicalize_shape(shape)
    fan_out = shape[0] * np.prod(shape[2:])
    fan_in = shape[1] * np.prod(shape[2:])
    stddev = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return normal(*shape, stddev=stddev)


def uniform(*shape, min=-1.0, max=1.0):
    shape = jax.core.canonicalize_shape(shape)
    return core.Param(
        shape, lambda key: jax.random.uniform(key, shape, minval=min, maxval=max)
    )


def kaiming_uniform(*shape, a=0, scale=1.0):
    shape = jax.core.canonicalize_shape(shape)
    fan_in = np.prod(shape[1:])
    gain = math.sqrt(2.0 / (1 + a**2))
    bound = scale * gain * math.sqrt(3.0 / fan_in)
    return uniform(*shape, min=-bound, max=bound)


def mup_input_init(*shape, mean=0.0, std=1.0):
    shape = jax.core.canonicalize_shape(shape)
    fan_in = np.prod(shape[1:])
    stddev = std / fan_in
    return normal(*shape, mean=mean, stddev=stddev)


def mup_output_init(*shape, mean=0.0, std=1.0):
    shape = jax.core.canonicalize_shape(shape)
    fan_in = np.prod(shape[1:])
    stddev = std / fan_in**2
    return normal(*shape, mean=mean, stddev=stddev)


def mup_hidden_init(*shape, mean=0.0, std=1.0):
    shape = jax.core.canonicalize_shape(shape)
    fan_in = np.prod(shape[1:])
    stddev = std / fan_in
    return normal(*shape, mean=mean, stddev=stddev)


def sum_init(*inits):
    def init(*shape):
        ps = [i(*shape).initializer for i in inits]

        def _init(key):
            ks = jax.random.split(key, len(ps))
            vs = [p(k) for p, k in zip(ps, ks)]
            return sum(vs)

        return core.Param(shape, _init)

    return init


def scale_init(scale, init, *shape):
    base = init(*shape).initializer
    return core.Param(shape, lambda key: scale * base(key))
