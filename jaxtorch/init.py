import jax
import jax.numpy as jnp
import numpy as np
from jaxtorch import core

def mkdesc(name, *args, **kwargs):
    return name + '(' + ', '.join([repr(x) for x in args] + [f'{k}={v}' for (k,v) in kwargs.items()]) + ')'

def zeros(*shape):
    shape = jax.core.canonicalize_shape(shape)
    return core.Param(lambda key: jnp.zeros(shape), desc=mkdesc('zeros', *shape))

def ones(*shape):
    shape = jax.core.canonicalize_shape(shape)
    return core.Param(lambda key: jnp.ones(shape), desc=mkdesc('ones', *shape))

def normal(*shape, stddev=1.0):
    shape = jax.core.canonicalize_shape(shape)
    return core.Param(lambda key: stddev * jax.random.normal(key, shape), desc=mkdesc('normal', *shape, stddev=stddev))

def const(tensor):
    shape = jax.core.canonicalize_shape(tensor.shape)
    return core.Param(lambda key: tensor)

def glorot_normal(*shape):
    shape = jax.core.canonicalize_shape(shape)
    stddev = np.sqrt(2.0 / (shape[0] + shape[1]))
    return core.Param(lambda key: stddev * jax.random.normal(key, shape), desc=mkdesc('glorot_normal', *shape))

def uniform(*shape, min=-1.0, max=1.0):
    shape = jax.core.canonicalize_shape(shape)
    return core.Param(lambda key: jax.random.uniform(key, shape, minval=min, maxval=max))