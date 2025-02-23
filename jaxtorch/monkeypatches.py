"""Implement a bunch of convenience methods for jax arrays.

"""

import sys
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import functools
from einops import rearrange, reduce, repeat

from jaxtorch._util import ArrayTypes


def register(**kwargs):
    for (attr, fun) in kwargs.items():
        if hasattr(jnp.zeros([]), attr):
            print(f'Not monkeypatching DeviceArray and Tracer with `{attr}`, because that method is already implemented.', file=sys.stderr)
            continue
        for ty in ArrayTypes:
            setattr(ty, attr, fun)
        setattr(jax.core.Tracer, attr, fun)

# jnp.broadcast_to is implemented in terms of jax.Array.broadcast_to
# if it exists, so we need to reimplement it to not cause an infinite recursion.
# Annoying.
def broadcast_to(arr, shape):
  shape = (shape,) if jnp.ndim(shape) == 0 else shape
  shape = tuple(shape)  # check that shape is concrete
  arr_shape = tuple(arr.shape)
  if arr_shape == shape:
      return arr
  nlead = len(shape) - len(arr_shape)
  shape_tail = shape[nlead:]
  compatible = all(arr_d in (1, shape_d) for (arr_d, shape_d) in zip(arr_shape, shape_tail))
  if nlead < 0 or not compatible:
      raise ValueError(f"Incompatible shapes for broadcasting: {arr_shape} and requested shape {shape}")
  diff = tuple(i for i, (arr_d, shape_d) in enumerate(zip(arr_shape, shape_tail)) if arr_d != shape_d)
  kept_dims = tuple(nlead + i for i in range(len(arr_shape)) if i not in diff)
  return jax.lax.broadcast_in_dim(jnp.squeeze(arr, diff), shape, kept_dims)

register(
    square = lambda arr: arr**2,
    sqrt = jnp.sqrt,
    abs = jnp.abs,
    add = lambda a, b: a + b,
    sub = lambda a, b: a - b,
    div = lambda a, b: a / b,
    mul = lambda a, b: a * b,
    sin = jnp.sin,
    cos = jnp.cos,
    arcsin = jnp.arcsin,
    arccos = jnp.arccos,
    log = jnp.log,
    exp = jnp.exp,
    expm1 = jnp.expm1,
    log1p = jnp.log1p,
    neg = lambda x: -x,
    clamp = lambda a, minval=None, maxval=None: jnp.clip(a, a_min=minval, a_max=maxval),
    unsqueeze = lambda arr, axis=0: jnp.expand_dims(arr, axis),
    rearrange = rearrange,
    reduce = reduce,
    broadcast_to = broadcast_to,
    isfinite = jnp.isfinite,
    isnan = jnp.isnan,
    logical_not = jnp.logical_not,
)
