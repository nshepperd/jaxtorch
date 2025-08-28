"""Wraps cbor2 with hooks for encoding and decoding tensors."""
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

# Standard tags for multidimensional arrays from RFC8746
# (little-endian, row-major).
TAG_FLOAT32 = 85
TAG_FLOAT64 = 86
TAG_INT32 = 78
TAG_INT64 = 79
TAG_ARRAY = 40

def encode_flat(arr):
    from cbor2 import CBORTag
    if arr.dtype == np.float32:
        return CBORTag(TAG_FLOAT32, arr.tobytes())
    if arr.dtype == np.int32:
        return CBORTag(TAG_INT32, arr.tobytes())
    else:
        raise NotImplementedError(f"Cannot encode array of dtype {arr.dtype}")

def default_encoder(encoder, value):
    from cbor2 import CBORTag
    if isinstance(value, jax.Array):
        encoder.encode(np.array(value))
    elif isinstance(value, np.ndarray):
        encoder.encode(CBORTag(TAG_ARRAY, [list(value.shape), encode_flat(value)]))
    else:
        raise NotImplementedError(f"Cannot encode object of type {type(value)}")

def tag_hook(decoder, tag, shareable_index=None, device=None):
    if tag.tag == TAG_ARRAY:
        [shape, value] = tag.value
        assert isinstance(value, np.ndarray)
        return jnp.asarray(value.reshape(shape))
    elif tag.tag == TAG_FLOAT32:
        return np.frombuffer(tag.value, dtype=np.float32)
    elif tag.tag == TAG_INT32:
        return np.frombuffer(tag.value, dtype=np.int32)
    elif tag.tag == TAG_INT64:
        return np.frombuffer(tag.value, dtype=np.int64)
    else:
        return tag

def dumps(obj, **kwargs):
    import cbor2
    return cbor2.dumps(obj, default=default_encoder, **kwargs)

def dump(obj, fp, **kwargs):
    import cbor2
    return cbor2.dump(obj, fp, default=default_encoder, **kwargs)

def loads(s, *, device=None, **kwargs):
    import cbor2
    return cbor2.loads(s, tag_hook=partial(tag_hook, device=device), **kwargs)

def load(fp, *, device=None, **kwargs):
    import cbor2
    return cbor2.load(fp, tag_hook=partial(tag_hook, device=device), **kwargs)
