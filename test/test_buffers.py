import torch
import numpy as np
import jax
import jax.numpy as jnp
from jaxtorch.core import Module, Context, PRNG, Buffer
jax.config.update("jax_default_matmul_precision", "float32")

def check(old, new, rtol=1e-6, atol=1e-6):
    old_np = np.asarray(old)
    new_np = np.asarray(new)
    assert old_np.shape == new_np.shape
    np.testing.assert_allclose(old_np, new_np, rtol=rtol, atol=atol)

def totorch(x):
    return torch.tensor(np.asarray(x))
def fromtorch(x):
    return jnp.array(np.asarray(x))

class Rotary(Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.cos = Buffer((max_seq_len, dim // 2), persistent=False)
        self.sin = Buffer((max_seq_len, dim // 2), persistent=False)

    def setup(self, cx):
        freqs = (1/1024.0)**(jnp.linspace(0, 1, self.dim // 2))
        theta = jnp.arange(self.max_seq_len)[:, None] * freqs[None, :]
        cx[self.cos] = jnp.cos(freqs * theta)
        cx[self.sin] = jnp.sin(freqs * theta)

    def forward(self, cx: Context, x: jax.Array):
        # x: n s h d
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        cos = cx[self.cos][:seq_len, None, :]
        sin = cx[self.sin][:seq_len, None, :]
        x1, x2 = jnp.split(x, 2, axis=-1)
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return jnp.concatenate([y1, y2], axis=-1)

def test_rotary():
    rotary = Rotary(dim=64)
    cx = Context({}, jax.random.PRNGKey(0))
    rotary.setup(cx)
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 10, 4, 64))
    y = rotary.forward(cx, x)
    assert y.shape == x.shape
    np.testing.assert_allclose(jnp.square(x).sum(-1), jnp.square(y).sum(-1), rtol=1e-6, atol=1e-6)

    # persistent=False
    assert rotary.state_dict(cx.params, cx.buffers) == {}