import sys
sys.path = ['.'] + sys.path
import torch
import numpy as np
import jax
import jax.numpy as jnp
import jaxtorch
from jaxtorch.core import Module, Context, PRNG

torch.backends.cudnn.deterministic = True

def check(old, new, limit=1e-6):
    old_np = np.array(old)
    new_np = np.array(new)
    assert old_np.shape == new_np.shape
    difference = np.abs(old_np - new_np).max()
    magnitude = np.mean((np.abs(old_np) + np.abs(new_np))/2)
    # Pretty loose bounds due to float32 precision. There seem to be
    # some implementation differences resulting in different error
    # bounds betwen jax and torch.
    assert difference/magnitude < limit, difference/magnitude

def totorch(x):
    return torch.tensor(np.array(x))
def fromtorch(x):
    return jnp.array(np.array(x))

@torch.no_grad()
def test_conv1d():
    rng = PRNG(jax.random.PRNGKey(0))
    x = jax.random.normal(key=rng.split(), shape=[1, 4, 5])
    w = jax.random.normal(key=rng.split(), shape=[3, 4, 3])
    b = jax.random.normal(key=rng.split(), shape=[3])

    x_torch = totorch(x)
    w_torch = totorch(w)
    b_torch = totorch(b)

    new_result = jaxtorch.nn.functional.conv1d(x, w, bias=b, padding='same')
    old_result = torch.nn.functional.conv1d(x_torch, w_torch, bias=b_torch, padding='same')
    check(old_result, new_result)

    new_result = jaxtorch.nn.functional.conv1d(x, w, padding='valid')
    old_result = torch.nn.functional.conv1d(x_torch, w_torch, padding='valid')
    check(old_result, new_result)

    new_result = jaxtorch.nn.functional.conv1d(x, w, padding='valid', stride=2)
    old_result = torch.nn.functional.conv1d(x_torch, w_torch, padding='valid', stride=2)
    check(old_result, new_result)

    new_result = jaxtorch.nn.functional.conv1d(x, w, padding='valid', dilation=2)
    old_result = torch.nn.functional.conv1d(x_torch, w_torch, padding='valid', dilation=2)
    check(old_result, new_result)

    w = jax.random.normal(key=rng.split(), shape=[4, 2, 3])
    w_torch = totorch(w)
    new_result = jaxtorch.nn.functional.conv1d(x, w, padding='valid', groups=2)
    old_result = torch.nn.functional.conv1d(x_torch, w_torch, padding='valid', groups=2)
    check(old_result, new_result)

@torch.no_grad()
def test_conv2d():
    rng = PRNG(jax.random.PRNGKey(0))
    x = jax.random.normal(key=rng.split(), shape=[1, 4, 5, 5])
    w = jax.random.normal(key=rng.split(), shape=[3, 4, 3, 3])
    b = jax.random.normal(key=rng.split(), shape=[3])

    x_torch = totorch(x)
    w_torch = totorch(w)
    b_torch = totorch(b)

    new_result = jaxtorch.nn.functional.conv2d(x, w, bias=b, padding='same')
    old_result = torch.nn.functional.conv2d(x_torch, w_torch, bias=b_torch, padding='same')
    check(old_result, new_result)

    new_result = jaxtorch.nn.functional.conv2d(x, w, padding='valid')
    old_result = torch.nn.functional.conv2d(x_torch, w_torch, padding='valid')
    check(old_result, new_result)

    new_result = jaxtorch.nn.functional.conv2d(x, w, padding='valid', stride=2)
    old_result = torch.nn.functional.conv2d(x_torch, w_torch, padding='valid', stride=2)
    check(old_result, new_result)

    new_result = jaxtorch.nn.functional.conv2d(x, w, padding='valid', dilation=2)
    old_result = torch.nn.functional.conv2d(x_torch, w_torch, padding='valid', dilation=2)
    check(old_result, new_result)

    w = jax.random.normal(key=rng.split(), shape=[4, 2, 3, 3])
    w_torch = totorch(w)
    new_result = jaxtorch.nn.functional.conv2d(x, w, padding='valid', groups=2)
    old_result = torch.nn.functional.conv2d(x_torch, w_torch, padding='valid', groups=2)
    check(old_result, new_result)

@torch.no_grad()
def test_conv2d_init():
    rng = PRNG(jax.random.PRNGKey(0))
    new_mod = jaxtorch.nn.Conv2d(256, 512, 3)
    new_weights = new_mod.init_weights(rng.split())
    old_mod = torch.nn.Conv2d(256, 512, 3)
    check(new_weights['weight'].std(), old_mod.weight.std(), 0.01)

@torch.no_grad()
def test_groupnorm():
    rng = PRNG(jax.random.PRNGKey(0))
    new = jaxtorch.nn.GroupNorm(8, 32)
    old = torch.nn.GroupNorm(8, 32)

    px = new.init_weights(rng.split())

    old.weight.data.copy_(totorch(px[new.weight.name]))
    old.bias.data.copy_(totorch(px[new.bias.name]))

    x = jax.random.normal(key=rng.split(), shape=[2, 32, 2])
    x_torch = totorch(x)

    cx = Context(px, rng.split())
    new_result = new(cx, x)
    old_result = old(x_torch)
    check(old_result, new_result)

@torch.no_grad()
def test_dropout():
    rng = PRNG(jax.random.PRNGKey(0))
    module = jaxtorch.nn.Dropout() # defaults to p=0.5

    px = module.init_weights(rng.split())

    x = jax.random.normal(key=rng.split(), shape=[1, 32])
    cx = Context(px, rng.split())

    out_train = module(cx.train_mode_(), x)
    assert (out_train != 0).sum() < 20
    out_eval = module(cx.eval_mode_(), x)
    assert (out_eval != 0).sum() == 32
