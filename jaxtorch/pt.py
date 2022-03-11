"""Load and save jax state dicts and other objects as pytorch
checkpoint files.

"""

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import torch


@torch.no_grad()
def torch_to_jax(torch_dict):
    def from_torch(x):
        if isinstance(x, torch.Tensor):
            return jnp.asarray(x)
        return x

    return jax.tree_util.tree_map(from_torch, torch_dict)


@torch.no_grad()
def jax_to_torch(obj):
    def to_torch(x):
        if isinstance(x, jaxlib.xla_extension.DeviceArrayBase):
            return torch.as_tensor(np.array(x))
        return x

    return jax.tree_util.tree_map(to_torch, obj)


@torch.no_grad()
def load(f):
    """Converts torch.Tensor back to jax arrays after loading."""
    torch_dict = torch.load(f, map_location="cpu")
    return torch_to_jax(torch_dict)


@torch.no_grad()
def save(obj, f):
    """Converts jax arrays (anything under jaxlib.xla_extension.DeviceArrayBase) to torch.Tensor before saving."""
    torch_dict = jax_to_torch(obj)
    torch.save(torch_dict, f)
