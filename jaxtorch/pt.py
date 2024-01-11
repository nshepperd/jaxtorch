"""Load and save jax state dicts and other objects as pytorch
checkpoint files.

"""

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import torch

from jaxtorch._util import ArrayTypes

@torch.no_grad()
def load(f):
    """Converts torch.Tensor back to jax arrays after loading."""
    def from_torch(x):
        if isinstance(x, torch.Tensor):
            return jnp.asarray(x)
        return x
    torch_dict = torch.load(f, map_location='cpu')
    return jax.tree_util.tree_map(from_torch, torch_dict)

@torch.no_grad()
def save(obj, f):
    """Converts jax arrays to torch.Tensor before saving."""
    def to_torch(x):
        if isinstance(x, ArrayTypes):
            return torch.as_tensor(np.array(x))
        return x
    torch_dict = jax.tree_util.tree_map(to_torch, obj)
    torch.save(torch_dict, f)
