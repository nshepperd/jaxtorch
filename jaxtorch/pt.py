"""Load and save jax state dicts and other objects as pytorch
checkpoint files.

"""

import jax
import jax.numpy as jnp
import numpy as np


def load(f, device=None):
    """Converts torch.Tensor back to jax arrays after loading."""
    import torch
    def from_torch(x):
        if isinstance(x, torch.Tensor):
            return jnp.asarray(x.detach().cpu().numpy(), device=device)
        return x
    torch_dict = torch.load(f, map_location='cpu')
    return jax.tree_util.tree_map(from_torch, torch_dict)

def save(obj, f):
    """Converts jax arrays to torch.Tensor before saving."""
    import torch
    def to_torch(x):
        if isinstance(x, jax.Array):
            return torch.as_tensor(np.array(x))
        return x
    torch_dict = jax.tree_util.tree_map(to_torch, obj)
    torch.save(torch_dict, f)
