# Jaxtorch (a jax nn library)

This is my jax based nn library. I created this because I was annoyed
by the complexity and 'magic'-ness of the popular jax frameworks
(flax, haiku).

The objective is to enable pytorch-like model definition and training
with a minimum of magic. Simple example:

```python
import jax
import jax.numpy as jnp
import jaxlib
import jaxtorch

# Modules are just classes that inherit jaxtorch.Module
class Linear(jaxtorch.Module):
    # They can accept any constructor parameters
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        # Parameters are represented by a Param type, which identifies
        # them, and specifies how to initialize them.
        self.weight = jaxtorch.init.glorot_normal(out_features, in_features)
        assert type(self.weight) is jaxtorch.Param
        if bias:
            self.bias = jaxtorch.init.zeros(out_features)
        else:
            self.bias = None

    # The forward function accepts cx, a Context object as the first argument
    # always. This provides random number generation as well as the parameters.
    def forward(self, cx: jaxtorch.Context, x):
        # Parameters are looked up in the context using the stored identifier.
        y = x @ jnp.transpose(cx[self.weight])
        if self.bias:
            y = y + cx[self.bias]
        return y

model = Linear(3, 3)

# You initialize the weights by passing a RNG key.
# Calling init_weights also names all the parameters in the Module tree.
params = model.init_weights(jax.random.PRNGKey(0))

# Parameters are stored in a dictionary by name.
assert type(params) is dict
assert type(params[model.weight.name]) is jaxlib.xla_extension.DeviceArray
assert model.weight.name == 'weight'

def loss(params, key):
    cx = jaxtorch.Context(params, key)
    x = jnp.array([1.0,2.0,3.0])
    y = jnp.array([4.0,5.0,6.0])
    return jnp.mean((model(cx, x) - y)**2)
f_grad = jax.value_and_grad(loss)

for _ in range(100):
    (loss, grad) = f_grad(params, jax.random.PRNGKey(0))
    params = jax.tree_util.tree_map(lambda p, g: p - 0.01 * g, params, grad)
print(loss)
# 4.7440533e-08
```
