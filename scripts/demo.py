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
            # Equivalent:
            # self.bias = jaxtorch.Param(
            #    lambda key: jnp.zeros(out_features))
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

class Model(jaxtorch.Module):
    def __init__(self):
        self.layer = Linear(3, 3)

    def forward(self, cx, x):
        # For submodules you invoke them normally, passing the
        # context. __call__ simply redirects to forward, so
        # self.layer.forward(cx, x) works just as well.
        return self.layer(cx, x)

model = Model()

# PRNG is a stateful convenience wrapper around jax's pure PRNG. It
# lets you generate a sequence of PRNGKeys with split().
rng = jaxtorch.PRNG(jax.random.PRNGKey(0))

# You initialize the weights by passing a RNG key. This recursively
# collects all params from submodules.
params = model.init_weights(rng.split())

# Parameters are stored in a jaxtorch.ParamState object
assert type(params) is jaxtorch.ParamState

# ParamState is just a dictionary indexing weights by their Param
# identifier. It is a valid pytree and can be differentiated.
assert type(params[model.layer.weight]) is jaxlib.xla_extension.DeviceArray
print(params[model.layer.weight])
# [[ 0.4625378   0.59526694 -0.3165015 ]
#  [ 0.17670134  0.8430584  -0.14678049]
#  [-0.29896697  0.5365826  -0.93163735]]

# Use model.state_dict(params) to convert params to a plain python
# dictionary of weights by name, suitable for serializing to disk.
for (key, value) in model.state_dict(params).items():
    print(key, ':')
    print(value)
# layer.weight :
# [[-0.22408472 -0.0259067  -1.1793687 ]
#  [ 0.04579717 -0.02665656  0.45956933]
#  [-0.8320761  -0.977453   -0.21575125]]
# layer.bias :
# [0. 0. 0.]

# Can save this state dict in pytorch format.
jaxtorch.pt.save(model.state_dict(params), '/tmp/state_dict.pt')
model.load_state_dict(params, jaxtorch.pt.load('/tmp/state_dict.pt'))

def loss(params, key):
    # Context wraps params and a PRNG key.
    cx = jaxtorch.Context(params, key)
    x = jnp.array([1.0,2.0,3.0])
    y = jnp.array([4.0,5.0,6.0])
    return jnp.mean((model(cx, x) - y)**2)
f_grad = jax.value_and_grad(loss)


for _ in range(100):
    (loss, grad) = f_grad(params, rng.split())
    # Gradient descent by tree_map'ing over the params and grads
    params = jax.tree_util.tree_map(lambda p, g: p - 0.01 * g, params, grad)

print(loss)
# 4.7440533e-08