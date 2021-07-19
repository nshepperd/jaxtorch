import jax
import jax.numpy as jnp
import random

from jaxtorch import Module, PRNG, Context, ParamState
from jaxtorch import init

class Linear(Module):
    def __init__(self, c1, c2):
        self.c1 = c1
        self.c2 = c2
        self.weight = init.normal(c1, c2)
        self.bias = init.zeros(c2)

    def __repr__(self):
        return f'Linear({self.c1}, {self.c2})'

    def forward(self, cx, x):
        return x @ cx[self.weight] + cx[self.bias]

class Tanh(Module):
    def forward(self, cx, x):
        return jax.numpy.tanh(x)

class Dropout(Module):
  def __init__(self, rate=0.5):
    self.rate = rate

  def forward(self, cx, x):
    key = cx.rng.split()
    p = jax.random.bernoulli(key, 1.0 - self.rate, shape=x.shape)
    return x * p / (1.0 - self.rate)

class Sequential(Module):
    def __init__(self, *modules):
        self.modules = modules

    def forward(self, cx, x):
        for module in self.modules:
            x = module(cx, x)
        return x

    def gen_named_parameters(self):
        for (i, m) in enumerate(self.modules):
            for (k, p) in m.gen_named_parameters():
                yield (f'{i}.{k}', p)

class SGD(object):
    def __init__(self, parameters):
        self.parameters = list(parameters)

    def step(self, px, grad, lr):
        new_values = px.clone()
        for p in self.parameters:
            new_values[p] = px[p] - grad[p] * lr
        return new_values

def square(x):
    return x*x

# Now implement a xor solver:

class MLP(Module):
    def __init__(self):
        self.layers = Sequential(Linear(2, 3),
                                 Tanh(),
                                 # Dropout(0.9),
                                 Linear(3, 1),
                                 Tanh(),
                                 Linear(1,1))

    def forward(self, cx, x):
        return self.layers(cx, x)

model = MLP()

# XOR
data = [
    ([0, 0], 0),
    ([1, 0], 1),
    ([0, 1], 1),
    ([1, 1], 0)
]

opt = SGD(model.parameters())

rng = PRNG(jax.random.PRNGKey(0))

px = ParamState(model.parameters())
px.initialize(rng.split())
print(model.state_dict(px))

def loss(px, x, y, key):
    cx = Context(px, key)
    return square(model(cx, x) - y).mean()
loss_grad = jax.jit(jax.value_and_grad(loss))

counter = 1
while True:
    xs = []
    ys = []
    for _ in range(1):
        (x, y) = random.choice(data)
        xs.append(jnp.array(x, dtype=jnp.float32))
        ys.append(jnp.array(y, dtype=jnp.float32))
    x = jnp.stack(xs)
    y = jnp.stack(ys)
    v_loss, v_grad = loss_grad(px, x, y, key=rng.split())
    px = opt.step(px, v_grad, lr=0.01)
    if counter % 10 == 0:
        print(counter, v_loss)
    counter += 1