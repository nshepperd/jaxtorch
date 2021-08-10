import jax
import jax.numpy as jnp
from jaxtorch.core import Module, PRNG, Context, ParamState
from jaxtorch import init

def square(x):
    return x**2

class Linear(Module):
    def __init__(self, c1, c2, bias=True):
        self.c1 = c1
        self.c2 = c2
        self.weight = init.normal(c1, c2)
        if bias:
            self.bias = init.zeros(c2)
        else:
            self.bias = None

    def __repr__(self):
        return f'Linear({self.c1}, {self.c2})'

    def forward(self, cx, x):
        y = x @ cx[self.weight]
        if self.bias:
            y = y + cx[self.bias]
        return y

class Embedding(Module):
    def __init__(self, n, c):
        self.n = n
        self.c = c
        self.weight = init.normal(n, c)

    def __repr__(self):
        return f'Embedding({self.n}, {self.c})'

    def forward(self, cx, x):
        return cx[self.weight][x]


class Tanh(Module):
    def forward(self, cx, x):
        return jnp.tanh(x)


class Dropout(Module):
  def __init__(self, rate=0.5):
    self.rate = rate

  def forward(self, cx, x):
    key = cx.rng.split()
    p = jax.random.bernoulli(key, 1.0 - self.rate, shape=x.shape)
    return x * p / (1.0 - self.rate)

class GELU(Module):
    def forward(self, cx, x):
        return jax.nn.gelu(x)

class LayerNorm(Module):
    def __init__(self, *normalized_shape):
        self.normalized_shape = normalized_shape
        self.weight = init.ones(*normalized_shape)
        self.bias = init.zeros(*normalized_shape)
        self.axes = tuple(-i for i in range(1, len(normalized_shape)+1))

    def forward(self, cx, x):
        mu = x.mean(axis=self.axes, keepdims=True)
        sigma = jnp.sqrt(square(x - mu).mean(axis=self.axes, keepdims=True))
        normed = (x - mu) / sigma
        return cx[self.weight] * normed + cx[self.bias]

class Sequential(Module):
    def __init__(self, *modules):
        self.modules = modules

    def forward(self, cx, x):
        for module in self.modules:
            x = module(cx, x)
        return x

    def gen_named_modules(self):
        for (i, m) in enumerate(self.modules):
            yield (f'{i}', m)
            for (k, p) in m.gen_named_modules():
                yield (f'{i}.{k}', p)

    def gen_named_parameters(self):
        for (i, m) in enumerate(self.modules):
            for (k, p) in m.gen_named_parameters():
                yield (f'{i}.{k}', p)
