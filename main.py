import jax
import jax.numpy as jnp
import random

from module import Module
import init

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

    def step(self, cx, grad, lr):
        new_values = cx.clone()
        for p in self.parameters:
            new_values[p] = cx[p] - grad[p] * lr
        return new_values

model = Sequential(Linear(2, 3), Tanh(), Linear(3, 1), Tanh(), Linear(1,1))

def square(x):
    return x*x

def loss(cx, x, y):
    return square(model(cx, x) - y).mean()
loss_grad = jax.value_and_grad(loss)

cx = model.ctx()
cx.initialize(jax.random.PRNGKey(0))
print(model.state_dict(cx))

# XOR
data = [
    ([0, 0], 0),
    ([1, 0], 1),
    ([0, 1], 1),
    ([1, 1], 0)
]

opt = SGD(model.parameters())

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
    v_loss, v_grad = loss_grad(cx, x, y)
    cx = opt.step(cx, v_grad, lr=0.01)
    print(counter, v_loss)
    counter += 1