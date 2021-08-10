import jax
import numpy as np
import functools

class Param(object):
    """Represents a parameter of a Module, and specifies its shape and initialization."""
    def __init__(self, shape, initializer, desc=None):
        self.shape = shape
        self.initializer = initializer
        self.desc = desc

    def initialize(self, key):
        return self.initializer(key=key, shape=self.shape)

    def __repr__(self):
        if self.desc:
            return self.desc
        else:
            return f'Param({self.shape}, {self.initializer})'

class PRNG(object):
    """Just a stateful wrapper for a jax.random.PRNGKey."""
    def __init__(self, key):
        self.key = key
    def split(self):
        (self.key, subkey) = jax.random.split(self.key)
        return subkey

class ParamState(object):
    """Just a dictionary of tensors identified by Param."""
    def __init__(self, parameters):
        self.parameters = parameters
        self.values = {id(p) : None for p in self.parameters}

    def initialize(self, key):
        for par in self.parameters:
            (key, subkey) = jax.random.split(key)
            self.values[id(par)] = par.initialize(key=key)

    def clone(self):
        px = ParamState(self.parameters)
        px.values = dict(self.values)
        return px

    def merge(self, other):
        """Returns the right-biased union of two dictionaries."""
        px = ParamState(list(set(self.parameters) + set(other.parameters)))
        px.values = dict(self.values)
        px.values.update(other.values)
        return px

    def __getitem__(self, par):
        if isinstance(par, Param):
            return self.values[id(par)]
        else:
            raise TypeError('Expected a Param for indexing into ParamState')

    def __setitem__(self, par, v):
        if isinstance(par, Param):
            self.values[id(par)] = v
        else:
            raise TypeError('Expected a Param for indexing into ParamState')

    @staticmethod
    def flatten(px):
        return ([px.values], px.parameters)

    @staticmethod
    def unflatten(aux, values):
        px = ParamState(aux)
        px.values = dict(values[0])
        return px

jax.tree_util.register_pytree_node(
    ParamState,
    ParamState.flatten,
    ParamState.unflatten,
)

class Context(object):
    """Wraps a ParamState and a PRNG."""
    def __init__(self, px, key):
        self.px = px
        self.rng = PRNG(key)

    def __getitem__(self, par):
        if isinstance(par, Param):
            return self.px[par]
        else:
            raise TypeError('Expected a Param for indexing into Context')

class Module(object):
    def __call__(self, cx: Context, *args, **kwargs):
        return self.forward(cx, *args, **kwargs)

    def forward(self, cx: Context, *args, **kwargs):
        """Implements the forward pass. Must take cx as the first argument."""
        raise NotImplementedError

    def gen_named_modules(self):
        """Returns a generator that yields a sequence of (str, Module) for this
        and all children. May be overriden.
        """
        for (name, val) in self.__dict__.items():
            if isinstance(val, Module):
                yield (name, val)
                for (k, v) in val.gen_named_modules():
                    yield (name+'.'+k, v)

    def gen_named_parameters(self):
        """Returns a generator that yields a sequence of (str, Param) for this
        and all children. May be overriden.
        """
        for (name, val) in self.__dict__.items():
            if isinstance(val, Module):
                for (k, v) in val.gen_named_parameters():
                    yield (name+'.'+k, v)
            elif isinstance(val, Param):
                yield (name, val)

    def named_parameters(self):
        return list(self.gen_named_parameters())

    def parameters(self):
        return [p for (k, p) in self.gen_named_parameters()]

    def mkstate(self):
        return ParamState(self.parameters())

    def state_dict(self, px: ParamState):
        state = {}
        for (k, p) in self.gen_named_parameters():
            state[k] = np.array(px[p])
        return state

    def load_state_dict(self, px: ParamState, state):
        for (k, p) in self.gen_named_parameters():
            if k in state:
                px[p] = jax.numpy.asarray(state[k])
            else:
                print(f'Not loading missing parameter: {k}')
