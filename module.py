import jax

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

class Module(object):
    def __call__(self, cx, *args, **kwargs):
        return self.forward(cx, *args, **kwargs)

    def forward(self, cx, *args, **kwargs):
        """Implements the forward pass. Must take cx as the first argument."""
        raise NotImplementedError

    def gen_named_parameters(self):
        """Returns a generator that yields a sequence of (str, Param) for this
and all children. May be overriden."""
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

    def ctx(self):
        return Context(self.parameters())

    def state_dict(self, cx):
        state = {}
        for (k, p) in self.gen_named_parameters():
            state[k] = cx[p]
        return state

    def load_state_dict(self, cx, state):
        for (k, p) in self.gen_named_parameters():
            if k in state:
                cx[p] = state[k]
            else:
                print(f'Not loading missing parameter: {k}')


class Context(object):
    """Basically just a dictionary of tensors identified by a Param."""
    def __init__(self, parameters):
        self.parameters = parameters
        self.values = {id(p) : None for p in self.parameters}

    def initialize(self, key):
        for par in self.parameters:
            (key, subkey) = jax.random.split(key)
            self.values[id(par)] = par.initialize(key=key)

    def clone(self):
        cx = Context(self.parameters)
        cx.values = dict(self.values)
        return cx

    def __getitem__(self, par):
        return self.values[id(par)]

    def __setitem__(self, par, v):
        self.values[id(par)] = v

    @staticmethod
    def flatten(cx):
        return ([cx.values], cx.parameters)

    @staticmethod
    def unflatten(parameters, values):
        cx = Context(parameters)
        cx.values = dict(values[0])
        return cx

jax.tree_util.register_pytree_node(
    Context,
    Context.flatten,
    Context.unflatten,
)
