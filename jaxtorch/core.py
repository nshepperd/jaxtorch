import jax
import jaxlib
import numpy as np
import functools
import jaxtorch.monkeypatches

def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s

class Param(object):
    """Represents a parameter of a Module, and specifies its shape and initialization."""
    def __init__(self, shape, initializer, desc=None):
        self.shape = shape
        self.initializer = initializer
        self.desc = desc
        self.name = None

    def initialize(self, key):
        return self.initializer(key=key, shape=self.shape)

    def __repr__(self):
        if self.name is not None:
            return f'Param({self.name})'
        elif self.desc:
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
        self.values = {p : None for p in self.parameters}

    def initialize(self, key):
        for par in self.parameters:
            (key, subkey) = jax.random.split(key)
            self.values[par] = par.initialize(key=key)

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
            if self.values[par] is None:
                raise KeyError('Attempted to access uninitialized parameter:', par)
            return self.values[par]
        else:
            raise TypeError('Expected a Param for indexing into ParamState')

    def __setitem__(self, par, v):
        if isinstance(par, Param):
            self.values[par] = v
        else:
            raise TypeError('Expected a Param for indexing into ParamState')

    @staticmethod
    def flatten(px):
        return ([{id(par): val for (par, val) in px.values.items()}], px.parameters)

    @staticmethod
    def unflatten(aux, values):
        px = ParamState(aux)
        px.values = {par : values[0][id(par)] for par in aux}
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

    def self_named_modules(self):
        """Yields a sequence of (str, Module) for direct children of this
        module. May be overridden.

        """
        for (name, val) in self.__dict__.items():
            if isinstance(val, Module):
                yield (name, val)

    def self_named_parameters(self):
        """Yields a sequence of (str, Param) for direct children of this
        module. May be overridden.

        """
        for (name, val) in self.__dict__.items():
            if isinstance(val, Param):
                yield (name, val)

    def labeled_parameters_(self):
        for (name, par) in self.named_parameters():
            par.name = name
        return self.parameters()

    def gen_named_modules(self):
        "Yields (str, Module) for all descendants of this module."
        for (name, val) in self.self_named_modules():
            yield (name, val)
            for (k, v) in val.gen_named_modules():
                yield (name+'.'+k, v)

    def gen_named_parameters(self):
        "Yields (str, Param) for this module and all descendants."
        for (name, par) in self.self_named_parameters():
            yield (name, par)

        for (name, mod) in self.self_named_modules():
            for (k, v) in mod.gen_named_parameters():
                yield (name+'.'+k, v)

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
                if px[p].shape == state[k].shape:
                    px[p] = jax.numpy.asarray(state[k])
                else:
                    print(f'Not loading parameter from incompatible shape: {k} ({px[p].shape} vs {state[k].shape})')
            else:
                print(f'Not loading missing parameter: {k}')


    def _get_name(self):
        return self.__class__.__name__

    def extra_repr(self) -> str:
        r"""Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ''

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self.__dict__.items():
            if isinstance(module, Module):
                mod_str = repr(module)
                mod_str = _addindent(mod_str, 2)
                child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str
