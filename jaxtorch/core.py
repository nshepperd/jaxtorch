from __future__ import annotations
__docformat__ = "google"

import jax
import numpy as np
import functools
import jaxtorch.monkeypatches
import sys
from dataclasses import dataclass
from typing import TypeVar, Generic, Any, Dict
from collections import OrderedDict
import warnings


def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


class Param(object):
    """Represents a parameter of a Module, and specifies its shape and initialization."""

    def __init__(self, shape, initializer=None):
        self.shape = shape
        self.initializer = initializer
        self.name = None

    def setup(self, cx):
        if self.initializer is not None:
            cx[self] = self.initializer(cx.rng.split())

    def set_name(self, name: str):
        if self.name and not name.endswith(self.name):
            raise ValueError(
                f"Changing name of Param from {self.name} to {name}, this is probably a bug."
            )
        self.name = name

    def __repr__(self):
        if self.name is not None:
            return f"<Param at {self.name} {self.shape}>"
        else:
            return super().__repr__()


class PRNG(object):
    """Just a stateful convenience wrapper for a jax PRNGKey."""

    key: jax.Array

    def __init__(self, key):
        self.key = key

    def split(self, n=None) -> jax.Array:
        """Produce a new PRNGKey by splitting the current key.
        Optionally, split it into n subkeys before returning it.
        Example:
            >>> rng = PRNG(jax.random.PRNGKey(0))
            >>> noise = jax.random.normal(rng.split())
        """
        (self.key, subkey) = jax.random.split(self.key)
        if n is not None:
            subkey = jax.random.split(subkey, n)
        return subkey


class ContextRandom(object):
    """Lives inside a Context and provides convenience functions for
    random number generation that use the Context's stateful PRNG.

    """
    _rng: PRNG
    def __init__(self, rng: PRNG):
        self._rng = rng

    def _wrap(f):
        @functools.wraps(f)
        def wrapped(self, *args, **kwargs):
            return f(self._rng.split(), *args, **kwargs)

        return wrapped

    bernoulli = _wrap(jax.random.bernoulli)
    beta = _wrap(jax.random.beta)
    categorical = _wrap(jax.random.categorical)
    cauchy = _wrap(jax.random.cauchy)
    choice = _wrap(jax.random.choice)
    dirichlet = _wrap(jax.random.dirichlet)
    double_sided_maxwell = _wrap(jax.random.double_sided_maxwell)
    exponential = _wrap(jax.random.exponential)
    gamma = _wrap(jax.random.gamma)
    gumbel = _wrap(jax.random.gumbel)
    laplace = _wrap(jax.random.laplace)
    logistic = _wrap(jax.random.logistic)
    maxwell = _wrap(jax.random.maxwell)
    multivariate_normal = _wrap(jax.random.multivariate_normal)
    normal = _wrap(jax.random.normal)
    pareto = _wrap(jax.random.pareto)
    permutation = _wrap(jax.random.permutation)
    poisson = _wrap(jax.random.poisson)
    rademacher = _wrap(jax.random.rademacher)
    randint = _wrap(jax.random.randint)
    t = _wrap(jax.random.t)
    truncated_normal = _wrap(jax.random.truncated_normal)
    uniform = _wrap(jax.random.uniform)
    weibull_min = _wrap(jax.random.weibull_min)


class Context(object):
    """Wraps parameter dictionary and PRNG for stateful operations.

    Context objects are passed to `Module` forward passes and provide access to:
    - The parameter dictionary storing all model weights
    - A stateful random number generator for stochastic operations
    - The current mode (train/eval) for dropout etc.
    - Optional user data dictionary for custom state

    Args:
        params (dict): Dictionary mapping parameter names to values
        key (jax.Array): JAX random number generator key
        mode (str, optional): Either 'train' or 'eval'
        user (dict, optional): Custom user state dictionary

    Example:

        >>> cx = Context(params, key, mode='train')
        >>> cx.random.normal(shape=(10,))  # Generate random numbers
        >>> y = model(cx, x)  # Forward pass
    """

    params: Dict[str, jax.Array]
    rng: PRNG
    random: ContextRandom
    mode: str
    user: dict

    def __init__(
        self,
        params: Dict[str, jax.Array],
        key: jax.Array,
        mode: str = "train",
        user: dict = None,
    ):
        self.params = params
        self.rng = PRNG(key)
        self.random = ContextRandom(self.rng)
        self.mode = mode
        self.user = dict(user or {})

    def train_mode_(self):
        self.mode = "train"
        return self

    def eval_mode_(self):
        self.mode = "eval"
        return self

    def __getitem__(self, par):
        if isinstance(par, Param):
            return self.params[par.name]
        elif isinstance(par, str):
            return self.params[par]
        else:
            raise TypeError("Expected a Param for indexing into Context")

    def __setitem__(self, par, value):
        if isinstance(par, Param):
            self.params[par.name] = value
        elif isinstance(par, str):
            self.params[par] = value
        else:
            raise TypeError("Expected a Param for indexing into Context")

    def freeze(self):
        return FrozenContext(self.params, self.rng.key, self.mode, self.tmp)


@jax.tree_util.register_pytree_node_class
@dataclass
class FrozenContext(object):
    """Immutable version of `Context` compatible with jax transformations."""

    params: Dict[str, jax.Array]
    key: jax.Array
    mode: str
    user: dict

    def thaw(self):
        return Context(self.params, self.key, self.mode, self.user)

    def thaw_into(self, cx):
        cx.rng.key = self.key
        cx.params.update(self.params)
        cx.user = dict(self.user)
        return cx

    def tree_flatten(self):
        return (self.params, self.key, self.user), (self.mode,)

    @staticmethod
    def tree_unflatten(static, dynamic):
        (params, key, user) = dynamic
        (mode,) = static
        return FrozenContext(params, key, mode, user)


def transform_with_cx(*transforms):
    def tr(func):
        # Inner wrapped function: pure functional, inputs and outputs frozen context
        def inner(frozen_cx, *args, **kwargs):
            cx = frozen_cx.thaw()
            result = func(cx, *args, **kwargs)
            return cx.freeze(), result

        # Transform pure function
        for t in reversed(transforms):
            inner = t(inner)

        # Wrap the transformed pure function to accept Context
        def outer(cx, *args, **kwargs):
            frozen_cx = cx.freeze()
            frozen_cx, result = inner(frozen_cx, *args, **kwargs)
            frozen_cx.thaw_into(cx)
            return result

        return outer

    return tr


def transform_method_with_cx(*transforms):
    def tr(func):
        # Inner wrapped function: pure functional, inputs and outputs frozen context
        def inner(static_self, frozen_cx, *args, **kwargs):
            self = static_self.value
            cx = frozen_cx.thaw()
            result = func(self, cx, *args, **kwargs)
            return cx.freeze(), result

        # Transform pure function
        for t in reversed(transforms):
            inner = t(inner)

        # Wrap the transformed pure function to accept Context
        @functools.wraps(func)
        def outer(self, cx, *args, **kwargs):
            frozen_cx = cx.freeze()
            frozen_cx, result = inner(Static(self), frozen_cx, *args, **kwargs)
            frozen_cx.thaw_into(cx)
            return result

        return outer

    return tr


class Module(object):
    """Roughly equivalent of pytorch Modules, except parameters are
    stored externally, in a parameter dictionary.

    Params hold the name of the parameter, to make it easy to look
    them up.

    """

    name: str
    _modules: OrderedDict[str, Module]
    _params: OrderedDict[str, Param]

    def __init__(self):
        self.name = None
        self._modules = OrderedDict()
        self._params = OrderedDict()

    def __setattr__(self, name: str, value: Any):
        if isinstance(value, Module):
            if not hasattr(self, "_modules"):
                raise ValueError(
                    f"Must call super().__init__() in {self.__class__.__name__} before adding submodules."
                )
            self._modules[name] = value
            value.set_name(self.name + "." + name if self.name else name)
            super().__setattr__(name, value)
        elif isinstance(value, Param):
            if not hasattr(self, "_params"):
                raise ValueError(
                    f"Must call super().__init__() in {self.__class__.__name__} before adding parameters."
                )
            self._params[name] = value
            value.set_name(self.name + "." + name if self.name else name)
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def set_name(self, name: str):
        if self.name and not name.endswith(self.name):
            raise ValueError(
                f"Changing name of {self.__class__.__name__} from {self.name} to {name}, this is probably a bug."
            )
        self.name = name
        for k, v in self._modules.items():
            v.set_name(name + "." + k if name else k)
        for k, v in self._params.items():
            v.set_name(name + "." + k if name else k)

    def __call__(self, cx: Context, *args, **kwargs):
        return self.forward(cx, *args, **kwargs)

    def forward(self, cx: Context, *args, **kwargs):
        """Implements the forward pass. Must take Context as the first argument."""
        raise NotImplementedError

    def setup(self, cx: Context):
        """Called after the module is created to initialize parameters recursively.

        If you override this, call super().setup(cx) at the start of your implementation.
        """
        for name, mod in self._modules.items():
            mod.setup(cx)
        for name, par in self._params.items():
            par.setup(cx)

        if hasattr(self, "self_init_weights"):
            raise ValueError("self_init_weights is deprecated, override setup instead")
        if hasattr(self, "post_init_weights"):
            warnings.warn(
                "post_init_weights is deprecated, override setup instead",
                DeprecationWarning,
            )
            self.post_init_weights(cx)

    def initialize(self, key: jax.random.PRNGKey) -> dict[str, jax.Array]:
        """Initialize the model.

        You could jit this to control where the parameters are stored, or to avoid
        initializing parameters that are being loaded from state_dict.
        """
        cx = Context({}, key)
        self.setup(cx)
        return cx.params

    def init_weights(self, key):
        return self.initialize(key)

    def name_everything_(self):
        """No-op for compatibility (modules and params are named when assigned now)."""
        warnings.warn(
            "name_everything_ is deprecated, you don't need to call this any more",
            DeprecationWarning,
        )
        pass

    def gen_named_modules(self):
        "Yields (str, Module) for all descendants of this module."
        for name, val in self._modules.items():
            yield (name, val)
            for k, v in val.gen_named_modules():
                yield (name + "." + k, v)

    def gen_named_parameters(self):
        "Yields (str, Param) for this module and all descendants."
        for name, par in self._params.items():
            yield (name, par)

        for name, mod in self._modules.items():
            for k, v in mod.gen_named_parameters():
                yield (name + "." + k, v)

    def named_parameters(self):
        return list(self.gen_named_parameters())

    def named_modules(self):
        return list(self.gen_named_modules())

    def modules(self):
        return [m for (k, m) in self.gen_named_modules()]

    def parameters(self):
        return [p for (k, p) in self.gen_named_parameters()]

    def state_dict(self, params):
        return {name: params[par.name] for (name, par) in self.gen_named_parameters()}

    def load_state_dict(self, params, state, strict=True):
        """Load a previously saved state_dict into params. Returns params."""
        for k, p in self.gen_named_parameters():
            if k not in state:
                if strict:
                    raise ValueError(f"Not loading missing parameter: {k}")
                else:
                    print(f"Not loading missing parameter: {k}", file=sys.stderr)
                    continue

            if tuple(p.shape) != tuple(state[k].shape):
                msg = f"Not loading parameter from incompatible shape: {k} ({p.shape} vs {state[k].shape})"
                if strict:
                    raise ValueError(msg)
                else:
                    print(msg, file=sys.stderr)
                    continue

            params[p.name] = jax.numpy.asarray(state[k])
        return params

    def _get_name(self):
        return self.__class__.__name__

    def extra_repr(self) -> str:
        r"""Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ""

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


T = TypeVar("T")


@jax.tree_util.register_static
@dataclass(frozen=True)
class Static(Generic[T]):
    """Wrapper for a static value that can contain any python object as a pytree node.

    Sometimes more convenient then setting static_argnums= in jax functions.
    """

    value: T
