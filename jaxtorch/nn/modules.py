import math
import jax
import jax.numpy as jnp
import jaxtorch
import numbers
from jaxtorch.core import Module, PRNG, Context
from jaxtorch import init


class Identity(Module):
    def forward(self, cx, x):
        return x


class ModuleList(Module):
    def __init__(self, *modules):
        self.modules = []
        for item in modules:
            if isinstance(item, Module):
                self.modules.append(item)
            elif isinstance(item, (list, tuple)):
                self.modules.extend(item)
            else:
                raise ValueError("Expected module or sequence to ModuleList()")

    def __iter__(self):
        return iter(self.modules)

    def append(self, mod):
        self.modules.append(mod)

    def forward(self, cx, x):
        raise NotImplementedError

    def self_named_modules(self):
        for (i, m) in enumerate(self.modules):
            yield (f'{i}', m)


class Sequential(ModuleList):
    def forward(self, cx, x, *args, **kwargs):
        for module in self.modules:
            x = module(cx, x, *args, **kwargs)
        return x


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_init=None,
        bias_init=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        k = math.sqrt(1 / in_features)
        if weight_init:
            self.weight = weight_init(out_features, in_features)
        else:
            self.weight = init.uniform(out_features, in_features, min=-k, max=k)
        if bias:
            if bias_init:
                self.bias = bias_init(out_features)
            else:
                self.bias = init.uniform(out_features, min=-k, max=k)
        else:
            self.bias = None

    def forward(self, cx, x):
        x, weight = cx.policy.cast_to_compute((x, cx[self.weight]))
        bias = cx.policy.cast_to_compute(cx[self.bias]) if self.bias else None
        y = x @ jnp.transpose(weight)
        if self.bias:
            y = y + bias
        return y

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = init.normal(num_embeddings, embedding_dim)

    def forward(self, cx, x):
        return cx[self.weight][x]

    def extra_repr(self) -> str:
        s = '{num_embeddings}, {embedding_dim}'
        # if self.padding_idx is not None:
        #     s += ', padding_idx={padding_idx}'
        # if self.max_norm is not None:
        #     s += ', max_norm={max_norm}'
        # if self.norm_type != 2:
        #     s += ', norm_type={norm_type}'
        # if self.scale_grad_by_freq is not False:
        #     s += ', scale_grad_by_freq={scale_grad_by_freq}'
        # if self.sparse is not False:
        #     s += ', sparse=True'
        return s.format(**self.__dict__)



class Tanh(Module):
    def forward(self, cx, x):
        return jnp.tanh(x)


class Dropout(Module):
    def __init__(self, p=0.5):
        self.rate = p

    def forward(self, cx, x):
        if cx.mode == 'eval':
            return x
        mask = cx.random.bernoulli(1.0 - self.rate, shape=x.shape)
        return x * mask / (1.0 - self.rate)

class Dropout2d(Module):
    def __init__(self, p=0.5):
        self.rate = p

    def forward(self, cx, x):
        if cx.mode == 'eval':
            return x
        drop_shape = x.shape[:2] + (1,) * len(x.shape[2:])
        mask = cx.random.bernoulli(1.0 - self.rate, shape=drop_shape)
        return x * mask / (1.0 - self.rate)

class Sigmoid(Module):
    def forward(self, cx, x):
        return jax.nn.sigmoid(x)

class GELU(Module):
    def forward(self, cx, x):
        return jax.nn.gelu(x)

class ReLU(Module):
    def forward(self, cx, x):
        return jax.nn.relu(x)

class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        self.negative_slope = negative_slope

    def forward(self, cx, x):
        return jax.nn.leaky_relu(x, self.negative_slope)


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = init.ones(*normalized_shape)
            self.bias = init.zeros(*normalized_shape)
        else:
            self.weight = None
            self.bias = None
        self.axes = tuple(-i for i in range(1, len(normalized_shape)+1))

    def forward(self, cx, x):
        x = cx.policy.cast_to_compute(x)
        if self.weight:
            weight = cx[self.weight]
        else:
            weight = 1
        if self.bias:
            bias = cx[self.bias]
        else:
            bias = 0
        mu = x.mean(axis=self.axes, keepdims=True)
        sigma = jnp.sqrt((x - mu).square().mean(axis=self.axes, keepdims=True) + self.eps)
        normed = (x - mu) / sigma
        return weight * normed + bias


class Conv1d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        zero_init=False,
        weight_init=None,
        bias_init=None,
    ):
        assert in_channels % groups == 0
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        k = math.sqrt(groups / (in_channels * kernel_size))
        if zero_init:
            self.weight = init.zeros(
                out_channels,
                in_channels // groups,
                kernel_size,
            )
        elif weight_init:
            self.weight = weight_init(
                out_channels,
                in_channels // groups,
                kernel_size,
            )
        else:
            self.weight = init.uniform(
                out_channels,
                in_channels // groups,
                kernel_size,
                min=-k,
                max=k,
            )
        self.use_bias = bias
        if self.use_bias:
            if bias_init:
                self.bias = bias_init(out_channels)
            else:
                self.bias = init.uniform(out_channels, min=-k, max=k)
        else:
            self.bias = None

    def forward(self, cx, x):
        return jaxtorch.nn.functional.conv1d(
            x,
            cx[self.weight],
            self.bias and cx[self.bias],
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class Conv2d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        zero_init=False,
        weight_init=None,
        bias_init=None,
    ):
        assert in_channels % groups == 0
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        k = math.sqrt(groups / (in_channels * kernel_size * kernel_size))
        if zero_init:
            self.weight = init.zeros(
                out_channels, in_channels // groups, kernel_size, kernel_size
            )
        elif weight_init:
            self.weight = weight_init(
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        else:
            self.weight = init.uniform(
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
                min=-k,
                max=k,
            )
        self.use_bias = bias
        if self.use_bias:
            if bias_init:
                self.bias = bias_init(out_channels)
            else:
                self.bias = init.uniform(out_channels, min=-k, max=k)
        else:
            self.bias = None

    def forward(self, cx, x):
        out = jaxtorch.nn.functional.conv2d(
            x,
            cx[self.weight],
            self.bias and cx[self.bias],
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return out


class SiLU(Module):
    def forward(self, cx, x):
        return jax.nn.silu(x)

class GroupNorm(Module):
    def __init__(
        self,
        num_groups,
        num_channels,
        eps=1e-05,
        affine=True,
        weight_init=None,
        bias_init=None,
    ):
        self.num_groups = num_groups
        self.num_channels = num_channels
        assert self.num_channels % self.num_groups == 0
        self.eps = eps
        self.affine = affine
        if self.affine:
            if weight_init:
                self.weight = weight_init(num_channels)
            else:
                self.weight = init.ones(num_channels)
            if bias_init:
                self.bias = bias_init(num_channels)
            else:
                self.bias = init.zeros(num_channels)
        else:
            self.weight = None
            self.bias = None

    def forward(self, cx, x):
        B, C, *rest = x.shape
        assert C == self.num_channels
        x = x.reshape([B, self.num_groups, C//self.num_groups, *rest])
        mu = x.mean(axis=tuple(range(2,len(x.shape))), keepdims=True)
        var = x.var(axis=tuple(range(2,len(x.shape))), keepdims=True)
        y = (x - mu) / jnp.sqrt(var + self.eps)
        y = y.reshape([B, C, *rest])
        if self.affine:
            broadcast_shape = [self.num_channels] + [1] * len(rest)
            weight = cx[self.weight].reshape(broadcast_shape)
            bias = cx[self.bias].reshape(broadcast_shape)
            y = y * weight + bias
        return y

class PixelUnshuffle(Module):
    def __init__(self, downscale_factor):
        self.downscale_factor = downscale_factor
    def forward(self, cx, x):
        return x.rearrange('... c (h r) (w s) -> ... (c r s) h w', r = self.downscale_factor, s = self.downscale_factor)

class PixelShuffle(Module):
    def __init__(self, upscale_factor):
        self.upscale_factor = upscale_factor
    def forward(self, cx, x):
        return x.rearrange('... (c r s) h w -> ... c (h r) (w s)', r = self.upscale_factor, s = self.upscale_factor)