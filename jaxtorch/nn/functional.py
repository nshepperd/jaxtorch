from typing import Sequence, Tuple

import jax


def conv2d(
    input: jax.Array,
    weight: jax.Array,
    bias: jax.Array | None = None,
    stride: int = 1,
    padding: int | Sequence[int] | Sequence[Tuple[int, int]] | str = 0,
    dilation: int = 1,
    groups: int = 1,
):
    (B, C1, iH, iW) = input.shape
    (C2, _, kH, kW) = weight.shape
    assert weight.shape[1] == C1 // groups

    if isinstance(padding, int):
        jax_padding = ((padding, padding), (padding, padding))
    elif isinstance(padding, str):
        jax_padding = padding.upper()
    elif isinstance(padding[0], int) and isinstance(padding[1], int):
        jax_padding = ((padding[0], padding[0]), (padding[1], padding[1]))
    else:
        jax_padding = padding

    if dilation > 1:
        jax_dilation = (dilation, dilation)
    else:
        jax_dilation = None

    output = jax.lax.conv_general_dilated(
        lhs=input,
        rhs=weight,
        window_strides=[stride, stride],
        padding=jax_padding, # type: ignore
        lhs_dilation=None,
        rhs_dilation=jax_dilation,
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        feature_group_count=groups,
    )
    if bias is not None:
        output = output + bias.reshape(bias.shape[0], 1, 1)
    return output


def conv1d(
    input: jax.Array,
    weight: jax.Array,
    bias: jax.Array | None = None,
    stride: int = 1,
    padding: int | Sequence[int] | Sequence[Tuple[int, int]] | str = 0,
    dilation: int = 1,
    groups: int = 1,
):
    (B, C1, iH) = input.shape
    (C2, _, kH) = weight.shape
    assert weight.shape[1] == C1 // groups

    if isinstance(padding, int):
        jax_padding = ((padding, padding),)
    elif isinstance(padding, str):
        jax_padding = padding.upper()
    elif isinstance(padding[0], int):
        jax_padding = ((padding[0], padding[0]),)
    else:
        jax_padding = padding

    if dilation > 1:
        jax_dilation = (dilation,)
    else:
        jax_dilation = None

    output = jax.lax.conv_general_dilated(
        lhs=input,
        rhs=weight,
        window_strides=[stride],
        padding=jax_padding, # type: ignore
        lhs_dilation=None,
        rhs_dilation=jax_dilation,
        dimension_numbers=("NCH", "OIH", "NCH"),
        feature_group_count=groups,
    )
    if bias is not None:
        output = output + bias.reshape(bias.shape[0], 1)
    return output


def normalize(input, p=2.0, dim=1, eps=1e-12):
    if p != 2.0:
        raise NotImplementedError("only p=2.0 implemented so far")
    mag = input.square().sum(axis=dim, keepdims=True).sqrt()
    return input / mag.clamp(minval=eps)
