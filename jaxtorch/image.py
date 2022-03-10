import math
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from einops import repeat


def factor_int(n: int) -> Tuple[int, int]:
    f1 = int(math.ceil(math.sqrt(n)))
    while n % f1:
        f1 -= 1
    f2 = n // f1
    return min(f1, f2), max(f1, f2)


def compute_channel_change_mat(c_in: int, c_out: int) -> np.ndarray:
    assert max(c_in, c_out) % min(c_in, c_out) == 0
    io_ratio = max(c_in, c_out) // min(c_in, c_out)
    base = np.eye(min(c_in, c_out))
    if c_in < c_out:
        return repeat(base, "d1 d2 -> (d1 r) d2", r=io_ratio)
    elif c_out < c_in:
        # decreasing channel count, average nearby channels
        return repeat(base, "d1 d2 -> d1 (d2 r)", r=io_ratio) / io_ratio
    else:
        return base


upsample_arrays = dict(
    lanczos3=np.array(
        [
            0.0073782638646662235,
            0.030112292617559433,
            -0.06799723953008652,
            -0.13327467441558838,
            0.2710106074810028,
            0.8927707076072693,
            0.8927707672119141,
            0.2710106074810028,
            -0.13327467441558838,
            -0.06799724698066711,
            0.03011229634284973,
            0.007378263399004936,
        ],
    ),
    cubic=np.array(
        [
            -0.0234375,
            -0.0703125,
            0.2265625,
            0.8671875,
            0.8671875,
            0.2265625,
            -0.0703125,
            -0.0234375,
        ],
    ),
    linear=np.array([0.25, 0.75, 0.75, 0.25]),
)


downsample_arrays = dict(
    lanczos3=np.array(
        [
            0.003689131001010537,
            0.015056144446134567,
            -0.03399861603975296,
            -0.066637322306633,
            0.13550527393817902,
            0.44638532400131226,
            0.44638532400131226,
            0.13550527393817902,
            -0.066637322306633,
            -0.03399861603975296,
            0.015056144446134567,
            0.003689131001010537,
        ]
    ),
    cubic=np.array(
        [
            -0.01171875,
            -0.03515625,
            0.11328125,
            0.43359375,
            0.43359375,
            0.11328125,
            -0.03515625,
            -0.01171875,
        ]
    ),
    linear=np.array([0.125, 0.375, 0.375, 0.125]),
)


def upsample_kernel(
    c_in: int,
    c_out: int,
    method: str = "linear",
) -> np.ndarray:
    cmat = compute_channel_change_mat(c_in, c_out)
    kernel = upsample_arrays[method]
    weight = np.einsum("oi,h,w->oihw", cmat, kernel, kernel)
    return weight


def downsample_kernel(
    c_in: int,
    c_out: int,
    method="linear",
) -> np.ndarray:
    cmat = compute_channel_change_mat(c_in, c_out)
    kernel = downsample_arrays[method]
    weight = np.einsum("oi,h,w->oihw", cmat, kernel, kernel)
    return weight


def upsample2x_base(
    img: jnp.ndarray,
    kern: jnp.ndarray,
    format: str = "NCHW",
    norm:bool=True,
):
    ksize = kern.shape[-1]
    kern = jax.lax.convert_element_type(kern, img.dtype)
    out = jax.lax.conv_general_dilated(
        img,
        kern,
        window_strides=[1, 1],
        padding=[(ksize // 2, ksize // 2), (ksize // 2, ksize // 2)],
        lhs_dilation=[2, 2],
        rhs_dilation=None,
        dimension_numbers=(format, "OIHW", format),
    )

    if norm:
        # normalization for parts that touch the zero-padding
        norm = jax.lax.conv_general_dilated(
            jnp.ones([1, *img.shape[-3:]], dtype=img.dtype),
            kern,
            window_strides=[1, 1],
            padding=[(ksize // 2, ksize // 2), (ksize // 2, ksize // 2)],
            lhs_dilation=[2, 2],
            rhs_dilation=None,
            dimension_numbers=(format, "OIHW", format),
        )
        out = out / norm

    return out


def downsample2x_base(
    x: jnp.ndarray,
    kern: jnp.ndarray,
    format: str = "NCHW",
    norm:bool=True,
):
    ksize = kern.shape[-1]
    kern = jax.lax.convert_element_type(kern, x.dtype)
    out = jax.lax.conv_general_dilated(
        x,
        kern,
        window_strides=[2, 2],
        padding=[(ksize // 2 - 1, ksize // 2 - 1), (ksize // 2 - 1, ksize // 2 - 1)],
        lhs_dilation=[1, 1],
        rhs_dilation=None,
        dimension_numbers=(format, "OIHW", format),
    )

    if norm:
        # normalization for parts that touch the zero-padding
        norm = jax.lax.conv_general_dilated(
            jnp.ones([1, *x.shape[-3:]], dtype=x.dtype),
            kern,
            window_strides=[2, 2],
            padding=[
                (ksize // 2 - 1, ksize // 2 - 1),
                (ksize // 2 - 1, ksize // 2 - 1),
            ],
            lhs_dilation=[1, 1],
            rhs_dilation=None,
            dimension_numbers=(format, "OIHW", format),
        )
        out = out / norm

    return out


def upsample2x(
    img: jnp.ndarray,
    c_out: int = None,
    method: str = "linear",
    format: str = "NCHW",
) -> jnp.ndarray:
    c_in = img.shape[-3]
    if c_out is None:
        c_out = c_in
    kern = upsample_kernel(c_in, c_out, method=method)
    kern = jnp.array(kern, dtype=img.dtype)
    return upsample2x_base(img, kern, format)


def downsample2x(
    img: jnp.ndarray,
    c_out: int = None,
    method: str = "linear",
    format: str = "NCHW",
) -> jnp.ndarray:
    c_in = img.shape[-3]
    if c_out is None:
        c_out = c_in
    kern = downsample_kernel(c_in, c_out, method=method)
    kern = jax.lax.convert_element_type(kern, img.dtype)
    return downsample2x_base(img, kern, format)
