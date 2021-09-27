import jax
import jax.numpy as jnp

def upsample2x_base(image, kernel):
    ksize = kernel.shape[0]
    (n, c, h, w) = image.shape
    out = jax.lax.conv_general_dilated(image.reshape(n*c,1,h,w),
                                       kernel.reshape(1,1,ksize,ksize),
                                       window_strides=[1,1],
                                       padding=[(ksize//2,ksize//2),(ksize//2,ksize//2)],
                                       lhs_dilation=[2,2],
                                       rhs_dilation=None,
                                       dimension_numbers=('NCHW',
                                                          'IOHW', 'NCHW'))

    # normalization for parts that touch the zero-padding
    norm = jax.lax.conv_general_dilated(jnp.ones((1,1,h,w)),
                                        kernel.reshape(1,1,ksize,ksize),
                                        window_strides=[1,1],
                                        padding=[(ksize//2,ksize//2),(ksize//2,ksize//2)],
                                        lhs_dilation=[2,2],
                                        rhs_dilation=None,
                                        dimension_numbers=('NCHW',
                                                           'IOHW', 'NCHW'))
    return (out / norm).reshape(n, c, 2*h,2*w)

def upsample2x(image, method='linear'):
    if method == 'lanczos3':
        # extracted from the gradients of jax.image.resize(method='lanczos3')
        kernel = jnp.array([0.0073782638646662235, 0.030112292617559433,
                            -0.06799723953008652, -0.13327467441558838,
                            0.2710106074810028, 0.8927707076072693,
                            0.8927707672119141, 0.2710106074810028,
                            -0.13327467441558838, -0.06799724698066711,
                            0.03011229634284973, 0.007378263399004936])
    elif method == 'cubic':
        # extracted from the gradients of jax.image.resize(method='cubic')
        kernel = jnp.array([-0.0234375, -0.0703125, 0.2265625, 0.8671875, 0.8671875, 0.2265625, -0.0703125, -0.0234375])
    elif method == 'linear':
        # extracted from the gradients of jax.image.resize(method='linear')
        kernel = jnp.array([0.25, 0.75, 0.75, 0.25])
    kernel = kernel.reshape(-1,1) * kernel.reshape(1,-1)

    return upsample2x_base(image, kernel)
