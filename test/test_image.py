import sys
sys.path = ['.'] + sys.path
import numpy as np
import jax
import jax.numpy as jnp
import jaxtorch

def check_near(x, y):
    assert (x - y).abs().max() < 1e-6, x-y

def test_upsample2x():
    image = jax.random.normal(jax.random.PRNGKey(0), (1,1,10,10))
    check_near(jax.image.resize(image, (1,1,20,20), method='linear'), jaxtorch.image.upsample2x(image, method='linear'))
    check_near(jax.image.resize(image, (1,1,20,20), method='cubic'), jaxtorch.image.upsample2x(image, method='cubic'))
    check_near(jax.image.resize(image, (1,1,20,20), method='lanczos3'), jaxtorch.image.upsample2x(image, method='lanczos3'))

def test_downsample2x():
    image = jax.random.normal(jax.random.PRNGKey(0), (1,1,20,20))
    check_near(jax.image.resize(image, (1,1,10,10), method='linear'), jaxtorch.image.downsample2x(image, method='linear'))
    check_near(jax.image.resize(image, (1,1,10,10), method='cubic'), jaxtorch.image.downsample2x(image, method='cubic'))
    check_near(jax.image.resize(image, (1,1,10,10), method='lanczos3'), jaxtorch.image.downsample2x(image, method='lanczos3'))
