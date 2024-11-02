# Jaxtorch (a jax nn library)

This is my jax based nn library. I created this because I was annoyed
by the complexity and 'magic'-ness of the popular jax frameworks
(flax, haiku).

The objective is to enable pytorch-like model definition and training
with a minimum of magic. See the [demo notebook](demo.ipynb) for a simple example.

## Installation

```bash
pip install jaxtorch==0.4.0
```

In applications it's recommended to pin the version as jaxtorch is still in pre-alpha and there will probably be breaking changes in new versions as I work out the API.
