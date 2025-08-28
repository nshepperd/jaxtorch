import numpy as np
import jax
import jax.numpy as jnp
import jaxtorch.cbor
import io

def test_cbor():
    array = jax.random.normal(jax.random.PRNGKey(0), (3,4,5))
    roundtrip = jaxtorch.cbor.loads(jaxtorch.cbor.dumps(array))
    assert isinstance(roundtrip, jax.Array)
    np.testing.assert_allclose(array, roundtrip)

def test_pytorch():
    array = jax.random.normal(jax.random.PRNGKey(0), (3,4,5))
    f = io.BytesIO()
    jaxtorch.pt.save(array, f)
    f.seek(0)
    roundtrip = jaxtorch.pt.load(f)
    assert isinstance(roundtrip, jax.Array)
    np.testing.assert_allclose(array, roundtrip)