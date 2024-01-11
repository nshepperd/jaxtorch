import jax
import jaxlib

def ArrayTypes():
    array_types = []
    if hasattr(jaxlib.xla_extension, 'ArrayImpl'):
        array_types.append(jaxlib.xla_extension.ArrayImpl)
    if hasattr(jaxlib.xla_extension, 'DeviceArrayBase'):
        array_types.append(jaxlib.xla_extension.DeviceArrayBase)
    if hasattr(jax.interpreters.xla, 'DeviceArray'):
        array_types.append(jax.interpreters.xla.DeviceArray)
    if hasattr(jax, 'Array'):
        array_types.append(jax.Array)
    return tuple(array_types)
ArrayTypes = ArrayTypes()
