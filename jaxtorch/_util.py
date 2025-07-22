import jax
import jaxlib

def ArrayTypes():
    array_types = []
    try:
        from jaxlib.xla_extension import ArrayImpl
        array_types.append(ArrayImpl)
    except ImportError:
        pass
    
    try:
        from jaxlib.xla_extension import DeviceArrayBase
        array_types.append(DeviceArrayBase)
    except ImportError:
        pass
    
    try:
        from jax.interpreters.xla import DeviceArray
        array_types.append(DeviceArray)
    except ImportError:
        pass
    
    try:
        from jax import Array
        array_types.append(Array)
    except ImportError:
        pass

    return tuple(array_types)
ArrayTypes = ArrayTypes()
