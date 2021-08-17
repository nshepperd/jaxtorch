# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['jaxtorch', 'jaxtorch.nn']

package_data = \
{'': ['*']}

install_requires = \
['cbor2>=5.4.1,<6.0.0', 'einops>=0.3.0,<0.4.0']

setup_kwargs = {
    'name': 'jaxtorch',
    'version': '0.1.0',
    'description': 'PyTorch-style interface for JAX',
    'long_description': None,
    'author': 'nshepperd',
    'author_email': 'nshepperd@gmail.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.7,<4.0',
}


setup(**setup_kwargs)
