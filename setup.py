from setuptools import find_packages, setup
import os
import pathlib
import pkg_resources

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

with open('jaxtorch/_version.py') as fp:
    exec(fp.read())

# This call to setup() does all the work
setup(
    name="jaxtorch",
    version=__version__,
    description="A jax based nn library",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/nshepperd/jaxtorch",
    author="Emily Shepperd",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(include=['jaxtorch*']),
    include_package_data=True,
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    extras_require={'dev': []},
)
