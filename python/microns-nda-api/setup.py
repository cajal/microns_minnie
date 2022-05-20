#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, '..', 'version.py')) as f:
    exec(f.read())

with open(path.join(here, 'requirements.txt')) as f:
    requirements = f.read().split()

setup(
    name="microns-nda-api",
    version=__version__,
    description="api for microns-nda",
    author="Stelios Papadopoulos, Christos Papadopoulos",
    packages=find_packages(),
    install_requires=requirements
)