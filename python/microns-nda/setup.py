#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, '..', 'version.py')) as f:
    exec(f.read())

def find_config_package(name):
    return f"{name} @ file://localhost/{here}/../{name}#egg={name}"

config_package = find_config_package('microns-nda-config')

setup(
    name='microns-nda',
    version=__version__,
    description='Neural data access for MICrONS',
    author='Stelios Papadopoulos, Zhuokun Ding',
    author_email='spapadop@bcm.edu, zhuokund@bcm.edu',
    packages=find_packages(exclude=[]),
    install_requires=['numpy', 'scipy', 'tqdm', 'pandas', 'seaborn', 'matplotlib', config_package]
)