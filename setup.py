#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name="neuroquery",
    packages=["neuroquery"],
    package_data={"neuroquery": ["data/*"]},
    install_requires=[
        "nilearn",
        "sklearn",
        "pandas",
        "regex",
        "nltk(!=3.2.1)",
        "numpy",
        "scipy",
        "lxml",
    ],
)
