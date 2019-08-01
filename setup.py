#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="neuroquery",
    packages=find_packages(),
    package_data={"neuroquery": ["data/*"],
                  "neuroquery.tests": ["data/*"]},
    install_requires=[
        "nilearn",
        "sklearn",
        "pandas",
        "regex",
        "nltk(!=3.2.1)",
        "numpy",
        "scipy",
        "lxml",
        "requests",
    ],
)
