#!/usr/bin/env python

import pathlib
from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

with pathlib.Path("neuroquery", "VERSION.txt").open() as f:
    version = f.read().strip()

url = "https://github.com/neuroquery/neuroquery"

setup(
    name="neuroquery",
    description="Meta-analysis of neuroimaging studies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    maintainer="Jerome Dockes",
    maintainer_email="jerome@dockes.org",
    license="BSD 3-Clause License",
    packages=find_packages(),
    package_data={
        "neuroquery": ["VERSION.txt", "data/*"],
        "neuroquery.tests": ["data/*"],
    },
    version=version,
    url=url,
    install_requires=[
        "nilearn",
        "sklearn",
        "pandas",
        "regex",
        "nltk>=3.4.5",
        "numpy",
        "scipy",
        "lxml",
        "requests",
    ],
    python_requires=">=3",
    classifiers=["Programming Language :: Python :: 3"],
)
