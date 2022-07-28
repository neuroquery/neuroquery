import pathlib
from setuptools import setup

version = (
    pathlib.Path(__file__)
    .parent.joinpath("src", "neuroquery", "VERSION.txt")
    .read_text("utf-8")
    .strip()
)

setup(
    package_data={
        "neuroquery": ["VERSION.txt", "data/*"],
        "neuroquery.tests": ["data/*"],
    },
    version=version,
)
