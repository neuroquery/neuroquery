"""
NeuroQuery
----------

Statistical learning package to learn to map text onto brain activity with
meta-analysis of the neuroimaging literature.

"""
import pathlib

from neuroquery.encoding import NeuroQueryModel
from neuroquery.datasets import fetch_neuroquery_model

with (pathlib.Path(__file__).parent / "VERSION.txt").open() as f:
    __version__ = f.read().strip()

__all__ = [
    "__version__",
    "datasets",
    "encoding",
    "fetch_neuroquery_model",
    "img_utils",
    "NeuroQueryModel",
    "nmf",
    "ridge",
    "smoothed_regression",
    "tokenization",
]
