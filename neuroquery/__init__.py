"""
NeuroQuery
----------

Statistical learning package to learn to map text onto brain activity with
meta-analysis of the neuroimaging literature.

"""

from neuroquery.encoding import NeuroQueryModel
from neuroquery.datasets import fetch_neuroquery_model

__all__ = [
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
