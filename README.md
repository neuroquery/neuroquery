[![codecov](https://codecov.io/gh/neuroquery/neuroquery/branch/main/graph/badge.svg)](https://codecov.io/gh/neuroquery/neuroquery) 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/neuroquery/neuroquery.git/main?filepath=examples)

# NeuroQuery

NeuroQuery is a tool and a statistical model for meta-analysis of the functional
neuroimaging literature.

Given a text query, it can produce a brain map of the most relevant anatomical
structures according to the current scientific literature.

It can be used through a web interface: https://neuroquery.org

Technical details and extensive validation are provided in [this paper](https://elifesciences.org/articles/53385).

This Python package permits using NeuroQuery offline or integrating it in other
applications. 

## Getting started

[Quick demo](https://nbviewer.jupyter.org/github/neuroquery/neuroquery/blob/main/examples/minimal_example.ipynb)

### Dependencies

NeuroQuery requires Python 3, numpy, scipy, scikit-learn, nilearn, pandas,
regex, lxml, and requests.

nltk is an optional dependency needed only if you use stemming or lemmatization
for tokenization of input text.

python-Levenshtein is an optional dependency used only in some parts of
tokenization. If you use the vocabulary lists provided with `neuroquery` or in
`neuroquery_data` it is not needed.

### Installation

`neuroquery` can be installed with

```
pip install neuroquery
```

### Usage

In the `examples` folder, 
[`minimal_example.ipynb`](https://nbviewer.jupyter.org/github/neuroquery/neuroquery/blob/main/examples/minimal_example.ipynb)
shows basic usage of `neuroquery`.

`neuroquery` has a function to download a trained model so that users can get
started right away:

```python
from neuroquery import fetch_neuroquery_model, NeuroQueryModel
from nilearn.plotting import view_img

encoder = NeuroQueryModel.from_data_dir(fetch_neuroquery_model())
# encoder returns a dictionary containing a brain map and more,
# see examples or documentation for details
view_img(
    encoder("Parkinson's disease")["brain_map"], threshold=3.).open_in_browser()
```

`neuroquery` also provides classes to train new models from scientific
publications' text and stereotactic peak activation coordinates (see
[`training_neuroquery.ipynb`](https://nbviewer.jupyter.org/github/neuroquery/neuroquery/blob/main/examples/training_neuroquery.ipynb)
in the examples).

