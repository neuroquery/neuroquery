[![Build Status](https://travis-ci.com/neuroquery/neuroquery.svg?branch=master)](https://travis-ci.com/neuroquery/neuroquery) [![codecov](https://codecov.io/gh/neuroquery/neuroquery/branch/master/graph/badge.svg)](https://codecov.io/gh/neuroquery/neuroquery)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/neuroquery/neuroquery.git/master?filepath=examples%2Fplot_using_trained_model.ipynb)

# NeuroQuery

NeuroQuery is a tool and a statistical model for meta-analysis of the functional
neuroimaging literature.

Given a text query, it can produce a brain map of the most relevant anatomical
structures according to the current scientific literature.

It can be used through a web interface: https://neuroquery.saclay.inria.fr

This Python package permits using NeuroQuery offline or integrating it in other
applications. 

## Installation and usage

`neuroquery` can be installed with

```
pip install git+https://github.com/neuroquery/neuroquery.git
```

In the `examples` folder, `plot_using_trained_model.py` shows basic
usage of `neuroquery`.

`neuroquery` has a function to download a trained model so that users can get
started right away:

```python
from neuroquery import datasets, text_to_brain
from nilearn.plotting import view_img

encoder = text_to_brain.TextToBrain.from_data_dir(
    datasets.fetch_neuroquery_model())
# encoder returns a dictionary containing a brain map and more,
# see examples or documentation for details
view_img(
    encoder("Parkinson's disease")["z_map"], threshold=3.0).open_in_browser()
```

`neuroquery` also provides classes to train new models from scientific
publications' text and stereotactic peak activation coordinates.
