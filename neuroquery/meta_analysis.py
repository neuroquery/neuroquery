"""Implement some coordinate-based meta-analysis."""

import numpy as np
import nibabel as nib
from scipy.ndimage.filters import convolve

from neuroquery.img_utils import _uniform_kernel, coordinates_to_arrays


def _KDA_MKDA(coordinates, r=15, MKDA=False):
    if coordinates.empty:
        raise ValueError('Data frame empty.')

    iter_arrays, masker = coordinates_to_arrays(coordinates)
    affine = masker.mask_img_.affine

    mkda_array = next(iter_arrays)

    n1 = affine[0, 0]
    n2 = affine[1, 1]
    n3 = affine[2, 2]
    kernel = _uniform_kernel(r, n1, n2, n3)

    for array in iter_arrays:
        array = convolve(array, kernel, mode='constant')
        if MKDA:
            array = array > 0
        mkda_array += array

    return nib.Nifti1Image(mkda_array, affine)


def KDA(coordinates, r=15):
    """Implement KDA method.

    Parameters
    ----------
    coordinates : pandas.DataFrame
        Data frame storing the coordintaes.
    r : float
        Radius of the uniform kernel used by MKDA (mm).
        Defaults to 15.

    Returns
    -------
    Niimg-like object
        The meta-analysis map.

    References
    ----------
    .. [1] Tor D. Wager, Martin Lindquist, Lauren Kaplan, Meta-analysis of
    functional neuroimaging data: current and future directions, Social
    Cognitive and Affective Neuroscience, Volume 2, Issue 2, June 2007,
    Pages 150–158, https://doi.org/10.1093/scan/nsm015

    """
    return _KDA_MKDA(coordinates, r, MKDA=False)


def MKDA(coordinates, r=15):
    """Implement MKDA method.

    Parameters
    ----------
    coordinates : pandas.DataFrame
        Data frame storing the coordintaes.

    r : float
        Radius of the uniform kernel used by MKDA (mm).
        Defaults to 15.

    Returns
    -------
    Niimg-like object
        The meta-analysis map.

    References
    ----------
    .. [1] Tor D. Wager, Martin Lindquist, Lauren Kaplan, Meta-analysis of
    functional neuroimaging data: current and future directions, Social
    Cognitive and Affective Neuroscience, Volume 2, Issue 2, June 2007,
    Pages 150–158, https://doi.org/10.1093/scan/nsm015

    """
    return _KDA_MKDA(coordinates, r, MKDA=True)
