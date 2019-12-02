"""Implement some coordinate-based meta-analysis."""

import numpy as np
import nibabel as nib
from scipy.ndimage.filters import convolve

from neuroquery.img_utils import _uniform_kernel


def _KDA_MKDA(peaks_arrs, affine, r=15, MKDA=False):
    if not peaks_arrs:
        raise ValueError('No map provided.')

    # Convert to iterator if not already the case
    peaks_arrs = iter(peaks_arrs)

    mkda_arr = next(peaks_arrs)

    n1 = affine[0, 0]
    n2 = affine[1, 1]
    n3 = affine[2, 2]
    kernel = _uniform_kernel(r, n1, n2, n3)

    for peaks_arr in peaks_arrs:
        arr = convolve(peaks_arr, kernel, mode='constant')
        if MKDA:
            arr = arr > 0
        mkda_arr += arr

    return nib.Nifti1Image(mkda_arr, affine)


def KDA(peaks_arrs, affine, r=15):
    """Implement KDA method.

    Parameters
    ----------
        peaks_arrs : list or generator of 3D arrays.
            Each array stores the activation peaks of one study. Each array
            is assumed to be of the same shape.
        affine : array
            Affine shared by the arrays.
        r : float
            Radius of the uniform kernel used by MKDA (mm).
            Defaults to 15.

    Returns
    -------
    Niimg-like object
        The meta-analysis map

    References
    ----------
    .. [1] Tor D. Wager, Martin Lindquist, Lauren Kaplan, Meta-analysis of
    functional neuroimaging data: current and future directions, Social
    Cognitive and Affective Neuroscience, Volume 2, Issue 2, June 2007,
    Pages 150–158, https://doi.org/10.1093/scan/nsm015

    """
    return _KDA_MKDA(peaks_arrs, affine, r, MKDA=False)


def MKDA(peaks_arrs, affine, r=15):
    """Implement MKDA method.

    Parameters
    ----------
        peaks_arrs : list or generator of 3D arrays.
            Each array stores the activation peaks of one study. Each array is assumed to
            be of the same shape.
        affine : array
            Affine shared by the arrays.
        r : float
            Radius of the uniform kernel used by MKDA (mm).
            Defaults to 15.

    Returns
    -------
        Niimg-like object
            The meta-analysis map

    References
    ----------
    .. [1] Tor D. Wager, Martin Lindquist, Lauren Kaplan, Meta-analysis of
    functional neuroimaging data: current and future directions, Social
    Cognitive and Affective Neuroscience, Volume 2, Issue 2, June 2007,
    Pages 150–158, https://doi.org/10.1093/scan/nsm015

    """
    return _KDA_MKDA(peaks_arrs, affine, r, MKDA=True)
