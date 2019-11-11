"""Implement some coordinate-based meta-analysis."""

import numpy as np
import nibabel as nib

from scipy.ndimage.filters import convolve


def _uniform_kernel(r, n1=1, n2=1, n3=1):
    """Build an uniform 3D kernel.

    Args:
        r (int): Sphere radius.
        n1 (int): Normalization factor on the 1st axis.
        n2 (int): Normalization factor on the 2nd axis.
        n3 (int): Normalization factor on the 3rd axis.

    Returns:
        (array like) Array of shape (r//n1, r//n2, r//n3) storing the kernel.

    """
    A, B, C = r, r, r

    a = int(A/abs(n1))
    b = int(B/abs(n2))
    c = int(C/abs(n3))

    kernel = np.zeros((2*a+1, 2*b+1, 2*c+1))

    i0, j0, k0 = a, b, c

    for i in range(2*a+1):
        for j in range(2*b+1):
            for k in range(2*c+1):
                if ((i-i0)/a)**2 + ((j-j0)/b)**2 + ((k-k0)/c)**2 <= 1:
                    kernel[i, j, k] = 1

    return kernel


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

    Args:
        peaks_arrs (list): List or generator of 3D arrays. Each array
            stores the activation peaks of one study. Each array is assumed to
            be of the same shape.
        affine (array): Affine shared by the arrays.
        r (float): Radius of the uniform kernel used by MKDA (mm).
            Defaults to 15.

    Returns:
        (Niimg-like object): Meta-analysis map

    """
    return _KDA_MKDA(peaks_arrs, affine, r, MKDA=False)


def MKDA(peaks_arrs, affine, r=15):
    """Implement MKDA method.

    Args:
        peaks_arrs (list): List or generator of 3D arrays. Each array
            stores the activation peaks of one study. Each array is assumed to
            be of the same shape.
        affine (array): Affine shared by the arrays.
        r (float): Radius of the uniform kernel used by MKDA (mm).
            Defaults to 15.

    Returns:
        (Niimg-like object): Meta-analysis map

    """
    return _KDA_MKDA(peaks_arrs, affine, r, MKDA=True)
