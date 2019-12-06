import numpy as np
import pandas as pd

from nilearn import image, input_data
from nilearn.datasets import load_mni152_brain_mask


def get_masker(mask_img=None, target_affine=None):
    if isinstance(mask_img, input_data.NiftiMasker):
        return mask_img
    if mask_img is None:
        mask_img = load_mni152_brain_mask()
    if target_affine is not None:
        if np.ndim(target_affine) == 0:
            target_affine = np.eye(3) * target_affine
        elif np.ndim(target_affine) == 1:
            target_affine = np.diag(target_affine)
        mask_img = image.resample_img(
            mask_img, target_affine=target_affine, interpolation="nearest"
        )
    masker = input_data.NiftiMasker(mask_img=mask_img).fit()
    return masker


def coords_to_voxels(coords, ref_img=None):
    if ref_img is None:
        ref_img = load_mni152_brain_mask()
    affine = ref_img.affine
    coords = np.atleast_2d(coords)
    coords = np.hstack([coords, np.ones((len(coords), 1))])
    voxels = np.linalg.pinv(affine).dot(coords.T)[:-1].T
    voxels = voxels[(voxels >= 0).all(axis=1)]
    voxels = voxels[(voxels < ref_img.shape[:3]).all(axis=1)]
    voxels = np.floor(voxels).astype(int)
    return voxels


def coords_to_peaks_img(coords, mask_img):
    peaks = coords_to_peaks_array(coords, mask_img)
    peaks_img = image.new_img_like(mask_img, peaks)
    return peaks_img


def coords_to_peaks_array(coords, mask_img):
    mask_img = image.load_img(mask_img)
    voxels = coords_to_voxels(coords, mask_img)
    peaks = np.zeros(mask_img.shape)
    np.add.at(peaks, tuple(voxels.T), 1.0)
    return peaks


def gaussian_coord_smoothing(
    coords, mask_img=None, target_affine=None, fwhm=9.0
):
    masker = get_masker(mask_img, target_affine)
    peaks_img = coords_to_peaks_img(coords, mask_img=masker.mask_img_)
    img = image.smooth_img(peaks_img, fwhm=fwhm)
    return masker.inverse_transform(masker.transform(img).squeeze())


def coordinates_to_maps(
    coordinates, mask_img=None, target_affine=(4, 4, 4), fwhm=9.0
):
    print(
        "Transforming {} coordinates for {} articles".format(
            coordinates.shape[0], len(set(coordinates["pmid"]))
        )
    )
    masker = get_masker(mask_img=mask_img, target_affine=target_affine)
    images, img_pmids = [], []
    for pmid, img in iter_coordinates_to_maps(
        coordinates, mask_img=masker, fwhm=fwhm
    ):
        images.append(masker.transform(img).ravel())
        img_pmids.append(pmid)
    return pd.DataFrame(images, index=img_pmids), masker


def iter_coordinates_to_maps(
    coordinates, mask_img=None, target_affine=(4, 4, 4), fwhm=9.0
):
    masker = get_masker(mask_img=mask_img, target_affine=target_affine)
    articles = coordinates.groupby("pmid")
    for i, (pmid, coord) in enumerate(articles):
        print(
            "{:.1%} pmid: {:< 20}".format(i / len(articles), pmid),
            end="\r",
            flush=True,
        )
        img = gaussian_coord_smoothing(
            coord.loc[:, ["x", "y", "z"]].values, fwhm=fwhm, mask_img=masker
        )
        yield pmid, img


def iter_coordinates_to_arrays(
    coordinates, mask_img=None, target_affine=(4, 4, 4)
):
    masker = get_masker(mask_img=mask_img, target_affine=target_affine)
    articles = coordinates.groupby("pmid")
    for i, (pmid, coord) in enumerate(articles):
        print(
            "{:.1%} pmid: {:< 20}".format(i / len(articles), pmid),
            end="\r",
            flush=True,
        )
        array = coords_to_peaks_array(
            coord.loc[:, ["x", "y", "z"]].values, mask_img=masker.mask_img_
        )
        yield array

def coordinates_to_arrays(
    coordinates, mask_img=None, target_affine=(4, 4, 4)
):
    masker = get_masker(mask_img=mask_img, target_affine=target_affine)
    return iter_coordinates_to_arrays(coordinates, masker), masker


def _uniform_kernel(r, n1=1, n2=1, n3=1):
    """Build an uniform 3D kernel.

    Parameters
    ----------
    r : int
        Sphere radius >= 1.
    n1 : int
        Normalization factor on the 1st axis.
    n2 : int
        Normalization factor on the 2nd axis.
    n3 : int
        Normalization factor on the 3rd axis.

    Returns
    -------
    array like
        Array of shape (r//n1, r//n2, r//n3) storing the kernel.

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
