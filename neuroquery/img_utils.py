import numpy as np

from nilearn import image, input_data
from nilearn.datasets import load_mni152_brain_mask


def get_masker(mask_img=None):
    if mask_img is None:
        mask_img = load_mni152_brain_mask()
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
    mask_img = image.load_img(mask_img)
    voxels = coords_to_voxels(coords, mask_img)
    peaks = np.zeros(mask_img.shape)
    np.add.at(peaks, tuple(voxels.T), 1.0)
    peaks_img = image.new_img_like(mask_img, peaks)
    return peaks_img


def gaussian_coord_smoothing(coords, mask_img=None, fwhm=8.0):
    masker = get_masker(mask_img)
    peaks_img = coords_to_peaks_img(coords, mask_img=masker.mask_img_)
    img = image.smooth_img(peaks_img, fwhm=fwhm)
    return masker.inverse_transform(masker.transform(img).squeeze())
