import numpy as np

from nilearn import image, input_data, datasets


def get_masker():
    mask_img = image.resample_img(
        datasets.load_mni152_brain_mask(),
        target_affine=np.diag((4, 4, 4)),
        interpolation="nearest",
    )
    masker = input_data.NiftiMasker(mask_img=mask_img).fit()
    return masker
