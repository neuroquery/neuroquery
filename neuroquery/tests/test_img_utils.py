import numpy as np

from neuroquery import _img_utils


def test_get_masker():
    masker = _img_utils.get_masker()
    assert np.allclose(masker.mask_img_.affine[:3, :3], 4 * np.eye(3))
