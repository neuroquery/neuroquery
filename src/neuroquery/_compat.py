from nilearn import datasets

try:
    from nilearn import maskers
except ImportError:
    from nilearn import input_data as maskers


def load_mni152_brain_mask():
    try:
        return datasets.load_mni152_brain_mask(2)
    except TypeError:
        return datasets.load_mni152_brain_mask()
