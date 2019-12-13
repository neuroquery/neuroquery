import argparse
import pathlib

import pandas as pd

from neuroquery import img_utils

parser = argparse.ArgumentParser(
    description="Generate brain maps from (x, y, z) MNI coordinates "
    "grouped by pubmed ID",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "coordinates_csv",
    help=".csv file containing the coordinates. Must have columns"
    " 'pmid', 'x', 'y', and 'z'.",
)
parser.add_argument(
    "output_directory", help="directory where generated maps are saved."
)
parser.add_argument(
    "--fwhm", type=float, default=8.0, help="full width at half maximum"
)
parser.add_argument(
    "--resolution",
    type=float,
    default=4.0,
    help="resolution of created images in mm",
)
args = parser.parse_args()

out_dir = pathlib.Path(args.output_directory)
out_dir.mkdir(parents=True, exist_ok=True)

coordinates = pd.read_csv(args.coordinates_csv)
for pmid, img in img_utils.iter_coordinates_to_maps(
    coordinates, target_affine=args.resolution
):
    img_file = img.to_filename(str(out_dir / "pmid_{}.nii.gz".format(pmid)))

print("\n")
print(out_dir)
