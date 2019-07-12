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
args = parser.parse_args()

out_dir = pathlib.Path(args.output_directory)
out_dir.mkdir(parents=True, exist_ok=True)

coordinates = pd.read_csv(args.coordinates_csv)
articles = coordinates.groupby("pmid")
for i, (pmid, article_coordinates) in enumerate(articles):
    print(
        "{:.1%} pmid: {:< 20}".format(i / len(articles), pmid),
        end="\r",
        flush=True,
    )
    img_file = out_dir / "pmid_{}.nii.gz".format(pmid)
    if not img_file.is_file():
        img = img_utils.gaussian_coord_smoothing(
            article_coordinates.loc[:, ["x", "y", "z"]].values, fwhm=args.fwhm
        )
        img.to_filename(str(img_file))

print("\n")
print(out_dir)
