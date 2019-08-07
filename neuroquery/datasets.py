import os
import pathlib
import tempfile
import zipfile
import shutil

import requests


def get_neuroquery_data_dir(data_dir=None):
    if data_dir is None:
        data_dir = (
            pathlib.Path(os.environ.get("HOME", ".")) / "neuroquery_data"
        )
    else:
        data_dir = pathlib.Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    return str(data_dir)


def _download_neuroquery_model(out_dir):
    print("Downloading NeuroQuery model...")
    resp = requests.get(
        "https://raw.githubusercontent.com/neuroquery/neuroquery_data/"
        "master/neuroquery_model.zip"
    )
    resp.raise_for_status()
    print("Downloaded.")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = pathlib.Path(tmp_dir)
        zip_path = str(tmp_dir / "neuroquery_data_master.zip")
        with open(zip_path, "wb") as f:
            f.write(resp.content)
        with zipfile.ZipFile(zip_path) as zip_f:
            extract_dir = tmp_dir / "extract_dir"
            zip_f.extractall(str(extract_dir))
        model_path = extract_dir / "neuroquery_model"
        shutil.copytree(str(model_path), out_dir)
    return out_dir


def fetch_neuroquery_model(data_dir=None):
    data_dir = pathlib.Path(get_neuroquery_data_dir(data_dir))
    out_dir = data_dir / "neuroquery_model"
    if out_dir.is_dir():
        return str(out_dir)
    _download_neuroquery_model(str(out_dir))
    return str(out_dir)


def fetch_peak_coordinates(data_dir=None):
    data_dir = pathlib.Path(get_neuroquery_data_dir(data_dir))
    out_file = data_dir / "coordinates.csv"
    if out_file.is_file():
        return str(out_file)
    print("Downloading coordinates ...")
    resp = requests.get(
        "https://raw.githubusercontent.com/neuroquery/neuroquery_data/"
        "master/training_data/coordinates.csv"
    )
    resp.raise_for_status()
    out_file = str(out_file)
    with open(out_file, "wb") as f:
        f.write(resp.content)
    print("Downloaded.")
    return out_file
