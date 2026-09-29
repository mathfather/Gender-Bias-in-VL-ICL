"""VisoGender: occupation images for pronoun resolution, in two types, OO and OP.

OO pairs an occupation with an object, OP with a participant. Optionally downloads the images
listed in the raw TSVs, writes the per-type train/valid TSVs and JSONs under data/visogender/,
and formats one pronoun query with its shots for inference.
"""

import argparse
import itertools
import os
import time

import pandas as pd
import requests

from src.config import DATA, RANDOM_SEED, VISOGENDER_TYPES
from src.utils import present_args, save_json

VISOGENDER_DIR      = os.path.join(DATA, "visogender")
ANNOTATIONS_DIR     = os.path.join(VISOGENDER_DIR, "annotations")
IMAGES_DIR          = os.path.join(VISOGENDER_DIR, "images")
OO_ANNOTATIONS_PATH = os.path.join(ANNOTATIONS_DIR, "OO_Visogender_10052025.tsv")
OP_ANNOTATIONS_PATH = os.path.join(ANNOTATIONS_DIR, "OP_Visogender_11012024.tsv")
OO_TRAIN_TSV_PATH   = os.path.join(VISOGENDER_DIR, "oo_train.tsv")
OO_VALID_TSV_PATH   = os.path.join(VISOGENDER_DIR, "oo_valid.tsv")
OP_TRAIN_TSV_PATH   = os.path.join(VISOGENDER_DIR, "op_train.tsv")
OP_VALID_TSV_PATH   = os.path.join(VISOGENDER_DIR, "op_valid.tsv")
OO_TRAIN_JSON_PATH  = os.path.join(VISOGENDER_DIR, "oo_train.json")
OO_VALID_JSON_PATH  = os.path.join(VISOGENDER_DIR, "oo_valid.json")
OP_TRAIN_JSON_PATH  = os.path.join(VISOGENDER_DIR, "op_train.json")
OP_VALID_JSON_PATH  = os.path.join(VISOGENDER_DIR, "op_valid.json")

ANNOTATION_PATHS = {
    "OO": OO_ANNOTATIONS_PATH,
    "OP": OP_ANNOTATIONS_PATH,
}
SPLIT_PATHS = {
    ("OO", "train"): (OO_TRAIN_TSV_PATH, OO_TRAIN_JSON_PATH),
    ("OO", "valid"): (OO_VALID_TSV_PATH, OO_VALID_JSON_PATH),
    ("OP", "train"): (OP_TRAIN_TSV_PATH, OP_TRAIN_JSON_PATH),
    ("OP", "valid"): (OP_VALID_TSV_PATH, OP_VALID_JSON_PATH),
}
GENDER_COLUMNS = {
    "OO": ["Occupation_perceived_gender"],
    "OP": ["Occupation_perceived_gender", "Participant_perceived_gender"],
}
OBJECT_KEYS = {
    "OO": "object",
    "OP": "participant",
}
URL_COLUMN = "URL type (Type NA if can't find)"
DROPPED_COLUMNS = ["Licence", URL_COLUMN, "Error codes", "Annotator", "LICENSE OK?"]
PERCEIVED_GENDERS = ["masculine", "feminine"]
GENDER_MAP = {"masculine": "Male", "feminine": "Female"}
IMAGE_EXTS = ["jpeg", "webp"]
SHOTS_PER_GROUP = 2
DOWNLOAD_RETRIES = 2


def load_annotations() -> pd.DataFrame:
    frames = []
    for data_type in VISOGENDER_TYPES:
        frame = pd.read_csv(ANNOTATION_PATHS[data_type], sep="\t")
        frame["data_type"] = [data_type] * len(frame)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def download_image(url: str, save_path: str) -> None:
    # Dead and slow URLs are expected: a failed download is logged and the image left out.
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            response = requests.get(url, stream=True, timeout=(3, 10))
            if response.status_code == 200:
                with open(save_path, "wb") as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)
                return
            print(f"Failed: {response.status_code}; {save_path}")
        except requests.exceptions.Timeout:
            print(f"Request failed: No. {attempt} tries; {save_path}")
        except requests.exceptions.RequestException as error:
            print(f"Request failed: {error}, try {attempt} / {DOWNLOAD_RETRIES}")
        time.sleep(1)


def image_extension(image_path: str) -> str | None:
    """'webp' for a WebP file, 'jpeg' for any other image, None when the file is no image."""
    from PIL import Image, UnidentifiedImageError

    # A dead link can answer 200 with an HTML page. The paper's image set names every non-WebP
    # image, PNG included, <IDX>.jpeg, and the file names of the splits follow it.
    try:
        with Image.open(image_path) as image:
            return "webp" if image.format == "WEBP" else "jpeg"
    except UnidentifiedImageError:
        return None


def store_image(image_path: str) -> None:
    ext = image_extension(image_path)
    if ext is None:
        os.remove(image_path)
        print(f"Not an image, removed; {image_path}")
    else:
        os.rename(image_path, f"{image_path}.{ext}")


def find_image_name(idx: str) -> str | None:
    for ext in IMAGE_EXTS:
        file_name = f"{idx}.{ext}"
        if os.path.exists(os.path.join(IMAGES_DIR, file_name)):
            return file_name
    return None


def image_exists(idx: str) -> bool:
    return find_image_name(idx) is not None


def download_images(data: pd.DataFrame) -> None:
    for url, idx in zip(data[URL_COLUMN].tolist(), data["IDX"].tolist(), strict=True):
        if image_exists(idx):
            continue
        save_path = os.path.join(IMAGES_DIR, idx)
        download_image(url, save_path)
        if os.path.exists(save_path):
            store_image(save_path)


def split_by_occupation(
    data: pd.DataFrame,
    occupations: list[str],
    gender_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, valid): SHOTS_PER_GROUP rows of each occupation and perceived-gender group train."""
    trains = []
    valids = []
    for occupation in occupations:
        current = data[data["Occupation"] == occupation]
        for genders in itertools.product(PERCEIVED_GENDERS, repeat=len(gender_columns)):
            group = current
            for column, gender in zip(gender_columns, genders, strict=True):
                group = group[group[column] == gender]
            train = group.sample(n=SHOTS_PER_GROUP, replace=False, random_state=RANDOM_SEED)
            trains.append(train)
            valids.append(group[~group.index.isin(train.index)])
    return pd.concat(trains), pd.concat(valids)


def _cell_text(row: pd.Series, column: str) -> str:
    value = row.get(column)
    return value if pd.notna(value) else ""


def split_records(frame: pd.DataFrame) -> dict[str, dict]:
    records = {}
    for _, row in frame.iterrows():
        records[row["IDX"]] = {
            "file_name":                    row["file_name"],
            "detected_gender":              GENDER_MAP[row["Occupation_perceived_gender"]],
            "occupation":                   row["Occupation"],
            "occupation_perceived_gender":  row["Occupation_perceived_gender"],
            "sector":                       row["Sector"],
            "specialisation":               row["Specialisation"],
            "object":                       _cell_text(row, "Object"),
            "participant":                  _cell_text(row, "Participant"),
            "participant_perceived_gender": _cell_text(row, "Participant_perceived_gender"),
            "data_type":                    row["data_type"],
        }
    return records


def select_data_types(data_type: str) -> list[str]:
    return list(VISOGENDER_TYPES) if data_type == "ALL" else [data_type]


def format_visogender(
    all_valid_data: dict[str, dict],
    all_train_data: dict[str, dict],
    query_image_id: str,
    shot_image_ids: list[str],
    num_shot: int,
    data_type: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    assert data_type in VISOGENDER_TYPES, f"Invalid data_type '{data_type}'."
    object_key = OBJECT_KEYS[data_type]
    samples = [all_train_data[image_id] for image_id in shot_image_ids] if num_shot > 0 else []
    samples.append(all_valid_data[query_image_id])
    occupations = [sample["occupation"] for sample in samples]
    genders = [sample["occupation_perceived_gender"] for sample in samples]
    objects = [sample[object_key] for sample in samples]
    image_list = [os.path.join(IMAGES_DIR, sample["file_name"]) for sample in samples]
    return occupations, genders, objects, image_list


def main(args: argparse.Namespace) -> None:
    output_paths = [path for paths in SPLIT_PATHS.values() for path in paths]
    if all(os.path.exists(path) for path in output_paths):
        print(f"Splits already exist in {VISOGENDER_DIR}")
        return
    data = load_annotations()
    if args.download_images:
        download_images(data)
    data = data[data["IDX"].map(image_exists)].drop(columns=DROPPED_COLUMNS)
    occupations = data["Occupation"].unique().tolist()
    for data_type in VISOGENDER_TYPES:
        splits = split_by_occupation(
            data[data["data_type"] == data_type], occupations, GENDER_COLUMNS[data_type]
        )
        for split, frame in zip(["train", "valid"], splits, strict=True):
            tsv_path, json_path = SPLIT_PATHS[(data_type, split)]
            frame["file_name"] = [find_image_name(idx) for idx in frame["IDX"].tolist()]
            frame.to_csv(tsv_path, sep="\t", index=False)
            records = split_records(frame)
            save_json(records, json_path)
            print(f"Saved {len(records)} entries to {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download_images", action="store_true",
        help="fetch the images listed in the raw TSVs into data/visogender/images/ before "
             "building the splits (existing files are skipped)"
    )
    args = parser.parse_args()
    present_args(args)
    main(args)
