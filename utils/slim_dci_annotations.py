import argparse
import os

from src.data.dci import ANNOTATIONS_DIR, ANNOTATIONS_PATH, DCI_DIR
from src.utils import load_json, mkdir, save_json

RAW_ANNOTATIONS_DIR = os.path.join(DCI_DIR, "densely_captioned_images", "annotations")
KEEP_FIELDS = ["short_caption", "extra_caption", "image", "height", "width"]


def main() -> None:
    if os.path.exists(ANNOTATIONS_PATH):
        print(f"Slim annotations already exist: {ANNOTATIONS_PATH}")
        return
    if not os.path.isdir(RAW_ANNOTATIONS_DIR):
        raise FileNotFoundError(
            f"Raw annotations folder not found: {RAW_ANNOTATIONS_DIR} (extract dci.tar.gz into "
            f"{DCI_DIR})."
        )
    file_names = [name for name in os.listdir(RAW_ANNOTATIONS_DIR) if name.endswith("-data.json")]
    result = {}
    for file_name in sorted(file_names, key=lambda name: int(name.split("-")[0])):
        entry = load_json(os.path.join(RAW_ANNOTATIONS_DIR, file_name))
        result[file_name.split("-")[0]] = {key: entry[key] for key in KEEP_FIELDS}
    mkdir(ANNOTATIONS_DIR)
    save_json(result, ANNOTATIONS_PATH)
    print(f"Saved {len(result)} entries to {ANNOTATIONS_PATH}")


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
