"""DCI: single-person Densely Captioned Images whose short and extra captions agree on one gender.

Builds captions_bias.json and the train/valid splits under data/dci/ from the slimmed
annotations (utils.slim_dci_annotations), and formats one captioning query with its shots.
"""

import argparse
import os

import nltk

from src.config import DATA, FOLDER_MAP, MULTI_PERSON_SET
from src.data.cocobias import split_by_gender
from src.utils import detect_gender_from_text, load_json, save_json

DCI_DIR                 = os.path.join(DATA, "dci")
IMAGES_DIR              = os.path.join(DCI_DIR, FOLDER_MAP["dci"])
ANNOTATIONS_DIR         = os.path.join(DCI_DIR, "annotations")
ANNOTATIONS_PATH        = os.path.join(ANNOTATIONS_DIR, "annotations.json")
BIAS_CAPTION_PATH       = os.path.join(DCI_DIR, "captions_bias.json")
BIAS_CAPTION_TRAIN_PATH = os.path.join(DCI_DIR, "captions_bias_train.json")
BIAS_CAPTION_VALID_PATH = os.path.join(DCI_DIR, "captions_bias_valid.json")

# Removed from the valid split after manual inspection: 113031's captions describe a statue of a
# maned lion ("he"), not a person, and 115432 (Female) keeps the valid pool gender-balanced.
VALID_EXCLUDE_IDS = {"113031", "115432"}


def process_dci() -> None:
    """Writes captions_bias.json: images on disk whose two captions name one gender and no group."""
    annotations = load_json(ANNOTATIONS_PATH)
    result = {}
    skipped_missing = 0
    skipped_multi = 0
    for image_id, entry in annotations.items():
        if not os.path.exists(os.path.join(IMAGES_DIR, entry["image"])):
            skipped_missing += 1
            continue
        combined_tokens = set(nltk.word_tokenize(
            (entry["short_caption"] + " " + entry["extra_caption"]).lower()
        ))
        if combined_tokens & MULTI_PERSON_SET:
            skipped_multi += 1
            continue
        short_gender = detect_gender_from_text(entry["short_caption"])
        extra_gender = detect_gender_from_text(entry["extra_caption"])
        if short_gender not in ("Male", "Female") or short_gender != extra_gender:
            continue
        result[image_id] = {
            "short_caption":   entry["short_caption"],
            "extra_caption":   entry["extra_caption"],
            "file_name":       entry["image"],
            "height":          entry["height"],
            "width":           entry["width"],
            "detected_gender": short_gender,
        }

    save_json(result, BIAS_CAPTION_PATH)
    print(f"Saved {len(result)} samples to {BIAS_CAPTION_PATH}")
    if skipped_missing:
        print(f"Skipped {skipped_missing} entries with missing images")
    if skipped_multi:
        print(f"Skipped {skipped_multi} entries with multi-person words")


def split_dci() -> None:
    data = load_json(BIAS_CAPTION_PATH)
    valid_ids, train_ids = split_by_gender(data)
    valid_ids = [i for i in valid_ids if i not in VALID_EXCLUDE_IDS]
    valid_data = {i: data[i] for i in valid_ids}
    train_data = {i: data[i] for i in train_ids}
    save_json(valid_data, BIAS_CAPTION_VALID_PATH)
    save_json(train_data, BIAS_CAPTION_TRAIN_PATH)
    print(f"Train: {len(train_data)} | Valid: {len(valid_data)}")


def format_dci(
    all_valid_data: dict[str, dict],
    all_train_data: dict[str, dict],
    query_image_id: str,
    shot_image_ids: list[str],
    num_shot: int,
    image_type: str,
) -> tuple[list[str], list[str]]:
    assert image_type == "dci", f"DCI supports only image_type 'dci', got '{image_type}'."
    samples = [all_train_data[image_id] for image_id in shot_image_ids] if num_shot > 0 else []
    samples.append(all_valid_data[query_image_id])
    captions = [sample["short_caption"] for sample in samples]
    image_list = [os.path.join(IMAGES_DIR, sample["file_name"]) for sample in samples]
    return captions, image_list


def main() -> None:
    if os.path.exists(BIAS_CAPTION_TRAIN_PATH) and os.path.exists(BIAS_CAPTION_VALID_PATH):
        print(f"Splits already exist: {BIAS_CAPTION_TRAIN_PATH}, {BIAS_CAPTION_VALID_PATH}")
        return
    if not os.path.exists(BIAS_CAPTION_PATH):
        process_dci()
    split_dci()


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
