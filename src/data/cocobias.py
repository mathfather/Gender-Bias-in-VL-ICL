"""COCOBias: single-person MS-COCO 2014 val images whose caption gender matches Zhao et al.'s label.

Builds captions_val.json, captions_bias.json and the train/valid splits under data/coco2014/,
and formats one captioning query with its shots for inference.
"""

import argparse
import os
import random

from src.config import DATA, FOLDER_MAP, RANDOM_SEED
from src.utils import detect_gender_from_text, load_json, save_json

COCO_DIR                         = os.path.join(DATA, "coco2014")
COCO_IMAGES_DIR                  = os.path.join(COCO_DIR, "images")
COCO_VAL2014_IMAGES              = os.path.join(COCO_IMAGES_DIR, FOLDER_MAP["coco"])
CAPTION_ANNOTATIONS_DIR          = os.path.join(COCO_DIR, "annotations")
CAPTION_ANNOTATIONS_PATH         = os.path.join(CAPTION_ANNOTATIONS_DIR, "captions_val2014.json")
INSTANCES_ANNOTATIONS_PATH       = os.path.join(CAPTION_ANNOTATIONS_DIR, "instances_val2014.json")
CAPTION_PATH                     = os.path.join(COCO_DIR, "captions_val.json")
BIAS_ANNOTATIONS_DIR             = os.path.join(COCO_DIR, "bias")
BIAS_ANNOTATIONS_PATH            = os.path.join(BIAS_ANNOTATIONS_DIR, "images_val2014.csv")
BIAS_CAPTION_PATH                = os.path.join(COCO_DIR, "captions_bias.json")
BIAS_CAPTION_TRAIN_PATH          = os.path.join(COCO_DIR, "captions_bias_train.json")
BIAS_CAPTION_TRAIN_BALANCED_PATH = os.path.join(COCO_DIR, "captions_bias_train_balanced.json")
BIAS_CAPTION_VALID_PATH          = os.path.join(COCO_DIR, "captions_bias_valid.json")

VALID_RATIO = 0.8
PERSON_CATEGORY_ID = 1


def select_gendered_caption(captions: list[str], gender: str) -> str:
    for caption in captions:
        if detect_gender_from_text(caption) == gender:
            return caption
    raise ValueError(f"No caption of gender '{gender}' among {captions}.")


def detect_gender_from_captions(captions: list[str]) -> str:
    genders = {detect_gender_from_text(caption) for caption in captions}
    male_detected = bool(genders & {"Male", "Invalid"})
    female_detected = bool(genders & {"Female", "Invalid"})
    if female_detected and not male_detected:
        return "Female"
    if male_detected and not female_detected:
        return "Male"
    return "Unsure"


def split_by_gender(data: dict[str, dict]) -> tuple[list[str], list[str]]:
    """(valid_ids, train_ids) under RANDOM_SEED; valid ids cut to balance genders join neither."""
    genders = {entry["detected_gender"] for entry in data.values()}
    assert genders <= {"Male", "Female"}, f"Only Male/Female entries can be split, got {genders}."
    random.seed(RANDOM_SEED)
    data_ids = list(data)
    male_ids = [i for i in data_ids if data[i]["detected_gender"] == "Male"]
    female_ids = [i for i in data_ids if data[i]["detected_gender"] == "Female"]
    valid_male_ids = random.sample(male_ids, int(len(male_ids) * VALID_RATIO))
    valid_female_ids = random.sample(female_ids, int(len(female_ids) * VALID_RATIO))
    valid_id_set = set(valid_male_ids + valid_female_ids)
    train_ids = [i for i in data_ids if i not in valid_id_set]
    if len(valid_male_ids) > len(valid_female_ids):
        valid_male_ids = random.sample(valid_male_ids, len(valid_female_ids))
    else:
        valid_female_ids = random.sample(valid_female_ids, len(valid_male_ids))
    return valid_male_ids + valid_female_ids, train_ids


def process_coco2014() -> None:
    """Writes captions_val.json: gendered val2014 images with their captions and person boxes."""
    caption_json = load_json(CAPTION_ANNOTATIONS_PATH)
    image_id_to_file_name = {image["id"]: image["file_name"] for image in caption_json["images"]}
    data = {}
    for anno in caption_json["annotations"]:
        image_id = anno["image_id"]
        if image_id not in data:
            data[image_id] = {"file_name": image_id_to_file_name[image_id], "caption": []}
        data[image_id]["caption"].append(anno["caption"])
    for entry in data.values():
        entry["detected_gender"] = detect_gender_from_captions(entry["caption"])
    female_image_ids = [i for i in data if data[i]["detected_gender"] == "Female"]
    male_image_ids = [i for i in data if data[i]["detected_gender"] == "Male"]
    human_data = {image_id: data[image_id] for image_id in female_image_ids + male_image_ids}

    image_to_bboxes: dict[int, list[dict]] = {}
    for anno in load_json(INSTANCES_ANNOTATIONS_PATH)["annotations"]:
        if anno["category_id"] == PERSON_CATEGORY_ID:
            image_to_bboxes.setdefault(anno["image_id"], []).append({
                "bbox": anno["bbox"], "area": anno["area"], "iscrowd": anno["iscrowd"]
            })
    for image_id in list(human_data):
        if image_id in image_to_bboxes:
            human_data[image_id]["bboxes"] = image_to_bboxes[image_id]
            human_data[image_id]["bbox_count"] = len(image_to_bboxes[image_id])
        else:
            del human_data[image_id]
    save_json(human_data, CAPTION_PATH)


def process_cocobias() -> None:
    """Writes captions_bias.json: captions_val.json entries with Zhao et al.'s annotated_gender."""
    import pandas as pd

    bias_anno = pd.read_csv(BIAS_ANNOTATIONS_PATH, header=0)
    bias_anno = bias_anno[bias_anno["bb_gender"] != "Unsure"]
    bias_anno = bias_anno[bias_anno["bb_gender"] != "Both"]
    bias_anno = bias_anno[bias_anno["bb_gender"].notna()]
    # Rows without a skin label are dropped as well; the published splits depend on it.
    bias_anno = bias_anno[bias_anno["bb_skin"].notna()]

    data = load_json(CAPTION_PATH)
    bias_genders = bias_anno["bb_gender"].tolist()
    bias_image_ids = [str(i) for i in bias_anno["id"].tolist()]
    bias_data = {i: data[i] for i in bias_image_ids if i in data}
    for gender, image_id in zip(bias_genders, bias_image_ids, strict=True):
        if image_id in bias_data:
            bias_data[image_id]["annotated_gender"] = gender
    save_json(bias_data, BIAS_CAPTION_PATH)


def split_cocobias() -> None:
    data = load_json(BIAS_CAPTION_PATH)
    data = {
        image_id: entry for image_id, entry in data.items()
        if entry["bbox_count"] == 1 and entry["detected_gender"] == entry["annotated_gender"]
    }
    valid_ids, train_ids = split_by_gender(data)
    valid_data = {i: data[i] for i in valid_ids}
    train_data = {i: data[i] for i in train_ids}
    save_json(valid_data, BIAS_CAPTION_VALID_PATH)
    save_json(train_data, BIAS_CAPTION_TRAIN_PATH)
    print(f"Train: {len(train_data)} | Valid: {len(valid_data)}")


def format_coco2014(
    all_valid_data: dict[str, dict],
    all_train_data: dict[str, dict],
    query_image_id: str,
    shot_image_ids: list[str],
    num_shot: int,
    image_type: str,
) -> tuple[list[str], list[str]]:
    """(captions, image paths) of the shots then the query; the query is always a val2014 photo."""
    shot_folder = os.path.join(COCO_IMAGES_DIR, FOLDER_MAP[image_type])
    samples = [all_train_data[image_id] for image_id in shot_image_ids] if num_shot > 0 else []
    image_list = [os.path.join(shot_folder, sample["file_name"]) for sample in samples]
    query = all_valid_data[query_image_id]
    samples.append(query)
    image_list.append(os.path.join(COCO_VAL2014_IMAGES, query["file_name"]))
    captions = [
        select_gendered_caption(sample["caption"], sample["detected_gender"]) for sample in samples
    ]
    return captions, image_list


def main() -> None:
    if os.path.exists(BIAS_CAPTION_TRAIN_PATH) and os.path.exists(BIAS_CAPTION_VALID_PATH):
        print(f"Splits already exist: {BIAS_CAPTION_TRAIN_PATH}, {BIAS_CAPTION_VALID_PATH}")
        return
    if not os.path.exists(CAPTION_PATH):
        process_coco2014()
    if not os.path.exists(BIAS_CAPTION_PATH):
        process_cocobias()
    split_cocobias()


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
