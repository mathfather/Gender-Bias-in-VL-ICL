"""VisualCoT: GQA single-person questions with gender words, neutralized for the VQA query.

Builds cot_bias.json and the train/valid splits under data/visual-cot/ from the Visual-CoT
annotations and GQA scene graphs, and formats one QA query with its shots for inference.
"""

import argparse
import os
import re

import nltk

from src.config import DATA, GENDER_WORD_SET, MULTI_PERSON_SET
from src.data.cocobias import split_by_gender
from src.utils import detect_gender_from_text, load_json, load_jsonl, save_json

VISUALCOT_DIR           = os.path.join(DATA, "visual-cot")
ANNOTATIONS_DIR         = os.path.join(VISUALCOT_DIR, "annotations")
SCENE_GRAPHS_DIR        = os.path.join(ANNOTATIONS_DIR, "sceneGraphs")
IMAGES_DIR              = os.path.join(VISUALCOT_DIR, "images")
COT_TRAIN_PATH          = os.path.join(ANNOTATIONS_DIR, "gqa_cot_train.jsonl")
COT_VAL_PATH            = os.path.join(ANNOTATIONS_DIR, "gqa_cot_val_detailed.jsonl")
TRAIN_SCENE_GRAPHS_PATH = os.path.join(SCENE_GRAPHS_DIR, "train_sceneGraphs.json")
BIAS_COT_PATH           = os.path.join(VISUALCOT_DIR, "cot_bias.json")
BIAS_COT_TRAIN_PATH     = os.path.join(VISUALCOT_DIR, "cot_bias_train.json")
BIAS_COT_VALID_PATH     = os.path.join(VISUALCOT_DIR, "cot_bias_valid.json")

# The 0-shot run shows this GQA sample (2383701, a dog with a ball, no people) as its only shot,
# so the model sees the Thought/Answer format without any gender cue.
NEUTRAL_FORMAT_SHOT = {
    "question": "What animal is playing with the ball?",
    "answer": "dog",
    "full_answer": "The dog is playing with the ball.",
    "thought": (
        "1. Identify the object mentioned in the question within the image,"
        ' which is the "ball."'
        " 2. Determine which animal is interacting with the ball by looking"
        ' for an action that denotes "playing with" nearby the identified'
        " ball in the image. "
        " 3. Once the animal involved in the action has been located,"
        " ascertain the specific name or type of animal engaged in playing"
        " with the ball."
    ),
    "detected_gender": "Neutral",
}
NEUTRAL_FORMAT_IMAGE = os.path.join(IMAGES_DIR, "2383701.jpg")
QA_SAMPLE_KEYS = ["answer", "full_answer", "thought", "detected_gender"]

# Singular person nouns only: "people" is plural and "human" never occurs in GQA scene graphs.
PERSON_LABELS = {"man", "boy", "guy", "woman", "girl", "lady", "person", "child"}

# Gender words a token swap can neutralize; others ("him", "her", "mother", "policeman",
# "waitress", "female") need restructuring or are ambiguous.
NEUTRALIZABLE_WORDS = {"man", "woman", "boy", "girl", "lady", "guy", "he", "she"}
GENDER_TO_NEUTRAL = {
    "man":   "person",
    "woman": "person",
    "boy":   "person",
    "girl":  "person",
    "lady":  "person",
    "guy":   "person",
    "he":    "this person",
    "she":   "this person",
}


def get_single_person_image_ids(scene_graphs: dict) -> set[str]:
    single_person_ids = set()
    for image_id, graph in scene_graphs.items():
        person_count = 0
        for obj in graph["objects"].values():
            if obj["name"] in PERSON_LABELS:
                person_count += 1
            if person_count > 1:
                break
        if person_count == 1:
            single_person_ids.add(image_id)
    return single_person_ids


def filter_cot_single_person(cot_data: list[dict], single_person_ids: set[str]) -> list[dict]:
    return [item for item in cot_data if item["image"].replace(".jpg", "") in single_person_ids]


def _question_words(question: str) -> set[str]:
    tokens = set(question.lower().strip("?,.!").split())
    return {token.strip("?,.!") for token in tokens}


def filter_gendered_questions(cot_data: list[dict]) -> list[dict]:
    return [item for item in cot_data if _question_words(item["question"]) & GENDER_WORD_SET]


def filter_who_questions(cot_data: list[dict]) -> list[dict]:
    """Drops 'who' questions, which are trivial on single-person images."""
    return [item for item in cot_data if not item["question"].strip().lower().startswith("who")]


def filter_neutralizable_questions(cot_data: list[dict]) -> list[dict]:
    filtered = []
    for item in cot_data:
        gender_tokens = _question_words(item["question"]) & GENDER_WORD_SET
        if gender_tokens and gender_tokens <= NEUTRALIZABLE_WORDS:
            filtered.append(item)
    return filtered


def filter_multi_person_text(cot_data: list[dict]) -> list[dict]:
    filtered = []
    for item in cot_data:
        combined_text = " ".join([item["question"], item["answer"], item["full_answer"]]).lower()
        if not set(nltk.word_tokenize(combined_text)) & MULTI_PERSON_SET:
            filtered.append(item)
    return filtered


def deduplicate_by_image(cot_data: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for item in cot_data:
        image = item["image"]
        if image not in best or len(item["question"]) > len(best[image]["question"]):
            best[image] = item
    return list(best.values())


def process_visualcot() -> None:
    """Writes cot_bias.json from both CoT splits, whose images all come from GQA train."""
    single_person_ids = get_single_person_image_ids(load_json(TRAIN_SCENE_GRAPHS_PATH))
    train_filtered = filter_cot_single_person(load_jsonl(COT_TRAIN_PATH), single_person_ids)
    val_filtered = filter_cot_single_person(load_jsonl(COT_VAL_PATH), single_person_ids)

    combined = train_filtered + val_filtered
    combined = filter_gendered_questions(combined)
    combined = filter_who_questions(combined)
    combined = filter_multi_person_text(combined)
    combined = filter_neutralizable_questions(combined)
    combined = deduplicate_by_image(combined)

    result = {}
    for item in combined:
        image_id = item["image"].replace(".jpg", "")
        item["file_name"] = item.pop("image")
        item.pop("split", None)
        item["detected_gender"] = detect_gender_from_text(item["question"])
        result[image_id] = item

    save_json(result, BIAS_COT_PATH)
    print(f"Saved {len(result)} samples to {BIAS_COT_PATH}")


def neutralize_question(question: str) -> str:
    for gendered, neutral in GENDER_TO_NEUTRAL.items():
        question = re.sub(rf"\b{gendered}\b", neutral, question, flags=re.IGNORECASE)
    return question


def split_visualcot() -> None:
    data = load_json(BIAS_COT_PATH)
    valid_ids, train_ids = split_by_gender(data)
    valid_data = {i: data[i] for i in valid_ids}
    for item in valid_data.values():
        item["neutral_question"] = neutralize_question(item["question"])
    train_data = {i: data[i] for i in train_ids}
    save_json(valid_data, BIAS_COT_VALID_PATH)
    save_json(train_data, BIAS_COT_TRAIN_PATH)
    print(f"Train: {len(train_data)} | Valid: {len(valid_data)}")


def qa_sample(entry: dict, question: str) -> dict:
    return {"question": question, **{key: entry[key] for key in QA_SAMPLE_KEYS}}


def format_visualcot(
    all_valid_data: dict[str, dict],
    all_train_data: dict[str, dict],
    query_image_id: str,
    shot_image_ids: list[str],
    num_shot: int,
) -> tuple[list[dict], list[str]]:
    """(samples, image paths) of the shots, or the format shot at 0 shots, then the query."""
    samples = []
    image_list = []
    if num_shot > 0:
        for image_id in shot_image_ids:
            sample = all_train_data[image_id]
            samples.append(qa_sample(sample, sample["question"]))
            image_list.append(os.path.join(IMAGES_DIR, sample["file_name"]))
    else:
        samples.append(NEUTRAL_FORMAT_SHOT)
        image_list.append(NEUTRAL_FORMAT_IMAGE)
    query = all_valid_data[query_image_id]
    samples.append(qa_sample(query, query["neutral_question"]))
    image_list.append(os.path.join(IMAGES_DIR, query["file_name"]))
    return samples, image_list


def main() -> None:
    if os.path.exists(BIAS_COT_TRAIN_PATH) and os.path.exists(BIAS_COT_VALID_PATH):
        print(f"Splits already exist: {BIAS_COT_TRAIN_PATH}, {BIAS_COT_VALID_PATH}")
        return
    if not os.path.exists(BIAS_COT_PATH):
        process_visualcot()
    split_visualcot()


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
