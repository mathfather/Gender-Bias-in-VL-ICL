import argparse
import os
import random
import re
from types import SimpleNamespace

import pandas as pd
import torch
from tqdm import tqdm

from src.config import (
    NUM_SHOTS,
    QA_MAX_NEW_TOKEN,
    QA_MODELS,
    QA_PROMPT_MODES,
    RANDOM_EXP_TYPES,
    RANDOM_SEED,
    SHOT_SEEDS,
    TQDM_STYLE,
    VISUALCOT_DATA_TYPE,
)
from src.data import load_train_data, load_valid_data
from src.data.visualcot import format_visualcot
from src.model import ModelHelper, load_model
from src.sampling import present_shot_ids, select_shot_ids
from src.utils import build_result_folder, mkdir, present_args, save_json, select_device

DATA_NAME = "visualcot"

_ANSWER_RE = re.compile(r"answer:\s*(.+)", re.IGNORECASE)
_THOUGHT_RE = re.compile(r"thought:\s*(.+?)(?=\nanswer:|\Z)", re.IGNORECASE | re.DOTALL)


def extract_answer(text: str) -> str:
    match = _ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_thought(text: str) -> str:
    match = _THOUGHT_RE.search(text)
    if match:
        return match.group(1).strip()
    return ""


def run_qa_inference(
    args: argparse.Namespace | SimpleNamespace,
    model_helper: ModelHelper,
    valid_data: dict[str, dict],
    train_data: dict[str, dict],
    shot_image_ids: list[str],
) -> dict[str, list]:
    assert len(shot_image_ids) == args.num_shot, (
        f"Got {len(shot_image_ids)} shots for {args.num_shot}."
    )
    tqdm_valid_data = tqdm(
        valid_data, desc="Generating QA answers", bar_format=TQDM_STYLE, dynamic_ncols=True
    )
    image_ids, file_names, questions, reference_answers = [], [], [], []
    generated_answers, generated_thoughts = [], []
    for image_id in tqdm_valid_data:
        samples, image_list = format_visualcot(
            all_valid_data=valid_data, all_train_data=train_data, query_image_id=image_id,
            shot_image_ids=shot_image_ids, num_shot=args.num_shot
        )
        model_input = model_helper.prepare_qa_input(samples, image_list, args.prompt_mode)
        raw_output = model_helper.generate(model_input, QA_MAX_NEW_TOKEN)
        answer = extract_answer(raw_output)
        thought = extract_thought(raw_output)

        thought_preview = " ".join(thought.split()[:10]) + "..." if thought else ""
        tqdm_valid_data.set_postfix({"Generated": answer, "Thought": thought_preview})
        image_ids.append(image_id)
        file_names.append(valid_data[image_id]["file_name"])
        questions.append(valid_data[image_id]["question"])
        reference_answers.append(valid_data[image_id]["answer"])
        generated_answers.append(answer)
        generated_thoughts.append(thought)

    return {
        "image_id": image_ids,
        "file_name": file_names,
        "question": questions,
        "reference_answer": reference_answers,
        "generated_answer": generated_answers,
        "generated_thought": generated_thoughts,
    }


def qa_exp(
    args: argparse.Namespace | SimpleNamespace,
    model_helper: ModelHelper,
    valid_data: dict[str, dict],
    train_data: dict[str, dict],
) -> None:
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    result_folder_path = build_result_folder(
        args.exp_type, DATA_NAME, VISUALCOT_DATA_TYPE, args.shot_seed, args.model_name,
        args.num_shot, args.prompt_mode
    )
    result_path = os.path.join(result_folder_path, "result.tsv")
    if os.path.exists(result_path):
        return
    mkdir(result_folder_path)

    shot_image_ids = select_shot_ids(
        args.exp_type, args.num_shot, train_data, args.shot_seed, RANDOM_SEED, DATA_NAME
    )
    present_shot_ids(shot_image_ids, train_data)
    with torch.no_grad():
        results = run_qa_inference(args, model_helper, valid_data, train_data, shot_image_ids)
    pd.DataFrame.from_dict(results).to_csv(result_path, sep="\t", header=True, index=False)

    args_dict = vars(args) | {"seed": RANDOM_SEED, "max_new_token": QA_MAX_NEW_TOKEN}
    args_dict["shot_id"] = shot_image_ids
    save_json(args_dict, os.path.join(result_folder_path, "args.json"))


def main(args: argparse.Namespace) -> None:
    valid_data = load_valid_data(DATA_NAME)
    train_data = load_train_data(DATA_NAME)
    model_helper = load_model(args.model_name, device=select_device(args.gpu_id))
    qa_exp(args, model_helper, valid_data, train_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot_seed", type=int, default=SHOT_SEEDS[0], choices=SHOT_SEEDS)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--exp_type", type=str, default="1", choices=RANDOM_EXP_TYPES)
    parser.add_argument("--model_name", type=str, required=True, choices=QA_MODELS)
    parser.add_argument("--prompt_mode", type=str, default="advanced", choices=QA_PROMPT_MODES)
    parser.add_argument("--num_shot", type=int, required=True, choices=NUM_SHOTS)
    args = parser.parse_args()
    present_args(args)
    main(args)
