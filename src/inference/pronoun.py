import argparse
import os
import random
from types import SimpleNamespace

import pandas as pd
import torch
from tqdm import tqdm

from src.config import (
    NUM_SHOTS,
    RANDOM_EXP_TYPES,
    RANDOM_SEED,
    SHOT_SEEDS,
    TQDM_STYLE,
    VISOGENDER_TYPES,
)
from src.data import load_train_data, load_valid_data
from src.data.visogender import format_visogender, select_data_types
from src.model import MODEL_MAP, ModelHelper, load_model
from src.sampling import present_shot_ids, select_shot_ids
from src.utils import build_result_folder, mkdir, present_args, save_json, select_device

DATA_NAME = "visogender"


def run_pronoun_inference(
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
        list(valid_data), desc="Pronoun Inference", bar_format=TQDM_STYLE, dynamic_ncols=True
    )
    valid_ids, pred_genders, prob_his_list, prob_her_list = [], [], [], []
    for image_id in tqdm_valid_data:
        occupations, genders, objects, image_list = format_visogender(
            all_valid_data=valid_data, all_train_data=train_data, query_image_id=image_id,
            shot_image_ids=shot_image_ids, num_shot=args.num_shot, data_type=args.data_type
        )
        model_input = model_helper.prepare_pronoun_input(occupations, genders, objects, image_list)
        pred_gender, prob_his, prob_her = model_helper.sample_next_token(model_input)
        true_gender = valid_data[image_id]["occupation_perceived_gender"]
        tqdm_valid_data.set_postfix({"True": true_gender, "Pred": pred_gender})
        valid_ids.append(image_id)
        pred_genders.append(pred_gender)
        prob_his_list.append(prob_his)
        prob_her_list.append(prob_her)

    return {
        "IDX": valid_ids,
        "Occupation_perceived_gender": [
            valid_data[idx]["occupation_perceived_gender"] for idx in valid_ids
        ],
        "predicted_gender": pred_genders,
        "prob_his": prob_his_list,
        "prob_her": prob_her_list,
    }


def pronoun_exp(
    args: argparse.Namespace | SimpleNamespace,
    model_helper: ModelHelper,
    valid_data: dict[str, dict],
    train_data: dict[str, dict],
) -> None:
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    result_folder_path = build_result_folder(
        args.exp_type, DATA_NAME, args.data_type, args.shot_seed, args.model_name, args.num_shot
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
        results = run_pronoun_inference(args, model_helper, valid_data, train_data, shot_image_ids)
    pd.DataFrame.from_dict(results).to_csv(result_path, sep="\t", header=True, index=False)

    args_dict = vars(args) | {"seed": RANDOM_SEED}
    args_dict["shot_id"] = shot_image_ids
    save_json(args_dict, os.path.join(result_folder_path, "args.json"))


def main(args: argparse.Namespace) -> None:
    model_helper = load_model(args.model_name, device=select_device(args.gpu_id))
    for data_type in select_data_types(args.data_type):
        valid_data = load_valid_data(DATA_NAME, data_type=data_type)
        train_data = load_train_data(DATA_NAME, data_type=data_type)
        run_args = SimpleNamespace(**(vars(args) | {"data_type": data_type}))
        pronoun_exp(run_args, model_helper, valid_data, train_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot_seed", type=int, default=SHOT_SEEDS[0], choices=SHOT_SEEDS)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--exp_type", type=str, default="1", choices=RANDOM_EXP_TYPES)
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_MAP))
    parser.add_argument("--data_type", type=str, default="ALL", choices=VISOGENDER_TYPES + ["ALL"])
    parser.add_argument("--num_shot", type=int, required=True, choices=NUM_SHOTS)
    args = parser.parse_args()
    present_args(args)
    main(args)
