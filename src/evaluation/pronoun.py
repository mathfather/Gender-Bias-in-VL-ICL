import argparse
import os
from types import SimpleNamespace

import pandas as pd

from src.config import NUM_SHOTS, RANDOM_EXP_TYPES, SHOT_SEEDS, VISOGENDER_TYPES
from src.data.visogender import select_data_types
from src.metric import pronoun_error_rates, pronoun_reveal_rates
from src.model import MODEL_MAP
from src.utils import build_result_folder, present_args, save_json


def eval_pronoun(args: argparse.Namespace | SimpleNamespace) -> None:
    result_folder = build_result_folder(
        args.exp_type, "visogender", args.data_type, args.shot_seed, args.model_name,
        args.num_shot
    )
    result_path = os.path.join(result_folder, "result.tsv")
    eval_path = os.path.join(result_folder, "evaluation.json")
    if not os.path.exists(result_path):
        print(f"[SKIP] {result_path} not found")
        return
    if os.path.exists(eval_path):
        print(f"[SKIP] {eval_path} already exists")
        return

    result = pd.read_csv(result_path, sep="\t", header=0)
    true_g = result["Occupation_perceived_gender"]
    pred_g = result["predicted_gender"]
    evaluation = {**pronoun_error_rates(true_g, pred_g), **pronoun_reveal_rates(true_g, pred_g)}
    save_json(evaluation, eval_path)


def main(args: argparse.Namespace) -> None:
    for data_type in select_data_types(args.data_type):
        eval_pronoun(SimpleNamespace(**{**vars(args), "data_type": data_type}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_type", type=str, default="1", choices=RANDOM_EXP_TYPES)
    parser.add_argument("--shot_seed", type=int, default=SHOT_SEEDS[0], choices=SHOT_SEEDS)
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_MAP))
    parser.add_argument("--data_type", type=str, default="ALL", choices=VISOGENDER_TYPES + ["ALL"])
    parser.add_argument("--num_shot", type=int, required=True, choices=NUM_SHOTS)
    args = parser.parse_args()
    present_args(args)
    main(args)
