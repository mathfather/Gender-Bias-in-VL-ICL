import argparse

from src.config import RANDOM_EXP_TYPES, VISOGENDER_TYPES
from src.data.visogender import select_data_types
from src.evaluation.summarize import summarize_runs
from src.model import MODEL_MAP
from src.utils import present_args


def main(args: argparse.Namespace) -> None:
    for data_type in select_data_types(args.data_type):
        summarize_runs(args.exp_type, "visogender", data_type, args.model_name, args.model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_type", type=str, default="1", choices=RANDOM_EXP_TYPES)
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_MAP))
    parser.add_argument("--data_type", type=str, default="ALL", choices=VISOGENDER_TYPES + ["ALL"])
    args = parser.parse_args()
    present_args(args)
    main(args)
