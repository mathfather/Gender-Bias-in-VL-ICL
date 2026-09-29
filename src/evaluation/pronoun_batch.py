import argparse
from types import SimpleNamespace

from src.config import NUM_SHOTS, RANDOM_EXP_TYPES, VISOGENDER_TYPES, select_shot_seeds
from src.data.visogender import select_data_types
from src.evaluation.pronoun import eval_pronoun
from src.model import MODEL_MAP
from src.utils import present_args


def main(args: argparse.Namespace) -> None:
    for data_type in select_data_types(args.data_type):
        for num_shot in args.num_shot:
            for shot_seed in select_shot_seeds(args.exp_type, num_shot):
                run_args = SimpleNamespace(
                    exp_type=args.exp_type, shot_seed=shot_seed, model_name=args.model_name,
                    data_type=data_type, num_shot=num_shot
                )
                present_args(run_args)
                eval_pronoun(run_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_type", type=str, default="1", choices=RANDOM_EXP_TYPES)
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_MAP))
    parser.add_argument("--num_shot", type=int, nargs="+", default=NUM_SHOTS, choices=NUM_SHOTS)
    parser.add_argument("--data_type", type=str, default="ALL", choices=VISOGENDER_TYPES + ["ALL"])
    args = parser.parse_args()
    present_args(args)
    main(args)
