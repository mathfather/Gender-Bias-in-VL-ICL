import argparse
from types import SimpleNamespace

from src.config import (
    NUM_SHOTS,
    QA_MODELS,
    QA_PROMPT_MODES,
    RANDOM_EXP_TYPES,
    select_shot_seeds,
)
from src.data import load_train_data, load_valid_data
from src.inference.qa import DATA_NAME, qa_exp
from src.model import load_model
from src.utils import present_args, select_device


def main(args: argparse.Namespace) -> None:
    model_helper = load_model(args.model_name, device=select_device(args.gpu_id))
    valid_data = load_valid_data(DATA_NAME)
    train_data = load_train_data(DATA_NAME)
    for num_shot in args.num_shot:
        for shot_seed in select_shot_seeds(args.exp_type, num_shot):
            run_args = SimpleNamespace(
                shot_seed=shot_seed,
                gpu_id=args.gpu_id,
                exp_type=args.exp_type,
                model_name=args.model_name,
                prompt_mode=args.prompt_mode,
                num_shot=num_shot,
            )
            present_args(run_args)
            qa_exp(run_args, model_helper, valid_data, train_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--exp_type", type=str, default="1", choices=RANDOM_EXP_TYPES)
    parser.add_argument("--model_name", type=str, required=True, choices=QA_MODELS)
    parser.add_argument("--prompt_mode", type=str, default="advanced", choices=QA_PROMPT_MODES)
    parser.add_argument("--num_shot", type=int, nargs="+", default=NUM_SHOTS, choices=NUM_SHOTS)
    args = parser.parse_args()
    present_args(args)
    main(args)
