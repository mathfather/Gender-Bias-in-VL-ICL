import argparse

from src.config import QA_MODELS, QA_PROMPT_MODES, RANDOM_EXP_TYPES, VISUALCOT_DATA_TYPE
from src.evaluation.summarize import summarize_runs
from src.utils import present_args


def main(args: argparse.Namespace) -> None:
    prompt_modes = QA_PROMPT_MODES if args.prompt_mode == "ALL" else [args.prompt_mode]
    for prompt_mode in prompt_modes:
        summarize_runs(
            args.exp_type, "visualcot", VISUALCOT_DATA_TYPE, args.model_name,
            f"{args.model_name}_{prompt_mode}", prompt_mode
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_type", type=str, default="1", choices=RANDOM_EXP_TYPES)
    parser.add_argument("--model_name", type=str, required=True, choices=QA_MODELS)
    parser.add_argument("--prompt_mode", type=str, default="ALL", choices=QA_PROMPT_MODES + ["ALL"])
    args = parser.parse_args()
    present_args(args)
    main(args)
