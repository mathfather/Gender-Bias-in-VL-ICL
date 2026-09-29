import argparse
from types import SimpleNamespace

from src.config import NUM_SHOTS, QA_MODELS, QA_PROMPT_MODES, RANDOM_EXP_TYPES, select_shot_seeds
from src.data import load_valid_data
from src.evaluation.llm_as_judge import eval_qa, load_judge
from src.utils import present_args, select_device


def main(args: argparse.Namespace) -> None:
    prompt_modes = QA_PROMPT_MODES if args.prompt_mode == "ALL" else [args.prompt_mode]
    device = select_device(args.gpu_id)
    model, tokenizer, encoding = load_judge(device)
    valid_data = load_valid_data("visualcot")

    for prompt_mode in prompt_modes:
        for model_name in args.model_name:
            for num_shot in args.num_shot:
                for shot_seed in select_shot_seeds(args.exp_type, num_shot):
                    run_args = SimpleNamespace(
                        exp_type=args.exp_type, shot_seed=shot_seed, model_name=model_name,
                        num_shot=num_shot, prompt_mode=prompt_mode
                    )
                    present_args(run_args)
                    eval_qa(run_args, model, tokenizer, encoding, valid_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_type", type=str, default="1", choices=RANDOM_EXP_TYPES)
    parser.add_argument("--model_name", type=str, nargs="+", required=True, choices=QA_MODELS)
    parser.add_argument("--num_shot", type=int, nargs="+", default=NUM_SHOTS, choices=NUM_SHOTS)
    parser.add_argument(
        "--prompt_mode", type=str, default="advanced", choices=QA_PROMPT_MODES + ["ALL"]
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    present_args(args)
    main(args)
