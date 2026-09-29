"""Inference drivers: one module per task plus a *_batch sweep over shot counts and shot seeds.

Each run writes result.tsv and args.json under results/<setting>/ and skips a finished run:
    python -m src.inference.caption --model_name <model> --num_shot <k> [--exp_type <1-6>]
        [--data_name cocobias|dci] [--image_type <type>] [--shot_seed <seed>] [--gpu_id <id>]
    python -m src.inference.caption_batch --model_name <model> [--exp_type <1-6>]
        [--data_name cocobias|dci] [--image_type <type>] [--num_shot <k> ...] [--gpu_id <id>]
    python -m src.inference.pronoun --model_name <model> --num_shot <k> [--exp_type <1-4>]
        [--data_type OO|OP|ALL] [--shot_seed <seed>] [--gpu_id <id>]
    python -m src.inference.pronoun_batch --model_name <model> [--exp_type <1-4>]
        [--data_type OO|OP|ALL] [--num_shot <k> ...] [--gpu_id <id>]
    python -m src.inference.qa --model_name MiniCPM|Phi35V|Qwen3VL --num_shot <k>
        [--exp_type <1-4>] [--prompt_mode base|advanced] [--shot_seed <seed>] [--gpu_id <id>]
    python -m src.inference.qa_batch --model_name MiniCPM|Phi35V|Qwen3VL [--exp_type <1-4>]
        [--prompt_mode base|advanced] [--num_shot <k> ...] [--gpu_id <id>]
"""

# The drivers are not re-exported: importing them here would load each module twice under
# `python -m src.inference.<module>` (runpy warns about it).
__all__: list[str] = []
