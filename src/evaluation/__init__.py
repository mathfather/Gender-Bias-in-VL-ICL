"""Scoring of inference runs and their summary tables.

Each evaluator adds per-sample scores to a run's result.tsv and writes its evaluation.json; each
summarizer collects those files into summary/<setting>/<data_name>-<data_type>/:
    python -m src.evaluation.caption --model_name <model> --num_shot <k> [--shot_seed 100]
    python -m src.evaluation.caption_batch --model_name <model> [<model> ...] [--num_shot ...]
    python -m src.evaluation.caption_summarize --model_name <model>
    python -m src.evaluation.pronoun --model_name <model> --num_shot <k> [--shot_seed 100]
    python -m src.evaluation.pronoun_batch --model_name <model> [--num_shot ...]
    python -m src.evaluation.pronoun_summarize --model_name <model>
    python -m src.evaluation.llm_as_judge --model_name <model> --num_shot <k> [--shot_seed 100]
    python -m src.evaluation.llm_as_judge_batch --model_name <model> [<model> ...] [--num_shot ...]
    python -m src.evaluation.qa_summarize --model_name <model>
All take --exp_type (default 1). Caption modules also take --data_name, --image_type and
--instruction; pronoun modules --data_type (OO, OP or ALL); judge and QA summary --prompt_mode
(ALL loops both modes in the batch judge and the summary); caption and judge evaluators
take --gpu_id.

Import evaluators from their submodules: caption evaluation needs CLIP and the judge needs
openai-harmony (idefics3 environment only), so this package re-exports only the summary helpers.
"""

from src.evaluation.summarize import aggregate_results, collect_results, summarize_runs

__all__ = [
    "aggregate_results",
    "collect_results",
    "summarize_runs",
]
