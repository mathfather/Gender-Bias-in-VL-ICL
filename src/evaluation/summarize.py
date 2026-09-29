import os

import pandas as pd

from src.config import EXP_FOLDERS, NUM_SHOTS, SUMMARY, select_shot_seeds
from src.utils import build_result_folder, load_json, mkdir


def collect_results(
    exp_type: str,
    data_name: str,
    data_type: str,
    model_name: str,
    instruction: str | None=None,
    zero_shot_data_type: str | None=None,
) -> pd.DataFrame:
    rows = []
    for num_shot in NUM_SHOTS:
        for shot_seed in select_shot_seeds(exp_type, num_shot):
            eval_path = os.path.join(build_result_folder(
                exp_type, data_name, data_type, shot_seed, model_name, num_shot, instruction
            ), "evaluation.json")
            if not os.path.exists(eval_path) and num_shot == 0 and zero_shot_data_type:
                eval_path = os.path.join(build_result_folder(
                    exp_type, data_name, zero_shot_data_type, shot_seed, model_name, num_shot,
                    instruction
                ), "evaluation.json")
            if not os.path.exists(eval_path):
                print(f"[SKIP] {eval_path} not found")
                continue
            rows.append({"num_shot": num_shot, "shot_seed": shot_seed, **load_json(eval_path)})
    return pd.DataFrame(rows)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c not in ("num_shot", "shot_seed")]
    grouped = df.groupby("num_shot")[metric_cols]
    mean_df = grouped.mean().add_suffix(":mean")
    std_df = grouped.std().add_suffix(":std")
    ordered_cols = [f"{c}:{stat}" for c in metric_cols for stat in ("mean", "std")]
    return pd.concat([mean_df, std_df], axis=1)[ordered_cols].reset_index()


def summarize_runs(
    exp_type: str,
    data_name: str,
    data_type: str,
    model_name: str,
    stem: str,
    instruction: str | None=None,
    zero_shot_data_type: str | None=None,
) -> None:
    """Writes summary/<setting>/<data_name>-<data_type>/<stem>_all.tsv and <stem>_agg.tsv."""
    out_dir = os.path.join(SUMMARY, EXP_FOLDERS[str(exp_type)], f"{data_name}-{data_type}")
    df = collect_results(
        exp_type, data_name, data_type, model_name, instruction, zero_shot_data_type
    )
    if df.empty:
        print(f"No results found for {stem}; {out_dir} is not written")
        return

    mkdir(out_dir)
    all_path = os.path.join(out_dir, f"{stem}_all.tsv")
    df.to_csv(all_path, sep="\t", index=False)
    print(f"[SAVED] {all_path}  ({len(df)} rows)")

    agg_df = aggregate_results(df)
    agg_path = os.path.join(out_dir, f"{stem}_agg.tsv")
    agg_df.to_csv(agg_path, sep="\t", index=False)
    print(f"[SAVED] {agg_path}  ({len(agg_df)} rows)")
