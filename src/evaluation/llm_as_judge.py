import argparse
import os
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from openai_harmony import (
    Conversation,
    HarmonyEncoding,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    load_harmony_encoding,
)
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.config import (
    MODEL,
    NUM_SHOTS,
    QA_MODELS,
    QA_PROMPT_MODES,
    RANDOM_EXP_TYPES,
    SHOT_SEEDS,
    TQDM_STYLE,
    VISUALCOT_DATA_TYPE,
)
from src.data import load_valid_data
from src.metric import qa_error_rates
from src.prompt import JUDGE_PROMPT_TEMPLATE
from src.utils import build_result_folder, present_args, save_json, select_device

JUDGE_MODEL_PATH = os.path.join(MODEL, "gpt-oss-20b")
JUDGE_MAX_NEW_TOKENS = 128
MAX_ANSWER_WORDS = 10
SYSTEM_CONTENT = SystemContent.new().with_reasoning_effort(ReasoningEffort.LOW)
SCORE_PATTERN = re.compile(r"score:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


def load_judge(device: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, HarmonyEncoding]:
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_PATH)
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    # Pinned torch 2.3.1 / triton 2.3.1 lack MXFP4 kernels: transformers raises or dequantizes.
    try:
        model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL_PATH, device_map=device)
        print(f"Judge model loaded with default config (MXFP4) on {device}.")
    except Exception as e:
        print(f"MXFP4 loading failed ({e}), falling back to bfloat16.")
        model = AutoModelForCausalLM.from_pretrained(
            JUDGE_MODEL_PATH, torch_dtype=torch.bfloat16, quantization_config=None,
            device_map=device
        )
        print(f"Judge model loaded with bfloat16 on {device}.")
    return model, tokenizer, encoding


def parse_score(text: str) -> float:
    match = SCORE_PATTERN.search(text)
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    return 0.0


def exact_match(reference_answer: str, generated_answer: str) -> bool:
    return str(reference_answer).strip().lower() == str(generated_answer).strip().lower()


def judge_single(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    encoding: HarmonyEncoding,
    question: str,
    reference_answer: str,
    generated_answer: str,
) -> float:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, reference_answer=reference_answer, generated_answer=generated_answer
    )
    conversation = Conversation.from_messages([
        Message.from_role_and_content(Role.SYSTEM, SYSTEM_CONTENT),
        Message.from_role_and_content(Role.USER, prompt),
    ])
    token_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

    input_ids = torch.tensor([token_ids], device=model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, attention_mask=attention_mask, max_new_tokens=JUDGE_MAX_NEW_TOKENS,
            do_sample=False
        )

    new_tokens = output_ids[0, input_ids.shape[1]:]
    return float(parse_score(tokenizer.decode(new_tokens, skip_special_tokens=True)))


def eval_qa(
    args: argparse.Namespace | SimpleNamespace,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    encoding: HarmonyEncoding,
    valid_data: dict[str, dict],
) -> None:
    """Adds judge_score to the run's result.tsv and writes its evaluation.json."""
    result_folder = build_result_folder(
        args.exp_type, "visualcot", VISUALCOT_DATA_TYPE, args.shot_seed, args.model_name,
        args.num_shot, args.prompt_mode
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
    image_ids = result["image_id"].astype(str).tolist()
    result["detected_gender"] = [valid_data[i]["detected_gender"] for i in image_ids]

    scores = []
    tags = []
    pbar = tqdm(result.iterrows(), total=len(result), bar_format=TQDM_STYLE, desc="Judging")
    for _, row in pbar:
        ref_ans = str(row["reference_answer"])
        gen_ans = str(row["generated_answer"])
        if exact_match(ref_ans, gen_ans):
            score, tag = 1.0, "EM"
        elif len(gen_ans.split()) > MAX_ANSWER_WORDS:
            score, tag = 0.0, "SKIP"
        else:
            score = judge_single(model, tokenizer, encoding, row["question"], ref_ans, gen_ans)
            tag = "LLM"
        scores.append(score)
        tags.append(tag)
        pbar.set_postfix_str(f"[{tag}] score={score:.2f} | ref={ref_ans[:30]} | gen={gen_ans[:50]}")
    print(
        f"Exact match: {tags.count('EM')}/{len(scores)} | LLM judge: {tags.count('LLM')}/"
        f"{len(scores)} | Skipped (too long): {tags.count('SKIP')}/{len(scores)}"
    )

    scores_arr = np.array(scores)
    result["judge_score"] = scores_arr
    evaluation = qa_error_rates(scores_arr, result["detected_gender"])

    result.to_csv(result_path, sep="\t", header=True, index=False)
    save_json(evaluation, eval_path)
    print(f"Saved evaluation to {eval_path}: {evaluation}")


def main(args: argparse.Namespace) -> None:
    device = select_device(args.gpu_id)
    model, tokenizer, encoding = load_judge(device)
    eval_qa(args, model, tokenizer, encoding, load_valid_data("visualcot"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_type", type=str, default="1", choices=RANDOM_EXP_TYPES)
    parser.add_argument("--shot_seed", type=int, default=SHOT_SEEDS[0], choices=SHOT_SEEDS)
    parser.add_argument("--model_name", type=str, required=True, choices=QA_MODELS)
    parser.add_argument("--num_shot", type=int, required=True, choices=NUM_SHOTS)
    parser.add_argument("--prompt_mode", type=str, default="advanced", choices=QA_PROMPT_MODES)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    present_args(args)
    main(args)
