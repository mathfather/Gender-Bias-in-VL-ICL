# https://huggingface.co/Qwen/Qwen-VL

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.model.base import ModelHelper
from src.prompt import PRONOUN_MAP, PRONOUN_QUERY_TEMPLATE, PRONOUN_TEMPLATE


class QwenVLHelper(ModelHelper):
    HIS_TOKEN_ID = 25235
    HER_TOKEN_ID = 1923

    @classmethod
    def load(cls, model_path: str, device: str="cuda"):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            bf16=True,
            device_map=device,
            low_cpu_mem_usage=True,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eod_id
        return cls(model, tokenizer)

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def prepare_caption_input(
        self,
        caption_list: list[str],
        image_list: list[str],
        instruction: str,
    ) -> BatchEncoding:
        parts = []
        for image, caption in zip(image_list[:-1], caption_list[:-1], strict=True):
            parts.append(f"<img>{image}</img> {instruction} {caption}")
        parts.append(f"<img>{image_list[-1]}</img> {instruction}")
        return self.tokenizer("".join(parts), return_tensors="pt")

    def prepare_pronoun_input(
        self,
        occupations: list[str],
        genders: list[str],
        objects: list[str],
        image_list: list[str],
    ) -> BatchEncoding:
        parts = []
        for image, occupation, gender, obj in zip(
            image_list[:-1], occupations[:-1], genders[:-1], objects[:-1], strict=True
        ):
            pronoun = PRONOUN_MAP[gender]
            parts.append(f"<img>{image}</img> {PRONOUN_TEMPLATE.format(occupation, pronoun, obj)}")
        query_prefix = PRONOUN_QUERY_TEMPLATE.format(occupations[-1])
        parts.append(f"<img>{image_list[-1]}</img> {query_prefix}")
        return self.tokenizer("".join(parts), return_tensors="pt")

    def generate(self, model_input: BatchEncoding, max_new_tokens: int) -> str:
        generated_output = self.model.generate(
            input_ids=model_input["input_ids"].to(self.model.device),
            attention_mask=model_input["attention_mask"].to(self.model.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            min_new_tokens=1,
            repetition_penalty=1,
            no_repeat_ngram_size=0,
            length_penalty=1,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eod_id,
            eos_token_id=self.tokenizer.eod_id,
        )
        return self.tokenizer.batch_decode(
            generated_output[:, model_input["input_ids"].size(1):],
            skip_special_tokens=True
        )[0].strip()

    def sample_next_token(self, model_input: BatchEncoding) -> tuple[str, float, float]:
        model_input = model_input.to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**model_input)
            next_logits = outputs.logits[:, -1, :]
            return self._classify_gender(next_logits)
