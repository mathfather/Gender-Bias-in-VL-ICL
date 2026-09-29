# https://huggingface.co/microsoft/Phi-3.5-vision-instruct

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, PreTrainedModel, ProcessorMixin

from src.model.base import ModelHelper, format_qa_turns
from src.model.utils import load_images
from src.prompt import PRONOUN_MAP, PRONOUN_QUERY_TEMPLATE, PRONOUN_TEMPLATE


class Phi35VHelper(ModelHelper):
    HIS_TOKEN_ID = 670
    HER_TOKEN_ID = 902

    @classmethod
    def load(cls, model_path: str, device: str="cuda"):
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, num_crops=4)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map=device,
            _attn_implementation="flash_attention_2",
        ).eval()
        return cls(model, processor)

    def __init__(self, model: PreTrainedModel, processor: ProcessorMixin) -> None:
        self.model = model
        self.processor = processor

    def prepare_caption_input(
        self,
        caption_list: list[str],
        image_list: list[str],
        instruction: str,
    ) -> dict:
        messages = []
        for image_idx, shot_caption in enumerate(caption_list[:-1], start=1):
            messages.append({"role": "user", "content": f"<|image_{image_idx}|>\n{instruction}"})
            messages.append({"role": "assistant", "content": shot_caption})
        query_idx = len(caption_list)
        messages.append({"role": "user", "content": f"<|image_{query_idx}|>\n{instruction}"})
        return {"messages": messages, "images": load_images(image_list)}

    def prepare_pronoun_input(
        self,
        occupations: list[str],
        genders: list[str],
        objects: list[str],
        image_list: list[str],
    ) -> dict:
        messages = []
        shots = zip(occupations[:-1], genders[:-1], objects[:-1], strict=True)
        for image_idx, (occupation, gender, obj) in enumerate(shots, start=1):
            pronoun = PRONOUN_MAP[gender]
            messages.append({"role": "user", "content": f"<|image_{image_idx}|>"})
            messages.append({
                "role": "assistant",
                "content": PRONOUN_TEMPLATE.format(occupation, pronoun, obj)
            })
        messages.append({"role": "user", "content": f"<|image_{len(occupations)}|>"})
        return {
            "messages": messages,
            "images": load_images(image_list),
            "query_prefix": PRONOUN_QUERY_TEMPLATE.format(occupations[-1]),
        }

    def prepare_qa_input(
        self,
        samples: list[dict],
        image_list: list[str],
        prompt_mode: str,
    ) -> dict:
        shots, query_text = format_qa_turns(samples, prompt_mode)
        messages = []
        for image_idx, (user_text, response) in enumerate(shots, start=1):
            messages.append({"role": "user", "content": f"<|image_{image_idx}|>\n{user_text}"})
            messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"<|image_{len(samples)}|>\n{query_text}"})
        return {"messages": messages, "images": load_images(image_list)}

    def generate(self, model_input: dict, max_new_tokens: int) -> str:
        prompt = self.processor.tokenizer.apply_chat_template(
            model_input["messages"], tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(prompt, model_input["images"], return_tensors="pt")
        inputs = inputs.to(self.model.device)
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            min_new_tokens=1,
            repetition_penalty=1,
            no_repeat_ngram_size=0,
            length_penalty=1,
            num_return_sequences=1,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def sample_next_token(self, model_input: dict) -> tuple[str, float, float]:
        prompt = self.processor.tokenizer.apply_chat_template(
            model_input["messages"], tokenize=False, add_generation_prompt=True
        )
        # The space before query_prefix is part of the published prompt.
        prompt += " " + model_input["query_prefix"]
        inputs = self.processor(prompt, model_input["images"], return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs)
            next_logits = out.logits[:, -1, :]
            return self._classify_gender(next_logits)
