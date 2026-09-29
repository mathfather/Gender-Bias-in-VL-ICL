# https://github.com/OpenBMB/MiniCPM-o#in-context-few-shot-learning

import torch
from PIL import Image
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    BatchFeature,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)

from src.model.base import ModelHelper, format_qa_turns
from src.model.utils import load_images
from src.prompt import PRONOUN_MAP, PRONOUN_QUERY_TEMPLATE, PRONOUN_TEMPLATE


class MiniCPMHelper(ModelHelper):
    HIS_TOKEN_ID = 25235
    HER_TOKEN_ID = 1923

    @classmethod
    def load(cls, model_path: str, device: str="cuda"):
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        return cls(model, tokenizer, processor)

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        processor: ProcessorMixin,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor

    def _prepare_minicpm_input(self, messages: list[dict], query_prefix: str) -> BatchFeature:
        images = []
        chat = []
        for msg in messages:
            parts = []
            for content in msg["content"]:
                if isinstance(content, Image.Image):
                    images.append(content)
                    parts.append("(<image>./</image>)")
                elif isinstance(content, str):
                    parts.append(content)
                else:
                    raise TypeError(
                        f"Unsupported message content {type(content).__name__}; "
                        "expected PIL.Image or str."
                    )
            chat.append({"role": msg["role"], "content": "\n".join(parts)})
        prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=None,
        )
        # The space before query_prefix is part of the published prompt.
        prompt += " " + query_prefix
        return self.processor(
            [prompt],
            [images],
            [[]],
            [[]],
            max_slice_nums=1,
            use_image_id=False,
            return_tensors="pt",
            max_length=None,
        ).to(self.model.device)

    def prepare_caption_input(
        self,
        caption_list: list[str],
        image_list: list[str],
        instruction: str,
    ) -> list[dict]:
        images = load_images(image_list)
        messages = []
        for shot_caption, image in zip(caption_list[:-1], images[:-1], strict=True):
            messages.append({"role": "user", "content": [image, instruction]})
            messages.append({"role": "assistant", "content": [shot_caption]})
        messages.append({"role": "user", "content": [images[-1], instruction]})
        return messages

    def prepare_pronoun_input(
        self,
        occupations: list[str],
        genders: list[str],
        objects: list[str],
        image_list: list[str],
    ) -> dict:
        images = load_images(image_list)
        messages = []
        for image, occupation, gender, obj in zip(
            images[:-1], occupations[:-1], genders[:-1], objects[:-1], strict=True
        ):
            pronoun = PRONOUN_MAP[gender]
            messages.append({"role": "user", "content": [image]})
            messages.append({
                "role": "assistant",
                "content": [PRONOUN_TEMPLATE.format(occupation, pronoun, obj)]
            })
        messages.append({"role": "user", "content": [images[-1]]})
        return {
            "messages": messages,
            "query_prefix": PRONOUN_QUERY_TEMPLATE.format(occupations[-1]),
        }

    def prepare_qa_input(
        self,
        samples: list[dict],
        image_list: list[str],
        prompt_mode: str,
    ) -> list[dict]:
        images = load_images(image_list)
        shots, query_text = format_qa_turns(samples, prompt_mode)
        messages = []
        for (user_text, response), image in zip(shots, images[:-1], strict=True):
            messages.append({"role": "user", "content": [image, user_text]})
            messages.append({"role": "assistant", "content": [response]})
        messages.append({"role": "user", "content": [images[-1], query_text]})
        return messages

    def generate(self, model_input: list[dict], max_new_tokens: int) -> str:
        return self.model.chat(
            msgs=model_input,
            tokenizer=self.tokenizer,
            sampling=False,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            num_beams=1,
            repetition_penalty=1,
        )

    def sample_next_token(self, model_input: dict) -> tuple[str, float, float]:
        inputs = self._prepare_minicpm_input(model_input["messages"], model_input["query_prefix"])
        with torch.inference_mode():
            inputs_embeds, _ = self.model.get_vllm_embedding(inputs)
            outputs = self.model.llm(
                input_ids=None,
                position_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=inputs["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
            next_logits = outputs.logits[:, -1, :]
            return self._classify_gender(next_logits)
