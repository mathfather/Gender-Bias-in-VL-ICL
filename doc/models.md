# Models

Related documents and sources:

- [models/README.md](../models/README.md)
- [installation.md](installation.md)
- [experiments.md](experiments.md)
- `utils/download_huggingface_model.py`, `src/model/utils.py`
- https://huggingface.co/models

Every model loads from a local folder under `models/`. The download script stores each Hugging Face snapshot at `models/<last segment of the repo id>` and applies the two folder renames the loaders expect (`Qwen/Qwen-VL` to `models/QwenVL`, `openbmb/MiniCPM-o-2_6` to `models/MiniCPM`). [models/README.md](../models/README.md) shows the resulting layout.

```bash
# evaluated LVLMs (one command per model)
python -m utils.download_huggingface_model --repo_id Qwen/Qwen-VL
python -m utils.download_huggingface_model --repo_id openbmb/MiniCPM-o-2_6
python -m utils.download_huggingface_model --repo_id microsoft/Phi-3.5-vision-instruct
python -m utils.download_huggingface_model --repo_id HuggingFaceM4/Idefics3-8B-Llama3
python -m utils.download_huggingface_model --repo_id OpenGVLab/InternVL3_5-8B
python -m utils.download_huggingface_model --repo_id Qwen/Qwen3-VL-8B-Instruct

# auxiliary models
python -m utils.download_huggingface_model --repo_id unsloth/gpt-oss-20b                     # VQA judge
python -m utils.download_huggingface_model --repo_id black-forest-labs/FLUX.1-dev            # gated; bias mitigation only
python -m utils.download_huggingface_model --repo_id stabilityai/stable-diffusion-3.5-large  # gated; bias mitigation only
```

## 1. Evaluated LVLMs

The CLI key is the `--model_name` value of every driver; the local folder is the value of `MODEL_MAP` in `src/model/utils.py`.

| CLI key      | Model                   | Hugging Face repo                   | Local folder              | Environment |
| ------------ | ----------------------- | ----------------------------------- | ------------------------- | ----------- |
| `QwenVL`     | Qwen-VL                 | `Qwen/Qwen-VL`                      | `QwenVL`                  | qwenvl      |
| `MiniCPM`    | MiniCPM-o 2.6           | `openbmb/MiniCPM-o-2_6`             | `MiniCPM`                 | qwenvl      |
| `Phi35V`     | Phi-3.5-vision-instruct | `microsoft/Phi-3.5-vision-instruct` | `Phi-3.5-vision-instruct` | qwenvl      |
| `idefics3`   | Idefics3-8B-Llama3      | `HuggingFaceM4/Idefics3-8B-Llama3`  | `Idefics3-8B-Llama3`      | idefics3    |
| `InternVL35` | InternVL3.5-8B          | `OpenGVLab/InternVL3_5-8B`          | `InternVL3_5-8B`          | idefics3    |
| `Qwen3VL`    | Qwen3-VL-8B-Instruct    | `Qwen/Qwen3-VL-8B-Instruct`         | `Qwen3-VL-8B-Instruct`    | idefics3    |

All six repos are public and ungated. VQA runs on MiniCPM-o 2.6, Phi-3.5-vision-instruct, and Qwen3-VL-8B-Instruct only (`--model_name` MiniCPM, Phi35V, Qwen3VL).

## 2. Auxiliary models

| Purpose                       | Repo or package                          | Local folder                 | Access |
| ----------------------------- | ---------------------------------------- | ---------------------------- | ------ |
| VQA judge (GPT-OSS-20B)       | `unsloth/gpt-oss-20b`                    | `gpt-oss-20b`                | public |
| synthetic images (FLUX.1-dev) | `black-forest-labs/FLUX.1-dev`           | `FLUX.1-dev`                 | gated  |
| synthetic images (SD35L)      | `stabilityai/stable-diffusion-3.5-large` | `stable-diffusion-3.5-large` | gated  |
| CLIPScore, SIIR, and SITR     | `ViT-B/32` through the `clip` package    | downloaded on first use      | public |

FLUX.1-dev and SD35L are gated on Hugging Face: accept the terms on their model pages and run `huggingface-cli login` before downloading.
