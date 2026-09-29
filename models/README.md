# Models

Download the model weights with `utils/download_huggingface_model.py`; [doc/models.md](../doc/models.md) lists the commands, the CLI keys, and the environments.

The six LVLM folder names must match `MODEL_MAP` in `src/model/utils.py` (the download script applies the two renames). The judge and diffusion folders keep the last segment of their repo ids. After downloading, the folder looks like this:

```
models/
├── QwenVL/                       # Qwen/Qwen-VL (renamed by the download script)
├── MiniCPM/                      # openbmb/MiniCPM-o-2_6 (renamed by the download script)
├── Phi-3.5-vision-instruct/      # microsoft/Phi-3.5-vision-instruct
├── Idefics3-8B-Llama3/           # HuggingFaceM4/Idefics3-8B-Llama3
├── InternVL3_5-8B/               # OpenGVLab/InternVL3_5-8B
├── Qwen3-VL-8B-Instruct/         # Qwen/Qwen3-VL-8B-Instruct
├── gpt-oss-20b/                  # unsloth/gpt-oss-20b (VQA judge)
├── FLUX.1-dev/                   # optional, gated; bias mitigation experiment
└── stable-diffusion-3.5-large/   # optional, gated; bias mitigation experiment
```
