# Installation

Related documents and sources:

- [requirements-qwenvl.txt](../requirements-qwenvl.txt), [requirements-idefics3.txt](../requirements-idefics3.txt), [requirements-diffusion.txt](../requirements-diffusion.txt)
- [models.md](models.md)
- [experiments.md](experiments.md)
- `src/model/`, `src/evaluation/llm_as_judge.py`
- https://docs.astral.sh/uv/

Running the models needs Linux, CUDA GPUs with bfloat16 support, and Python 3.10 to 3.12 (the pinned torch 2.3.1 has no wheels for Python 3.13). Environments are created with conda. The six models need two environments with mutually incompatible `transformers` pins (4.44.2 and 4.57.1).

## 1. Environment `qwenvl`

This environment runs Qwen-VL, MiniCPM-o 2.6, and Phi-3.5-vision-instruct.

```bash
conda create -n qwenvl python=3.10 -y && conda activate qwenvl
uv pip install -r requirements-qwenvl.txt
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## 2. Environment `idefics3`

This environment runs Idefics3-8B-Llama3, InternVL3.5-8B, Qwen3-VL-8B-Instruct, and the GPT-OSS-20B judge.

```bash
conda create -n idefics3 python=3.11 -y && conda activate idefics3
uv pip install -r requirements-idefics3.txt
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## 3. Environment `diffusion` (optional)

This environment generates the synthetic in-context images with FLUX.1-dev and Stable Diffusion 3.5 Large; only the bias mitigation experiment needs it.

```bash
conda create -n diffusion python=3.11 -y && conda activate diffusion
uv pip install -r requirements-diffusion.txt
```

## 4. NLTK data

Run once in every environment; data preparation, inference, evaluation, and image generation all tokenize with NLTK:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## 5. Notes

- Run every command from the repository root; imports resolve relative to it.
- Five of the six loaders hard-code flash-attention 2. Without a working flash-attn build, switch the attention implementation at `src/model/minicpm.py:32`, `src/model/idefics3.py:23`, `src/model/phi35v.py:23`, and `src/model/qwen3vl.py:29` (set `"sdpa"`), and at `src/model/internvl35.py:117` (set `use_flash_attn=False`). Qwen-VL needs no change.
- Qwen-VL, MiniCPM-o 2.6, Phi-3.5-vision-instruct, and InternVL3.5-8B load their snapshot's modeling code with `trust_remote_code=True`.
- The GPT-OSS-20B judge checkpoint is stored in MXFP4. With the pinned torch 2.3.1 it is dequantized and loaded in bfloat16, which takes about 40 GB of VRAM.
