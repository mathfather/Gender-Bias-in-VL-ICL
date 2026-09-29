# VisualCoT (GQA)

VQA bias evaluation set built from the GQA subset of VisualCoT: single-person images with gendered questions, evaluated with gender-neutralized query questions. The training pool has 1,068 samples (804 male / 264 female) and the gender-balanced validation set 2,112.

## 1. Download

| What                 | Source                                                                                                                                                                                         | Place at                                                                                                                                                            |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CoT annotations      | [`deepcs233/Visual-CoT`](https://huggingface.co/datasets/deepcs233/Visual-CoT): both files of the `cot_with_detailed_reasoning_steps/` folder (not the similarly named file under `metadata/`) | `gqa_cot_train.jsonl` to `data/visual-cot/annotations/gqa_cot_train.jsonl`; `gqa_cot_val.jsonl` renamed to `data/visual-cot/annotations/gqa_cot_val_detailed.jsonl` |
| GQA scene graphs     | `https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip`                                                                                                                               | extract `train_sceneGraphs.json` into `data/visual-cot/annotations/sceneGraphs/`                                                                                    |
| GQA images (20.3 GB) | `https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip` ([GQA downloads](https://cs.stanford.edu/people/dorarad/gqa/download.html))                                                        | place all `.jpg` files directly in `data/visual-cot/images/`                                                                                                        |

Direct URLs of the CoT files:

```bash
curl -L -o data/visual-cot/annotations/gqa_cot_train.jsonl \
  "https://huggingface.co/datasets/deepcs233/Visual-CoT/resolve/main/cot_with_detailed_reasoning_steps/gqa_cot_train.jsonl"
curl -L -o data/visual-cot/annotations/gqa_cot_val_detailed.jsonl \
  "https://huggingface.co/datasets/deepcs233/Visual-CoT/resolve/main/cot_with_detailed_reasoning_steps/gqa_cot_val.jsonl"
```

## 2. Build the splits

```bash
python -m src.data.visualcot   # cot_bias.json -> train (1,068) / valid (2,112)
```

The build combines the train and val CoT files, keeps single-person images (GQA scene graphs) whose question contains a gender word, drops "who" questions and multi-person text, keeps only questions whose gender words can be neutralized by token replacement, and keeps the longest question per image. Question neutralization (`neutral_question` on valid entries) is rule-based and deterministic. Zero-shot VQA prepends the non-human format example `2383701.jpg`, which comes with the GQA images.

## 3. Final layout

```
data/visual-cot/
├── annotations/
│   ├── gqa_cot_train.jsonl
│   ├── gqa_cot_val_detailed.jsonl   (renamed from gqa_cot_val.jsonl)
│   └── sceneGraphs/train_sceneGraphs.json
├── images/*.jpg
├── cot_bias.json          (built)
├── cot_bias_train.json    (built, 1068)
└── cot_bias_valid.json    (built, 2112)
```
