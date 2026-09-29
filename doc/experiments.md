# Experiments

Related documents and sources:

- [installation.md](installation.md)
- [models.md](models.md)
- [datasets.md](datasets.md)
- `src/config.py`, `src/utils.py`, `src/sampling/`, `src/inference/`, `src/evaluation/`, `utils/`

Every entry point is a `python -m` module with `--help`, run from the repository root. The pipeline of a task is inference, then evaluation, then summary aggregation. Single-run modules take one `(num_shot, shot_seed)`; the `*_batch` drivers loop over shot counts and seeds with the convention of the paper, encoded in `select_shot_seeds` in `src/config.py`: 0-shot runs use seed 100 only, settings 1 to 4 run seeds 100 to 104, and settings 5 and 6 run seed 100 only because retrieval is deterministic. The inference seed is 42 everywhere, and decoding is greedy.

## 1. Settings

The settings follow the paper: Random Sample (RS), Male-only Sample (MS), Female-only Sample (FS), Balanced Sample (BS), Similarity-based Image-Image Retrieval (SIIR), and Similarity-based Image-Text Retrieval (SITR). Each has a `--exp_type` value and a results folder; settings 5 and 6 run on COCOBias only and need the rankings of Sec. 6.

| `--exp_type` | Setting | Results folder            | Requires                  |
| ------------ | ------- | ------------------------- | ------------------------- |
| 1            | RS      | `1-random`                | -                         |
| 2            | MS      | `2-random_male`           | -                         |
| 3            | FS      | `3-random_female`         | -                         |
| 4            | BS      | `4-random_balanced`       | -                         |
| 5            | SIIR    | `5-image_image_retrieval` | `<image_type>-II-cos.pkl` |
| 6            | SITR    | `6-image_text_retrieval`  | `<image_type>-IT-cos.pkl` |

## 2. In-context image sources

`--image_type` selects the images of the in-context examples in the captioning pipeline; the query images are always the real images, and the synthetic images are generated from the captions of the training pool (see [data/coco2014/README.md](../data/coco2014/README.md) Sec. 3). VisoGender and VisualCoT have no such flag.

| `--image_type`  | In-context images                                 | Settings |
| --------------- | ------------------------------------------------- | -------- |
| `coco`          | real COCO val2014                                 | 1 to 6   |
| `coco-balanced` | real COCO val2014, gender-balanced 267M/267F pool | 5 and 6  |
| `flux`          | FLUX.1-dev images                                 | 2 and 3  |
| `sd35l`         | SD35L images                                      | 2 and 3  |
| `dci`           | real DCI images                                   | 1 to 4   |

The caption modules for inference, evaluation, and summary aggregation reject any other combination of `--exp_type`, `--data_name`, and `--image_type` (`check_caption_run` in `src/utils.py`). The paper runs `flux` and `sd35l` with `--num_shot 2 4 8` on Idefics3-8B-Llama3, Phi-3.5-vision-instruct, MiniCPM-o 2.6, and Qwen3-VL-8B-Instruct (`--model_name` idefics3, Phi35V, MiniCPM, Qwen3VL); their 0-shot rows come from the `coco` runs, because the summary falls back to `cocobias-coco` for 0-shot.

## 3. Image captioning (COCOBias and DCI)

```bash
# inference -> results/<setting>/cocobias-coco-<seed>/<model>-<k>shot-I1/result.tsv
python -m src.inference.caption_batch --model_name Phi35V --exp_type 2 \
    --data_name cocobias --image_type coco --num_shot 0 2 4 6 8 --gpu_id 0

# evaluation (adds error rates, reveal rates, BLEU-4, CLIPScore, and caption length to each run)
python -m src.evaluation.caption_batch --model_name Phi35V --exp_type 2 \
    --data_name cocobias --image_type coco --num_shot 0 2 4 6 8 --gpu_id 0

# aggregation over shots and seeds -> summary/<setting>/cocobias-coco/Phi35V_{all,agg}.tsv
python -m src.evaluation.caption_summarize --model_name Phi35V --exp_type 2 \
    --data_name cocobias --image_type coco
```

For DCI pass `--data_name dci --image_type dci`. The caption instruction is fixed to `I1`, "Describe the image in one sentence.", which also names the `-I1` result folder suffix. Generation stops after 50 new tokens and the caption is cut to its first sentence (NLTK `sent_tokenize`). CLIPScore is 100 times the cosine similarity of CLIP ViT-B/32 image and caption features, clipped at zero.

## 4. Pronoun prediction (VisoGender)

Pronoun prediction runs settings 1 to 4. `--data_type` selects the OO or OP subtask (`ALL` runs both).

```bash
python -m src.inference.pronoun_batch --model_name idefics3 --exp_type 3 \
    --data_type ALL --num_shot 0 2 4 6 8 --gpu_id 0
python -m src.evaluation.pronoun_batch --model_name idefics3 --exp_type 3 --data_type ALL
python -m src.evaluation.pronoun_summarize --model_name idefics3 --exp_type 3 --data_type ALL
```

## 5. VQA (VisualCoT)

VQA runs settings 1 to 4 on MiniCPM-o 2.6, Phi-3.5-vision-instruct, and Qwen3-VL-8B-Instruct (`--model_name` MiniCPM, Phi35V, Qwen3VL). `--prompt_mode advanced` is chain-of-thought prompting and `base` is direct prompting; generation stops after 256 new tokens. The query question is the gender-neutralized one, while the in-context questions keep their original wording.

```bash
python -m src.inference.qa_batch --model_name Qwen3VL --exp_type 2 \
    --prompt_mode advanced --num_shot 0 2 4 6 8 --gpu_id 0
python -m src.evaluation.llm_as_judge_batch --model_name Qwen3VL --exp_type 2 \
    --prompt_mode advanced --num_shot 0 2 4 6 8 --gpu_id 0
python -m src.evaluation.qa_summarize --model_name Qwen3VL --exp_type 2 --prompt_mode ALL
```

The judge is GPT-OSS-20B (`models/gpt-oss-20b`, `idefics3` environment) with the Visual-CoT scoring prompt, low reasoning effort, and a budget of 128 new tokens. It receives the original question, the reference answer, and the generated answer.

## 6. Retrieval rankings (settings 5 and 6)

SIIR and SITR need the CLIP similarity rankings between the COCOBias validation and training images, computed once:

```bash
python -m src.sampling.sim_retrieval --sim_type II --gpu_id 0   # data/coco2014/coco-II-cos.pkl
python -m src.sampling.sim_retrieval --sim_type IT --gpu_id 0   # data/coco2014/coco-IT-cos.pkl
python -m utils.balance_train_pool                              # captions_bias_train_balanced.json (267M/267F)
python -m utils.filter_similarity_balanced                      # coco-balanced-{II,IT}-cos.pkl
```

## 7. Output layout

```
results/<setting>/<dataset>-<type>-<seed>/<model>-<k>shot[-<instruction|prompt_mode>]/
├── result.tsv        # per-sample generations, plus per-sample metrics after evaluation
├── args.json         # arguments of the run and the selected shot ids
└── evaluation.json   # aggregate metrics of the run

summary/<setting>/<dataset>-<type>/<model>[_<prompt_mode>]_{all,agg}.tsv
```

Two examples are `results/2-random_male/cocobias-coco-100/Phi35V-8shot-I1/result.tsv` and `summary/3-random_female/visogender-OP/idefics3_agg.tsv`. The `_all` summary has one row per `(num_shot, shot_seed)` and the `_agg` summary the mean and standard deviation over seeds per `num_shot` (columns `<key>:mean` and `<key>:std`). The keys of `evaluation.json` map to the paper's metrics as follows. `MR-*` and `RR-*` values are percentages, `BLEU-*` and `CLIPScore-*` are scores on a 0 to 100 scale, and `NW-*` are token counts.

| Key                                                          | Paper                                       |
| ------------------------------------------------------------ | ------------------------------------------- |
| `MR-M`, `MR-F`, `MR-O`, `MR-M-F`                             | ER_m, ER_f, ER_o, ER_m-f                    |
| `MR-M-Rev`, `MR-F-Rev`, `MR-O-Rev`, `MR-M-F-Rev`             | the same error rates over revealed captions |
| `RR-M`, `RR-F`, `RR-O`                                       | RR_m, RR_f, RR_o                            |
| `BLEU-M`, `BLEU-F`, `BLEU-O`, `BLEU-M-F`                     | BLEU-4 per gender, overall, and the gap     |
| `CLIPScore-M`, `CLIPScore-F`, `CLIPScore-O`, `CLIPScore-M-F` | CLIPScore per gender, overall, and the gap  |
| `NW-O`, `NW-M`, `NW-F`, `NW-M-F`                             | AvgL overall, per gender, and the gap       |

Evaluation adds `detected_gender`, `predicted_gender`, `BLEU-4`, `CLIPScore`, and `n_word` to the `result.tsv` of a captioning run, and `detected_gender` and `judge_score` to that of a VQA run.

## 8. Reproducibility

The dataset splits (seed 42), the inference seed (42), shot sampling (seeds 100 to 104 over a candidate pool of up to 100 examples, where the k-shot sequence is a prefix of the (k+2)-shot sequence), greedy decoding, and the retrieval rankings given fixed CLIP features are seeded and deterministic. The diffusion-generated in-context images, the judge scores, CLIP features across GPU architectures, and the VisoGender image set (it depends on which source URLs resolve) are not bit-reproducible. Tokenization depends on the NLTK version, which we pin it to `nltk==3.9.2`.
