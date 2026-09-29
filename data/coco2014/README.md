# COCOBias (COCO 2014)

Image-captioning bias evaluation set built from MS-COCO 2014 val: images with exactly one person whose caption-detected gender agrees with the human gender annotation. The training pool has 805 images (538 male / 267 female) and the gender-balanced validation set 2,130.

## 1. Download

| What                                             | Source                                                                                                                                                                        | Place at                                                                                       |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| COCO 2014 val images                             | `http://images.cocodataset.org/zips/val2014.zip`                                                                                                                              | unzip so images sit at `data/coco2014/images/val2014/*.jpg`                                    |
| COCO 2014 annotations                            | `http://images.cocodataset.org/annotations/annotations_trainval2014.zip`                                                                                                      | extract `captions_val2014.json` and `instances_val2014.json` into `data/coco2014/annotations/` |
| Gender/skin annotations (Zhao et al., ICCV 2021) | request via the [Princeton Visual AI data-request form](https://forms.gle/y179Fk63dgTWo9cSA) on the [project page](https://princetonvisualai.github.io/imagecaptioning-bias/) | place `images_val2014.csv` in `data/coco2014/bias/`                                            |

The gender/skin annotations are distributed for non-commercial research only and may not be redistributed, which is why they are not bundled here. Request the file through the project-page form or from the authors; only `images_val2014.csv` is needed (the instance-level CSV of the package is unused).

## 2. Build the splits

```bash
python -m src.data.cocobias          # captions_val.json -> captions_bias.json -> train/valid splits
python -m utils.balance_train_pool   # captions_bias_train_balanced.json (267M/267F, image_type "coco-balanced")
```

The build keeps an image when its captions agree on one gender, it has exactly one person bounding box, its gender annotation is `Male` or `Female` and agrees with the captions, and its skin annotation is present (rows without a skin label are dropped as well). 20% of each gender goes to the training pool; the rest forms the validation set, whose majority gender is down-sampled to balance it.

## 3. Optional: synthetic in-context images

The bias mitigation experiment (`--image_type flux` or `sd35l`) replaces the training-pool images with images generated from a gendered caption of each training image. The model weights are listed in [doc/models.md](../../doc/models.md), and the commands run in the `diffusion` environment. Generation is seeded but not bit-reproducible across hardware, and SD35L draws its noise from the global random state, so its images also depend on the generation order within a run.

```bash
python -m utils.generate_diffusion_images --model_name FLUX.1-dev                     # images/FLUX.1-dev/
python -m utils.generate_diffusion_images --model_name stable-diffusion-3.5-large     # images/stable-diffusion-3.5-large/
```

## 4. Optional: retrieval rankings

The similarity-based retrieval settings SIIR and SITR (`--exp_type` 5 and 6) need the CLIP ViT-B/32 rankings, computed from the images of Sec. 1 and the splits of Sec. 2; CLIP is fetched on first use.

```bash
python -m src.sampling.sim_retrieval --sim_type II   # coco-II-cos.pkl
python -m src.sampling.sim_retrieval --sim_type IT   # coco-IT-cos.pkl
python -m utils.filter_similarity_balanced           # coco-balanced-{II,IT}-cos.pkl
```

## 5. Final layout

```
data/coco2014/
├── annotations/{captions_val2014.json, instances_val2014.json}
├── bias/images_val2014.csv
├── images/{val2014, FLUX.1-dev, stable-diffusion-3.5-large}/*.jpg   (the last two only if generated)
├── captions_val.json                   (built)
├── captions_bias.json                  (built)
├── captions_bias_train.json            (built, 805)
├── captions_bias_train_balanced.json   (built, 534)
├── captions_bias_valid.json            (built, 2130)
└── coco-{II,IT}-cos.pkl, coco-balanced-{II,IT}-cos.pkl   (built, optional)
```
