# DCI (Densely Captioned Images)

Image-captioning bias evaluation set built from Meta's DCI: single-person images whose short and extra captions agree on one gender. The training pool has 75 images (45 male / 30 female) and the gender-balanced validation set 238.

## 1. Download

| What                  | Source                                                                                                                                     | Place at                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| DCI annotations       | `https://dl.fbaipublicfiles.com/densely_captioned_images/dci.tar.gz` (see [facebookresearch/DCI](https://github.com/facebookresearch/DCI)) | extract into `data/dci/` so the archive's folder sits at `data/dci/densely_captioned_images/` |
| Photos (SA-1B subset) | follow the DCI instructions: accept Meta's SA-1B license and download the referenced archive (`sa_000138.tar`)                             | place the `sa_*.jpg` files directly in `data/dci/images/`                                     |

## 2. Build the splits

Place the images first: the build drops entries whose image is missing on disk.

```bash
python -m utils.slim_dci_annotations   # merges <id>-data.json files, drops masks -> annotations/annotations.json
python -m src.data.dci                 # captions_bias.json (373) -> train (75) / valid (238)
```

The automatic filter keeps 373 images. Two entries are then excluded from the validation split (`VALID_EXCLUDE_IDS` in `src/data/dci.py`): one whose captions describe a statue rather than a person, and one female entry dropped to keep the validation set gender-balanced after that removal. The paper's count of 371 images is the set after these two exclusions.

## 3. Final layout

```
data/dci/
├── densely_captioned_images/    (extracted archive; can be deleted after slimming)
├── annotations/annotations.json (built, 7805 entries)
├── images/sa_*.jpg
├── captions_bias.json           (built, 373)
├── captions_bias_train.json     (built, 75)
└── captions_bias_valid.json     (built, 238)
```
