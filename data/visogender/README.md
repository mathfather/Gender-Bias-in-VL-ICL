# VisoGender

Pronoun-prediction benchmark of occupation images, evaluated by comparing the next-token probabilities of "his" and "her". It has two subtasks, Occupation-Object (OO) and Occupation-Participant (OP), and its training pools are gender-balanced by construction (OO 46/46, OP 92/92).

## 1. Download the raw annotation TSVs

From [oxai/visogender](https://github.com/oxai/visogender); the file names must match exactly:

```bash
curl -L -o data/visogender/annotations/OO_Visogender_10052025.tsv \
  https://raw.githubusercontent.com/oxai/visogender/main/data/visogender_data/OO/OO_Visogender_10052025.tsv
curl -L -o data/visogender/annotations/OP_Visogender_11012024.tsv \
  https://raw.githubusercontent.com/oxai/visogender/main/data/visogender_data/OP/OP_Visogender_11012024.tsv
```

## 2. Download the images and build the splits

```bash
python -m src.data.visogender --download_images
```

This fetches every image URL listed in the TSVs into `images/` (files that already exist are skipped), then builds the OO and OP train/valid splits over the images that downloaded: two images per occupation and gender for OO, two per occupation, occupation gender, and participant gender for OP. A download is stored as `<IDX>.webp` when it is a WebP file and as `<IDX>.jpeg` for every other image format, which is the naming of the paper's image set. A download that is not an image, such as an HTML page answered by a dead link, is discarded.

For the paper, 670 of the 690 source URLs resolved. The split construction is seeded, but its input is the set of downloaded images, so if fewer images download, both the train and the valid splits differ from the paper's (OO 92/138, OP 184/256).

## 3. Final layout

```
data/visogender/
├── annotations/{OO_Visogender_10052025.tsv, OP_Visogender_11012024.tsv}
├── images/*.{jpeg,webp}
├── oo_train.{json,tsv}, oo_valid.{json,tsv}   (built)
└── op_train.{json,tsv}, op_valid.{json,tsv}   (built)
```
