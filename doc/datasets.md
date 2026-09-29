# Datasets

Related documents and sources:

- [data/README.md](../data/README.md)
- [data/coco2014/README.md](../data/coco2014/README.md), [data/dci/README.md](../data/dci/README.md), [data/visogender/README.md](../data/visogender/README.md), [data/visual-cot/README.md](../data/visual-cot/README.md)
- [experiments.md](experiments.md)
- `src/data/`, `utils/`

Every split is built with seed 42 from the original sources by the preparation code in `src/data/` and `utils/`; each dataset README gives the download links and build commands, and [data/README.md](../data/README.md) shows the expected on-disk layout.

## 1. Overview

| Folder                                      | Dataset (task)                  | You download                                                     | You build                           |
| ------------------------------------------- | ------------------------------- | ---------------------------------------------------------------- | ----------------------------------- |
| [coco2014/](../data/coco2014/README.md)     | COCOBias (image captioning)     | COCO val2014 images and annotations, gender and skin annotations | `captions_bias_*` train/valid JSONs |
| [dci/](../data/dci/README.md)               | DCI (image captioning)          | `dci.tar.gz`, SA-1B photo subset                                 | slim annotations, train/valid JSONs |
| [visogender/](../data/visogender/README.md) | VisoGender (pronoun prediction) | two raw TSVs (images fetched by the build script)                | OO/OP train/valid splits            |
| [visual-cot/](../data/visual-cot/README.md) | VisualCoT (VQA, GQA images)     | CoT jsonls, GQA scene graphs and images                          | `cot_bias_*` train/valid JSONs      |

The DCI build drops entries whose image is missing on disk, so the SA-1B photos go in place before the build. The VisoGender build fetches its images itself (`--download_images`) and drops the entries whose download fails. The COCOBias and VisualCoT splits are built from annotations only. The downloads take about 6.5 GB for COCO, 12 GB for DCI (0.9 GB of annotations plus one SA-1B shard), under 1 GB for VisoGender, and 21 GB for VisualCoT.

Two sources are not open downloads. The COCO gender and skin annotations are requested through a form and may not be redistributed (see [data/coco2014/README.md](../data/coco2014/README.md)). The SA-1B photos require accepting Meta's SA-1B license before the download (see [data/dci/README.md](../data/dci/README.md)).

## 2. Split sizes

Training sets serve as the in-context example pools and validation sets as the evaluation queries; every validation set is gender-balanced except VisoGender-OP, which is near-balanced. The counts are male / female / all.

| Dataset       | Train             | Valid                 |
| ------------- | ----------------- | --------------------- |
| COCOBias      | 538 / 267 / 805   | 1,065 / 1,065 / 2,130 |
| DCI           | 45 / 30 / 75      | 119 / 119 / 238       |
| VisoGender-OO | 46 / 46 / 92      | 69 / 69 / 138         |
| VisoGender-OP | 92 / 92 / 184     | 126 / 130 / 256       |
| VisualCoT     | 804 / 264 / 1,068 | 1,056 / 1,056 / 2,112 |

The VisoGender counts come from the 670 of 690 image URLs that resolved for the paper; a URL that does not resolve drops its image and changes both splits (see [data/visogender/README.md](../data/visogender/README.md)).
