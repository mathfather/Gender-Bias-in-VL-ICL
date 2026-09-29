# Data

[doc/datasets.md](../doc/datasets.md) gives the dataset overview and the split sizes, and each subfolder README the download and build commands.

After downloading and building, the folder looks like this:

```
data/
├── coco2014/                     # COCOBias (image captioning); see coco2014/README.md
│   ├── annotations/              # COCO captions and instances JSONs
│   ├── bias/                     # gender and skin annotations CSV (form-gated)
│   ├── images/                   # val2014/ (COCO val2014 images), plus FLUX.1-dev/ and
│   │                             #   stable-diffusion-3.5-large/ if generated
│   └── captions_bias_*.json      # built splits
├── dci/                          # DCI (image captioning); see dci/README.md
│   ├── annotations/              # slimmed annotations.json (built)
│   ├── images/                   # SA-1B photo subset (sa_*.jpg)
│   └── captions_bias_*.json      # built splits
├── visogender/                   # VisoGender (pronoun prediction); see visogender/README.md
│   ├── annotations/              # raw OO and OP TSVs
│   ├── images/                   # fetched from the TSV URLs by `src.data.visogender --download_images`
│   └── {oo,op}_{train,valid}.*   # built splits
└── visual-cot/                   # VisualCoT (VQA); see visual-cot/README.md
    ├── annotations/              # CoT jsonls and GQA scene graphs
    ├── images/                   # GQA images (*.jpg)
    └── cot_bias_*.json           # built splits
```
