"""Standalone preparation scripts, run from the repo root.

Download a Hugging Face snapshot into models/ (Qwen-VL and MiniCPM-o get their MODEL_MAP folder):
    python -m utils.download_huggingface_model --repo_id <org/name>
Merge the raw DCI annotations, without masks, into data/dci/annotations/annotations.json:
    python -m utils.slim_dci_annotations
Downsample the COCOBias train pool to 50/50 genders (image_type "coco-balanced"):
    python -m utils.balance_train_pool
Restrict coco-{II,IT}-cos.pkl to that pool as coco-balanced-{II,IT}-cos.pkl:
    python -m utils.filter_similarity_balanced
Generate synthetic COCOBias train images from gendered captions (image_type "flux", "sd35l"):
    python -m utils.generate_diffusion_images --model_name {FLUX.1-dev,stable-diffusion-3.5-large}
"""
