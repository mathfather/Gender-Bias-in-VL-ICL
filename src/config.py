import os

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA    = os.path.join(ROOT, "data")
MODEL   = os.path.join(ROOT, "models")
RESULT  = os.path.join(ROOT, "results")
SUMMARY = os.path.join(ROOT, "summary")

RANDOM_SEED = 42
SHOT_SEEDS = [100, 101, 102, 103, 104]
NUM_SHOTS = [0, 2, 4, 6, 8]
SAMPLE_POOL_SIZE = 100
CAPTION_MAX_NEW_TOKEN = 50
QA_MAX_NEW_TOKEN = 256
TQDM_STYLE = "{l_bar}{bar:25}{r_bar}{bar:-10b}"

EXP_FOLDERS = {
    "1": "1-random",
    "2": "2-random_male",
    "3": "3-random_female",
    "4": "4-random_balanced",
    "5": "5-image_image_retrieval",
    "6": "6-image_text_retrieval",
}
RANDOM_EXP_TYPES = ["1", "2", "3", "4"]
RETRIEVAL_EXP_TYPES = ["5", "6"]
GENDERED_EXP_TYPES = ["2", "3"]

DATASETS = ["cocobias", "dci", "visogender", "visualcot"]
CAPTION_DATASETS = ["cocobias", "dci"]
VISOGENDER_TYPES = ["OO", "OP"]
VISUALCOT_DATA_TYPE = "gqa"
SIM_TYPES = ["II", "IT"]
QA_PROMPT_MODES = ["base", "advanced"]
QA_MODELS = ["MiniCPM", "Phi35V", "Qwen3VL"]

# "coco-balanced" reads the same images as "coco" but pairs them with the 267M/267F balanced pool.
FOLDER_MAP = {
    "coco":          "val2014",
    "coco-balanced": "val2014",
    "flux":          "FLUX.1-dev",
    "sd35l":         "stable-diffusion-3.5-large",
    "dci":           "images",
}
CAPTION_IMAGE_TYPES = list(FOLDER_MAP)
SYNTHETIC_IMAGE_TYPES = ["flux", "sd35l"]

MASCULINE = [
    "man", "men", "male", "father", "gentleman", "gentlemen", "boy", "boys", "uncle", "husband",
    "actor", "prince", "waiter", "son", "he", "his", "him", "himself", "brother", "brothers", "guy",
    "guys", "emperor", "emperors", "dude", "dudes", "boyfriend", "chairman", "policeman",
    "policemen", "groom",
]
FEMININE = [
    "woman", "women", "female", "lady", "ladies", "mother", "girl", "girls", "aunt", "wife",
    "actress", "princess", "waitress", "daughter", "she", "her", "hers", "herself", "sister",
    "sisters", "queen", "queens", "pregnant", "girlfriend", "chairwoman", "policewoman",
    "policewomen", "bride",
]
NEUTRAL = ["child", "human", "people", "person"]
MULTI_PERSON_SET = {
    "men", "gentlemen", "boys", "brothers", "guys", "emperors", "dudes", "policemen", "women",
    "ladies", "girls", "sisters", "queens", "policewomen", "people", "couple", "crowd", "group",
    "team", "family", "pair", "duo", "trio",
}
GENDER_WORD_SET = set(MASCULINE + FEMININE)


def select_shot_seeds(exp_type: str | int, num_shot: int) -> list[int]:
    if num_shot == 0 or str(exp_type) in RETRIEVAL_EXP_TYPES:
        return [SHOT_SEEDS[0]]
    return list(SHOT_SEEDS)
