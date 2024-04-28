import json
import os
import random

def load_tags():
    LOCALPATH = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    with open(os.path.join(LOCALPATH, "tags.json")) as f:
        content = json.load(f)
        return [int(k) for k in content.keys()]

TAG_IDS = load_tags()

def retag(sample: dict) -> dict:
    """Annotates all non-selected tags as O."""
    new_tags = []
    for t in sample["ner_tags"]:
        if t not in TAG_IDS:
            new_tags.append(0)
        else:
            new_tags.append(t)
    sample["new_tags"] = new_tags
    return sample

def filter_empty(sample: dict) -> dict:
    """Returns only samples that contain at least one selected tag. Otherwise, randomly returns 1% of empty samples"""
    SEED = 42
    contains_tag = sum(sample["new_tags"]) > 0
    random.seed(SEED)
    return contains_tag or random.randint(1, 100) == SEED
