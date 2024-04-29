import json
import os
import random
from pprint import pprint

from datasets import ClassLabel, DatasetDict, Sequence


def load_tags():
    LOCALPATH = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    with open(os.path.join(LOCALPATH, "tags.json")) as f:
        content: dict = json.load(f)
        pprint(content)
        ids, names = list(zip(*[(int(k), v) for k, v in content.items()]))
        return list(ids), list(names)


TAG_IDS, TAG_NAMES = load_tags()
print(TAG_IDS, TAG_NAMES)


def retag(sample: dict) -> dict:
    """Annotates all non-selected tags as O."""
    new_tags = []
    for t in sample["ner_tags"]:
        if t not in TAG_IDS:
            new_tags.append(0)
        else:
            new_tags.append(TAG_IDS.index(t) + 1)
    sample["ner_tags"] = new_tags
    return sample


def filter_empty(sample: dict) -> dict:
    """Returns only samples that contain at least one selected tag. Otherwise, randomly returns 0.1% of empty samples"""
    SEED = 42
    contains_tag = any([t > 0 for t in sample["ner_tags"]])
    random.seed(SEED)
    return contains_tag or random.randint(1, 1000) == SEED


def process(dataset: DatasetDict):
    mapped = dataset.map(retag, num_proc=os.cpu_count(), load_from_cache_file=False)
    filtered = mapped.filter(filter_empty, num_proc=os.cpu_count(), load_from_cache_file=False)
    return filtered.cast_column("ner_tags", Sequence(ClassLabel(names=["O"] + TAG_NAMES)))
