import gzip
import os
from tqdm import tqdm
import prior
import urllib.request

try:
    from prior import LazyJsonDataset
except:
    raise ImportError("Please update the prior package (pip install --upgrade prior).")


def load_dataset() -> prior.DatasetDict:
    """Load the houses dataset."""
    data = {}
    for split, size in zip(("train", "val"), (94310, 2711)):
        if split == "train":
            d = "./SimpleExploreHouse_train.jsonl.gz"
        else:
            d = "./SimpleExploreHouse_val.jsonl.gz"
        with gzip.open(d, "r") as f:
            houses = [line for line in tqdm(f, total=size, desc=f"Loading {split}")]
        data[split] = LazyJsonDataset(data=houses, dataset="explore_house", split=split)
    return prior.DatasetDict(**data)
