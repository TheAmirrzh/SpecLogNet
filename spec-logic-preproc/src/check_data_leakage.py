import json
from pathlib import Path
from collections import Counter

def load_ids(json_dir):
    ids = []
    for p in Path(json_dir).glob("*.json"):
        with open(p) as f:
            data = json.load(f)
            ids.append(data["source_file"])  # <-- adjust if needed: where do you store original filename?
    return ids

splits = {
    "train": "data_processed/phase1_standard_.../train",
    "val":   "data_processed/phase1_standard_.../val",
    "test":  "data_processed/phase1_standard_.../test",
}
print(len(splits["test"]))
all_ids = {}
for name, path in splits.items():
    all_ids[name] = set(load_ids(path))

# Check overlas
for a in splits:
    for b in splits:
        if a < b:
            overlap = all_ids[a] & all_ids[b]
            print(f"Overlap between {a} and {b}: {len(overlap)} files")
            if overlap:
                print("Examples:", list(overlap)[:5])
