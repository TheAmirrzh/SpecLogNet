import argparse
import torch
from collections import Counter, defaultdict
from torch.utils.data import Dataset
import os, sys, pathlib
REPO_ROOT = pathlib.Path("/Users/amirmac/WorkSpace/Codes/LogNet/spec-logic-preproc").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
def load_dataset(path):
    try:
        ds = torch.load(path, map_location="cpu", weights_only=False)
        assert isinstance(ds, Dataset), f"{path} is not a Dataset, got {type(ds)}"
        print(f"[OK] Loaded {path}: {type(ds)} with {len(ds)} samples")
        return ds
    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {e}")
        return None

def sample_structure(ds, k=3):
        print("\n=== SAMPLE STRUCTURE ===")
        for i in range(min(k, len(ds))):
            x = ds[i]
            print(f"  Sample {i}: type={type(x)}")
            if isinstance(x, dict):
                for key, val in x.items():
                    if hasattr(val, 'shape'):
                        print(f"    {key}: tensor {tuple(val.shape)}")
                    else:
                        print(f"    {key}: {type(val)} -> {val}")
            else:
                print(f"    Value: {x}")

def detect_label_leakage(train_ds, val_ds, test_ds):
    def extract_labels(ds):
        labels = []
        for i in range(len(ds)):
            x = ds[i]
            if isinstance(x, dict):
                label = x.get("label") or x.get("target") or None
                if label is not None:
                    labels.append(int(label))
            else:
                labels.append(None)
        return labels

    print("\n=== LABEL LEAKAGE CHECK ===")
    train_labels = extract_labels(train_ds)
    val_labels   = extract_labels(val_ds)
    test_labels  = extract_labels(test_ds)

    common_train_val = set(train_labels) & set(val_labels)
    common_val_test  = set(val_labels)  & set(test_labels)

    print(f"  Unique Train Labels: {len(set(train_labels))}")
    print(f"  Unique Val Labels:   {len(set(val_labels))}")
    print(f"  Unique Test Labels:  {len(set(test_labels))}")

    print(f"  Common Train ∩ Val:  {len(common_train_val)}")
    print(f"  Common Val ∩ Test:   {len(common_val_test)}")

def analyze_candidate_count(ds):
    """
    Checks: how many candidate answers does each sample compare against?
    Assumes the dataset contains something like x['candidates'] or 'negatives'
    """
    print("\n=== CANDIDATE SPACE ANALYSIS ===")
    key_candidates = ["candidates", "negatives", "all_rules"]  # adjust based on your dataset format

    found_key = None
    for key in key_candidates:
        if key in ds[0]:
            found_key = key
            break

    if not found_key:
        print("  Could not find candidate/negatives key in sample. Please adapt script.")
        return

    counts = []
    for i in range(len(ds)):
        sample = ds[i]
        candidates = sample[found_key]
        if hasattr(candidates, "__len__"):
            counts.append(len(candidates))

    print(f"  Avg candidate count: {sum(counts)/len(counts):.2f}")
    print(f"  Min / Max: {min(counts)} / {max(counts)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val",   required=True)
    parser.add_argument("--test",  required=True)
    args = parser.parse_args()

    train_ds = load_dataset(args.train)
    val_ds   = load_dataset(args.val)
    test_ds  = load_dataset(args.test)

    if not train_ds or not val_ds or not test_ds:
        print("One or more datasets failed to load. Exiting.")
        return

    print("\n============================")
    print("TRAIN SET SAMPLE STRUCTURE")
    sample_structure(train_ds)
    print("\nVAL SET SAMPLE STRUCTURE")
    sample_structure(val_ds)

    detect_label_leakage(train_ds, val_ds, test_ds)

    print("\n=== CANDIDATE SPACE (TRAIN) ===")
    analyze_candidate_count(train_ds)
    print("=== CANDIDATE SPACE (VAL) ===")
    analyze_candidate_count(val_ds)

if __name__ == "__main__":
    main()
