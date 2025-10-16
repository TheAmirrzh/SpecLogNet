# scripts/check_instance_overlap.py
import argparse, torch, hashlib
from collections import Counter, defaultdict
from pathlib import Path
from torch.utils.data import Dataset
import os, sys, pathlib
REPO_ROOT = pathlib.Path("/Users/amirmac/WorkSpace/Codes/LogNet/spec-logic-preproc").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
def load(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(obj, Dataset)
    return obj

def hash_sample(data_obj):
    # deterministic hash for a Data object (node features + edges + meta instance + step idx)
    import numpy as np
    # Node features
    x = getattr(data_obj, "x", None)
    edge_index = getattr(data_obj, "edge_index", None)
    y = getattr(data_obj, "y", None)
    meta = getattr(data_obj, "meta", {})
    parts = []
    if x is not None:
        parts.append(bytes(np.ascontiguousarray(x).data))
    if edge_index is not None:
        parts.append(bytes(np.ascontiguousarray(edge_index).data))
    if y is not None:
        parts.append(bytes(np.ascontiguousarray(y).data))
    # include instance id & step idx textual info
    iid = meta.get("instance_id", "")
    sidx = str(meta.get("step_idx", ""))
    parts.append(iid.encode("utf-8"))
    parts.append(sidx.encode("utf-8"))
    h = hashlib.sha1(b"|".join(parts)).hexdigest()
    return h, iid, sidx

def gather(path):
    ds = load(path)
    print(f"Loaded {path}: {len(ds)} samples, class={type(ds)}")
    ids = []
    hashes = []
    for i in range(len(ds)):
        h, iid, sidx = hash_sample(ds[i])
        ids.append(iid)
        hashes.append(h)
    return ids, hashes, len(ds)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--test", required=True)
    args = p.parse_args()

    train_ids, train_hashes, ntrain = gather(args.train)
    val_ids, val_hashes, nval = gather(args.val)
    test_ids, test_hashes, ntest = gather(args.test)

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    print("\nUnique instance_id counts: train,val,test =", len(train_set), len(val_set), len(test_set))
    print("Overlaps (instance_id): train∩val =", len(train_set & val_set),
          "train∩test =", len(train_set & test_set), "val∩test =", len(val_set & test_set))

    # exact sample duplicates via hash
    h_train = set(train_hashes)
    h_val = set(val_hashes)
    h_test = set(test_hashes)
    print("Exact sample duplicates: train∩val =", len(h_train & h_val),
          "train∩test =", len(h_train & h_test), "val∩test =", len(h_val & h_test))

if __name__ == "__main__":
    main()
