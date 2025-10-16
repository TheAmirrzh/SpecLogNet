# scripts/inspect_labels_and_graphs.py
import argparse, torch
import numpy as np
from torch.utils.data import Dataset
import os, sys, pathlib
REPO_ROOT = pathlib.Path("/Users/amirmac/WorkSpace/Codes/LogNet/spec-logic-preproc").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
def load(path):
    ds = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(ds, Dataset)
    return ds

def stats(ds, name):
    ys = []
    n_nodes = []
    n_edges = []
    for i in range(len(ds)):
        d = ds[i]
        y = getattr(d, "y", None)
        if y is not None:
            try:
                ys.append(int(y.item()) if hasattr(y, "item") else int(y))
            except:
                ys.append(str(y))
        x = getattr(d, "x", None)
        if x is not None:
            n_nodes.append(x.shape[0])
        e = getattr(d, "edge_index", None)
        if e is not None:
            # edge_index shape [2, E]
            try:
                n_edges.append(e.shape[1])
            except:
                n_edges.append(len(e))
    print(f"\n=== {name} ===")
    print("samples:", len(ds))
    print("label distribution (counts):", {k: v for k,v in __import__('collections').Counter(ys).most_common()})
    if n_nodes:
        import numpy as np
        print("nodes: mean", np.mean(n_nodes), "min", min(n_nodes), "max", max(n_nodes))
    if n_edges:
        print("edges: mean", np.mean(n_edges), "min", min(n_edges), "max", max(n_edges))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--test", required=True)
    args = p.parse_args()
    for name, path in [("TRAIN", args.train), ("VAL", args.val), ("TEST", args.test)]:
        ds = load(path)
        stats(ds, name)
