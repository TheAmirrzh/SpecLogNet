# src/train_step_predictor.py
"""
Training & dataset for step prediction on canonical JSONs produced by horn_generator or TPTP parser.

This file contains:
 - StepPredictionDataset: PyG-compatible dataset that produces one training sample per proof step.
 - train(...) function: trains the GCNStepPredictor on training set, validates on dev set.
 - evaluate(...) function: compute Hit@K metrics and average loss on a dataset.

Important design choices:
 - Each sample corresponds to a single graph-state (nodes, edges) and a single label:
     label = index (0..N-1) of the rule-node used for the proof step.
 - Node features:
     [type_onehot | derived_flag | optional spectral eigvecs...]
 - For simplicity & correctness: DataLoader uses batch_size=1 (so per-sample logits correspond to nodes in that sample).
"""

import os
import json
import glob
import random
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Data, DataLoader
except Exception:
    from torch.utils.data import DataLoader
    raise RuntimeError("torch_geometric is required. Install torch_geometric.") from None

from src.models.gcn_step_predictor import GCNStepPredictor

# -------------------------
# Dataset: per-proof-step sample
# -------------------------
class StepPredictionDataset(Dataset):
    """
    Loads canonical JSON files (one canonical instance per file) and constructs
    per-proof-step samples. Each sample is a PyG Data object with fields:
      - x: (N, F) node features
      - edge_index: (2, E)
      - y: int (index among nodes identifying the rule node to apply)
      - meta: optional metadata dict (kept in data.meta)
    """

    def __init__(self, json_files: List[str], spectral_dir: Optional[str] = None, seed: Optional[int] = 0):
        self.files = json_files
        self.samples: List[Dict] = []  # will hold tuples (canonical_instance, proof_step_index)
        self.spectral_dir = spectral_dir
        random.seed(seed)

        for f in self.files:
            inst = json.load(open(f, "r"))
            # basic validation
            if "nodes" not in inst or "edges" not in inst:
                continue
            proof_steps = inst.get("proof_steps", [])
            # Only use instances that have at least one proof step
            if not proof_steps:
                continue
            # append (inst, step_idx) for every proof step
            for si in range(len(proof_steps)):
                self.samples.append((inst, si))

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _node_order_map(nodes: List[dict]) -> Dict[int, int]:
        ordered = sorted(nodes, key=lambda n: int(n["nid"]))
        id2idx = {int(n["nid"]): idx for idx, n in enumerate(ordered)}
        return id2idx

    def _compute_initial_facts(self, inst: dict) -> set:
        # initial facts = fact nodes that are not derived by any proof_step
        all_derived = set([int(s["derived_node"]) for s in inst.get("proof_steps", [])])
        initial_facts = set()
        for n in inst["nodes"]:
            if n.get("type") == "fact" and int(n["nid"]) not in all_derived:
                initial_facts.add(int(n["nid"]))
        return initial_facts

    def _load_spectral(self, inst_id: str) -> Optional[np.ndarray]:
        if self.spectral_dir is None:
            return None
        # find any matching npz file that starts with inst_id
        cand = glob.glob(os.path.join(self.spectral_dir, f"{inst_id}_spectral_k*"))
        if not cand:
            return None
        data = np.load(cand[0])
        eigvecs = data["eigvecs"]  # shape (n_nodes, k)
        return eigvecs

    def __getitem__(self, idx):
        inst, step_idx = self.samples[idx]
        nodes = inst["nodes"]
        edges = inst["edges"]
        proof_steps = inst.get("proof_steps", [])
        assert 0 <= step_idx < len(proof_steps)

        # ordering & mapping
        nodes_sorted = sorted(nodes, key=lambda x: int(x["nid"]))
        nid_to_idx = {int(n["nid"]): i for i, n in enumerate(nodes_sorted)}
        n_nodes = len(nodes_sorted)

        # initial facts + derived before this step
        initial_facts = self._compute_initial_facts(inst)
        derived = set(initial_facts)
        for s in proof_steps[:step_idx]:
            derived.add(int(s["derived_node"]))

        # node type one-hot
        types = [n.get("type", "unk") for n in nodes_sorted]
        uniq_types = sorted(list(set(types)))
        type2i = {t:i for i,t in enumerate(uniq_types)}
        type_feat = np.zeros((n_nodes, len(uniq_types)), dtype=np.float32)
        for i,t in enumerate(types):
            type_feat[i, type2i[t]] = 1.0

        # derived flag
        derived_flag = np.zeros((n_nodes, 1), dtype=np.float32)
        for nid in derived:
            if nid in nid_to_idx:
                derived_flag[nid_to_idx[nid], 0] = 1.0

        # spectral (optional)
        spectral = self._load_spectral(inst.get("id", ""))
        if spectral is not None:
            # attempt to align size
            if spectral.shape[0] != n_nodes:
                # try to match by node id ordering: spectral assumed to follow node order by nid.
                # fallback to truncation/padding
                k = spectral.shape[1]
                if spectral.shape[0] < n_nodes:
                    pad = np.zeros((n_nodes - spectral.shape[0], k), dtype=np.float32)
                    spectral = np.vstack([spectral, pad])
                else:
                    spectral = spectral[:n_nodes, :]
            node_feat = np.concatenate([type_feat, derived_flag, spectral.astype(np.float32)], axis=1)
        else:
            node_feat = np.concatenate([type_feat, derived_flag], axis=1)

        # build edge_index (map global nids -> 0..n-1 indices)
        src = []
        dst = []
        for e in edges:
            s = int(e["src"]); d = int(e["dst"])
            if s in nid_to_idx and d in nid_to_idx:
                src.append(nid_to_idx[s]); dst.append(nid_to_idx[d])
        if len(src) == 0:
            edge_index = torch.empty((2,0), dtype=torch.long)
        else:
            edge_index = torch.tensor([src, dst], dtype=torch.long)

        # determine target: the used_rule nid for this proof step
        target_rule_nid = int(proof_steps[step_idx]["used_rule"])
        if target_rule_nid not in nid_to_idx:
            # safety: if the rule node isn't present, skip this sample (rare)
            # return a dummy sample: all zeros (should be filtered upstream)
            data = Data(x=torch.tensor(node_feat), edge_index=edge_index)
            data.y = torch.tensor([-1], dtype=torch.long)
            data.meta = {"id": inst.get("id", ""), "step_idx": step_idx}
            return data
        target_idx = nid_to_idx[target_rule_nid]

        data = Data(x=torch.tensor(node_feat, dtype=torch.float), edge_index=edge_index)
        data.y = torch.tensor([target_idx], dtype=torch.long)  # store as (1,)
        data.meta = {"id": inst.get("id", ""), "step_idx": step_idx, "n_nodes": n_nodes}
        return data

# -------------------------
# Helper metrics
# -------------------------
def hit_at_k(scores: torch.Tensor, target_idx: int, k: int) -> int:
    """
    scores: (N,) tensor of node scores (higher = better)
    target_idx: int index in 0..N-1
    returns 1 if target in top-k else 0
    """
    N = scores.size(0)
    k = min(k, N)
    topk = torch.topk(scores, k=k, largest=True).indices.tolist()
    return 1 if target_idx in topk else 0

# -------------------------
# Training & evaluation
# -------------------------
def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, device: torch.device = torch.device("cpu")) -> Dict:
    model.eval()
    total = 0
    loss_sum = 0.0
    hits = {1:0, 3:0, 10:0}
    ce = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            # ensure node features exist
            x = data.x
            edge_index = data.edge_index
            scores, _ = model(x, edge_index)
            # scores shape (N,)
            target_tensor = data.y.view(-1)  # (1,)
            target_idx = int(target_tensor.item())
            if target_idx < 0:
                continue
            # compute loss: reshape logits to (1, N) and target (1,)
            logits = scores.unsqueeze(0)
            targ = torch.tensor([target_idx], dtype=torch.long, device=device)
            loss = ce(logits, targ)
            loss_sum += float(loss.item())
            total += 1
            # hits
            for k in hits.keys():
                hits[k] += hit_at_k(scores, target_idx, k)
    if total == 0:
        return {"loss": None, "hit1": None, "hit3": None, "hit10": None, "n": 0}
    return {"loss": loss_sum / total, "hit1": hits[1]/total, "hit3": hits[3]/total, "hit10": hits[10]/total, "n": total}

def train(
    json_dir: str,
    spectral_dir: Optional[str] = None,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 1,
    hidden: int = 128,
    device_str: str = "cpu",
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42
):
    device = torch.device(device_str)
    # collect json files
    files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    random.Random(seed).shuffle(files)
    n = len(files)
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))
    train_files = files[: max(0, n - n_val - n_test)]
    val_files = files[n - n_val - n_test: n - n_test]
    test_files = files[n - n_test:]

    train_ds = StepPredictionDataset(train_files, spectral_dir=spectral_dir, seed=seed)
    val_ds = StepPredictionDataset(val_files, spectral_dir=spectral_dir, seed=seed+1)
    test_ds = StepPredictionDataset(test_files, spectral_dir=spectral_dir, seed=seed+2)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # determine in_feats from first sample
    sample0 = train_ds[0]
    in_feats = sample0.x.size(1)

    model = GCNStepPredictor(in_feats=in_feats, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    ce = torch.nn.CrossEntropyLoss()

    best_val_hit1 = -1.0
    for ep in range(1, epochs+1):
        model.train()
        train_losses = []
        for data in train_loader:
            data = data.to(device)
            x = data.x
            edge_index = data.edge_index
            target_idx = int(data.y.view(-1).item())
            if target_idx < 0:
                continue
            opt.zero_grad()
            scores, _ = model(x, edge_index)
            logits = scores.unsqueeze(0)  # shape (1, N)
            targ = torch.tensor([target_idx], dtype=torch.long, device=device)
            loss = ce(logits, targ)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))
        # epoch end: evaluate
        train_loss = float(np.mean(train_losses)) if len(train_losses)>0 else None
        val_metrics = evaluate_model(model, val_loader, device)
        print(f"[Epoch {ep}] train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} val_hit1={val_metrics['hit1']:.4f} val_hit3={val_metrics['hit3']:.4f}")
        # save best
        if val_metrics['hit1'] is not None and val_metrics['hit1'] > best_val_hit1:
            best_val_hit1 = val_metrics['hit1']
            ckpt_path = os.path.join("checkpoints", "gcn_step_best.pt")
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({"model_state": model.state_dict(), "in_feats": in_feats}, ckpt_path)
    # final test
    test_metrics = evaluate_model(model, test_loader, device)
    print("FINAL TEST:", test_metrics)
    return model, test_metrics

# If invoked as script
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-dir", required=True, help="Directory with canonical JSON instances (generated by horn_generator)")
    parser.add_argument("--spectral-dir", default=None, help="Optional directory with spectral .npz files")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train(args.json_dir, spectral_dir=args.spectral_dir, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, hidden=args.hidden, device_str=args.device)