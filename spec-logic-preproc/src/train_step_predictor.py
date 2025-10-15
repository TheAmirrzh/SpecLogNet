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
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau # Import scheduler directly

# Explicitly try to import PyG components
try:
    from torch_geometric.data import Data, DataLoader as PyGDataLoader
except Exception:
    # Fallback to standard DataLoader (will raise error later if PyG Data is used)
    from torch.utils.data import DataLoader as PyGDataLoader 
    raise RuntimeError("torch_geometric is required. Install torch_geometric.") from None

# Use the PyG DataLoader exclusively
DataLoader = PyGDataLoader

from src.models.gcn_step_predictor import GCNStepPredictor

# --- CUSTOM COLLATE FUNCTION TO FILTER BAD SAMPLES ---
# The logic in __getitem__ returns a dummy Data object with y=[-1] for bad samples.
# This function filters them out before collation/batching.
def filter_collate(batch):
    # Filter out samples where data.y is [-1] (invalid samples)
    filtered_batch = [data for data in batch if not (hasattr(data, 'y') and data.y.item() == -1)]
    if not filtered_batch:
        # If the batch is empty after filtering, return an empty list. 
        # This will need to be caught in the training loop.
        return []
    
    # Since we are using PyG DataLoader, we rely on its internal collate mechanism.
    # We return the filtered list, and PyG DataLoader handles the actual stacking.
    return filtered_batch

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
            try:
                inst = json.load(open(f, "r"))
            except Exception as e:
                print(f"Warning: Could not load JSON file {f}: {e}")
                continue
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
            # safety: if the rule node isn't present, return a dummy sample
            data = Data(x=torch.tensor(node_feat), edge_index=edge_index)
            data.y = torch.tensor([-1], dtype=torch.long) # FLAG FOR FILTERING
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
            if not data: # Catch empty batch from filter_collate
                continue
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
    exp_dir: Optional[str] = None, # Added exp_dir argument
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

    # Use the PyG DataLoader and explicitly provide the filter_collate function
    # NOTE: PyGDataLoader handles Data objects, but if bad samples remain, they can break it.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=filter_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=filter_collate)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=filter_collate)

    # determine in_feats from first sample
    if len(train_ds) == 0:
        raise RuntimeError("Training dataset is empty.")
        
    # Attempt to get a good sample for in_feats determination
    sample0 = None
    for item in train_ds:
        if item.y.item() != -1:
            sample0 = item
            break
    
    if sample0 is None:
        raise RuntimeError("Training dataset is empty or contains only invalid samples after filtering.")
    
    in_feats = sample0.x.size(1)

    model = GCNStepPredictor(in_feats=in_feats, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    ce = torch.nn.CrossEntropyLoss()
    
    # FIX: Remove 'verbose=True' to avoid PyTorch TypeError.
    scheduler = ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=5
    )

    best_val_hit1 = -1.0
    # Store training configuration for results file
    config = {
        "json_dir": json_dir,
        "spectral_dir": spectral_dir,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "hidden": hidden,
        "device": device_str,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "seed": seed
    }

    for ep in range(1, epochs+1):
        model.train()
        train_losses = []
        for data in train_loader:
            if not data: # Skip empty batches resulting from filter_collate
                continue

            # data here is a *list* of filtered Data objects if batch_size > 1
            # If batch_size=1, data is technically a list containing one Data object.
            # PyGDataLoader handles combining the list of Data objects into a single batched Data object.
            # We must ensure data is actually the combined Data object before moving to device.
            if isinstance(data, list):
                # If we get a list, it means filter_collate ran, and since batch_size=1, it's a list with one item.
                # Use PyG's explicit collate function here for robustness.
                from torch_geometric.data.dataloader import default_collate as pyg_default_collate
                data = pyg_default_collate(data)
            
            # The type check below should catch unexpected behaviour if collate_fn isn't respected.
            # Since the outer error is happening *before* the inner check, removing the check is safer.
            # if not isinstance(data, Data):
            #     print(f"\nFATAL ERROR: Expected PyG Data object but found {type(data)}. Check PyG installation.")
            #     raise TypeError(f"Collate failure: Expected Data but got {type(data)}")

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
        
        # Scheduler step using validation metric
        if val_metrics['hit1'] is not None:
            scheduler.step(val_metrics['hit1'])
        
        print(f"[Epoch {ep}] train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} val_hit1={val_metrics['hit1']:.4f} val_hit3={val_metrics['hit3']:.4f}")
        
        # save best checkpoint based on validation hit@1
        if val_metrics['hit1'] is not None and val_metrics['hit1'] > best_val_hit1:
            best_val_hit1 = val_metrics['hit1']
            ckpt_dir = os.path.join(exp_dir, "checkpoints") if exp_dir else "checkpoints"
            ckpt_path = os.path.join(ckpt_dir, "gcn_step_best.pt")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "in_feats": in_feats}, ckpt_path)
            
    # final test
    test_metrics = evaluate_model(model, test_loader, device)
    
    # Save final results to the experiment directory
    if exp_dir:
        results = {
            "config": config,
            "test_metrics": test_metrics,
        }
        results_path = os.path.join(exp_dir, "final_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
            
    print("FINAL TEST:", test_metrics)
    return model, test_metrics

# If invoked as script
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-dir", required=True, help="Directory with canonical JSON instances (generated by horn_generator)")
    parser.add_argument("--spectral-dir", default=None, help="Optional directory with spectral .npz files")
    
    # Arguments expected by the calling script
    parser.add_argument("--exp-name", default="default_exp", help="Experiment name (old, for logs)")
    parser.add_argument("--log-dir", default="experiments", help="Log base directory (old, for logs)")
    # Argument expected by new Phase1Runner flow (for saving results)
    parser.add_argument("--exp-dir", default=None, help="Explicit directory path to save model/results.")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128) # Keep old name for compatibility
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    
    # These are not used by train_step_predictor's train function, but are used in the main logic calling it.
    # Since we don't have the wrapper script, we will simulate the main script logic here.
    args = parser.parse_args()
    
    # If exp_dir is provided by the master script, use it for saving. Otherwise, construct one from old args.
    if not args.exp_dir:
        # Construct the intended directory path from old arguments
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir_path = os.path.join(args.log_dir, f"{args.exp_name}_{timestamp}")
    else:
        exp_dir_path = args.exp_dir

    # Call the main training function, mapping old arguments to new names where necessary
    train(
        json_dir=args.json_dir,
        spectral_dir=args.spectral_dir,
        exp_dir=exp_dir_path,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden=args.hidden_dim, # Map hidden-dim to hidden
        device_str=args.device,
        # Default splits used if the calling script doesn't handle them
        val_fraction=0.1,
        test_fraction=0.1,
        seed=args.seed
    )
