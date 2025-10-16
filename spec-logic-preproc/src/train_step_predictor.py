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


def hit_at_k(scores: torch.Tensor, target_idx: int, k: int) -> float:
    """Hit@K metric: 1 if target in top-k, else 0. Clamps k for small graphs."""
    effective_k = min(k, len(scores))
    if effective_k == 0:
        return 0.0
    top_k = torch.topk(scores, effective_k).indices
    return 1.0 if target_idx in top_k else 0.0


def evaluate_model(model, dataloader, device):
    """Evaluate model on dataset, returning avg loss and Hit@K."""
    model.eval()
    losses = []
    hit1, hit3, hit10 = [], [], []
    ce = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            scores, _ = model(data.x, data.edge_index)
            target_idx = int(data.y.item())
            
            if target_idx < 0:
                continue
            
            logits = scores.unsqueeze(0)
            targ = torch.tensor([target_idx], dtype=torch.long, device=device)
            loss = ce(logits, targ)
            losses.append(float(loss.item()))
            
            hit1.append(hit_at_k(scores, target_idx, 1))
            hit3.append(hit_at_k(scores, target_idx, 3))
            hit10.append(hit_at_k(scores, target_idx, 10))
    
    return {
        "loss": np.mean(losses) if losses else None,
        "hit1": np.mean(hit1) if hit1 else None,
        "hit3": np.mean(hit3) if hit3 else None,
        "hit10": np.mean(hit10) if hit10 else None
    }


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
        self.samples: List[Tuple[Dict, int]] = []  # will hold tuples (canonical_instance, proof_step_index)
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
            print(f"WARNING: spectral_dir is None, spectral features will not be used")
            return None
        if not inst_id:
            # If instance ID is missing from the JSON, we can't find the spectral file.
            print(f"WARNING: Instance ID is missing, cannot load spectral features")
            return None

        # Find any spectral file for this instance ID, regardless of k.
        # This is more robust than hardcoding k.
        pattern = os.path.join(self.spectral_dir, f"{inst_id}_spectral_k*.npz")
        candidates = glob.glob(pattern)

        if not candidates:
            # This is the critical point: if no file is found, we now print a warning
            print(f"WARNING: Spectral file not found for pattern: {pattern}")
            print(f"Spectral features will not be added to the model input")
            return None
        
        spectral_data = np.load(candidates[0])["eigvecs"]
        return spectral_data

    def __getitem__(self, idx):
        inst, step_idx = self.samples[idx]
        nodes = inst["nodes"]
        edges = inst["edges"]
        proof_steps = inst.get("proof_steps", [])
        
        # Map nids to consecutive indices 0..N-1
        id2idx = self._node_order_map(nodes)
        
        # Node features: one-hot type (fact=0, rule=1) + derived flag (0/1)
        n_nodes = len(nodes)
        x = torch.zeros((n_nodes, 2), dtype=torch.float)  # [type_onehot(1), derived(1)]
        
        # Derived facts up to this step
        derived_up_to_step = set(int(s["derived_node"]) for s in proof_steps[:step_idx+1])
        initial_facts = self._compute_initial_facts(inst)
        
        for node in nodes:
            idx = id2idx[int(node["nid"])]
            ntype = 0 if node["type"] == "fact" else 1
            derived = 1 if int(node["nid"]) in derived_up_to_step else 0
            x[idx] = torch.tensor([ntype, derived])
        
        # Optional: append spectral features
        spectral_feats = self._load_spectral(inst.get("id", ""))
        if spectral_feats is not None:
            spectral_feats = torch.from_numpy(spectral_feats).float()  # (N, k)
            x = torch.cat([x, spectral_feats], dim=-1)
        
        # Edges: src/dst using idx
        edge_index = []
        for e in edges:
            src_idx = id2idx[int(e["src"])]
            dst_idx = id2idx[int(e["dst"])]
            edge_index.append([src_idx, dst_idx])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Target: index of the rule node used in this step
        step = proof_steps[step_idx]
        rule_nid = int(step["used_rule"])
        y = torch.tensor([id2idx[rule_nid]], dtype=torch.long)
        
        # PyG Data
        data = Data(x=x, edge_index=edge_index, y=y)
        data.meta = {"instance_id": inst.get("id", ""), "step_idx": step_idx}
        
        return data


def train(
    json_dir: str,
    spectral_dir: Optional[str] = None,
    exp_dir: Optional[str] = None,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 1,
    hidden: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
    device_str: str = "cpu",
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42
):
    """Train GCNStepPredictor on dataset."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Load and split files
    all_files = glob.glob(os.path.join(json_dir, "**/*.json"), recursive=True)
    random.shuffle(all_files)
    
    n_total = len(all_files)
    n_val = int(n_total * val_fraction)
    n_test = int(n_total * test_fraction)
    n_train = n_total - n_val - n_test
    
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    test_files = all_files[n_train + n_val:]
    
    train_dataset = StepPredictionDataset(train_files, spectral_dir, seed)
    val_dataset = StepPredictionDataset(val_files, spectral_dir, seed)
    test_dataset = StepPredictionDataset(test_files, spectral_dir, seed)
    
    if len(train_dataset) == 0:
        raise ValueError("No valid training samples found.")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Model
    in_feats = train_dataset[0].x.shape[1]
    model = GCNStepPredictor(in_feats, hidden, num_layers, dropout).to(device)
    
    # Optimizer and scheduler
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5, verbose=True)
    
    ce = torch.nn.CrossEntropyLoss()
    
    # Training
    best_val_hit1 = -float('inf')
    if exp_dir:
        os.makedirs(exp_dir, exist_ok=True)
    
    for ep in range(1, epochs+1):
        model.train()
        train_losses = []
        for data in train_loader:
            # Explicitly check if the correct DataLoader type is being used.
            if not isinstance(data, Data):
                print(f"\nFATAL ERROR: Expected PyG Data object but found {type(data)}. Check PyG installation.")
                raise TypeError(f"Collate failure: Expected Data but got {type(data)}")

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
            "config": {"epochs": epochs, "lr": lr, "hidden": hidden, "num_layers": num_layers},
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
        num_layers=args.num_layers,
        dropout=args.dropout,
        device_str=args.device,
        # Default splits used if the calling script doesn't handle them
        val_fraction=0.1,
        test_fraction=0.1,
        seed=args.seed
    )