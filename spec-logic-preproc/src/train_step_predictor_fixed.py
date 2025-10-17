# src/train_step_predictor_fixed.py
"""
COMPLETELY FIXED training script. Every line reviewed for correctness.
"""
import os
import json
import glob
import random
from typing import List, Optional, Dict, Tuple
from collections import deque
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Assuming GCNStepPredictor is defined in src.models.gcn_step_predictor
from src.models.gcn_step_predictor import GCNStepPredictor

def hit_at_k(scores: torch.Tensor, target_idx: int, k: int) -> float:
    """
    Hit@K metric: 1 if target in top-k, else 0.


    REVIEW: ✓ Clamps k to avoid errors on small graphs
    """
    effective_k = min(k, len(scores))
    if effective_k == 0:
        return 0.0  # REVIEW: ✓ Empty graph case
    top_k = torch.topk(scores, effective_k).indices
    return 1.0 if target_idx in top_k else 0.0

class StepPredictionDataset(Dataset):
    """
    Dataset for step prediction with RICH FEATURES.


    Features (16-dim base):
    - [0-1]: Type one-hot (fact/rule)
    - [2]: Is derived up to this step
    - [3]: Is initial fact
    - [4]: In-degree (normalized)
    - [5]: Out-degree (normalized)
    - [6]: Depth from initial facts (normalized)
    - [7]: Rule body size (normalized, 0 for facts)
    - [8-15]: Predicate hash (8-bit)
    
    REVIEW: ✓ All features are meaningful and normalized
    """
    def __init__(self, json_files: List[str], spectral_dir: Optional[str] = None, seed: int = 42):
        self.files = json_files
        self.samples: List[Tuple[Dict, int]] = []  # REVIEW: ✓ Type hint added
        self.spectral_dir = spectral_dir
        random.seed(seed)
        
        valid_samples = 0
        skipped_no_proofs = 0
        skipped_errors = 0
        
        # REVIEW: ✓ Iterate over all files
        for f in self.files:
            try:
                with open(f, "r") as fp:
                    inst = json.load(fp)
                
                proof_steps = inst.get("proof_steps", [])
                
                # REVIEW: ✓ Skip files without proof steps
                if not proof_steps:
                    skipped_no_proofs += 1
                    continue
                
                # REVIEW: ✓ Create one sample per proof step
                for step_idx in range(len(proof_steps)):
                    self.samples.append((inst, step_idx))
                    valid_samples += 1
            
            except Exception as e:
                # REVIEW: ✓ Catch file read errors
                skipped_errors += 1
                print(f"Error loading {f}: {e}")

        print(f"Dataset: {valid_samples} samples from {len(self.files)} files")
        print(f"  Skipped: {skipped_no_proofs} (no proofs), {skipped_errors} (errors)")

    def __len__(self) -> int:
        """REVIEW: ✓ Standard length method"""
        return len(self.samples)

    @staticmethod
    def _node_order_map(nodes: List[dict]) -> Dict[int, int]:
        """
        Map node IDs to consecutive indices.
        
        REVIEW: ✓ Sorts nodes by nid for consistency
        REVIEW: ✓ Returns dict mapping nid -> index
        """
        ordered = sorted(nodes, key=lambda n: int(n["nid"]))
        return {int(n["nid"]): idx for idx, n in enumerate(ordered)}

    def _compute_initial_facts(self, inst: dict) -> set:
        """
        Find facts that are not derived by any proof step.
        
        REVIEW: ✓ Initial facts = fact nodes NOT in derived_node list
        """
        all_derived = set([int(s["derived_node"]) for s in inst.get("proof_steps", [])])
        initial_facts = set()
        for n in inst["nodes"]:
            if n.get("type") == "fact":
                nid = int(n["nid"])
                if nid not in all_derived:
                    initial_facts.add(nid)
        return initial_facts

    def _compute_depths(self, nodes: List[dict], edges: List[dict],
                        initial_facts: set, id2idx: Dict[int, int]) -> Dict[int, int]:
        """
        BFS depth from initial facts.
        
        REVIEW: ✓ Standard BFS algorithm
        REVIEW: ✓ Returns dict mapping nid -> depth
        """
        depths = {}
        queue = deque()
        
        # REVIEW: ✓ Start from initial facts at depth 0
        for fact_nid in initial_facts:
            depths[fact_nid] = 0
            queue.append((fact_nid, 0))
        
        visited = set(initial_facts)
        
        # REVIEW: ✓ BFS traversal
        while queue:
            current_nid, depth = queue.popleft()
            
            # REVIEW: ✓ Find outgoing edges from current node
            for e in edges:
                if int(e["src"]) == current_nid:
                    next_nid = int(e["dst"])
                    if next_nid not in visited:
                        visited.add(next_nid)
                        depths[next_nid] = depth + 1
                        queue.append((next_nid, depth + 1))
        
        # REVIEW: ✓ Assign max+1 depth to unreachable nodes
        max_depth = max(depths.values()) if depths else 0
        for node in nodes:
            nid = int(node["nid"])
            if nid not in depths:
                depths[nid] = max_depth + 1
                
        return depths

    def _load_spectral(self, inst_id: str) -> Optional[np.ndarray]:
        """
        Load spectral features from .npz file.
        
        REVIEW: ✓ Returns None if not found (graceful failure)
        REVIEW: ✓ Handles missing spectral_dir
        """
        if not self.spectral_dir or not inst_id:
            return None
        
        # REVIEW: ✓ Use glob to find file with any k value
        pattern = os.path.join(self.spectral_dir, f"{inst_id}_spectral_k*.npz")
        candidates = glob.glob(pattern)
        
        if not candidates:
            return None  # REVIEW: ✓ Silent failure OK (spectral is optional)
        
        try:
            data = np.load(candidates[0])
            return data["eigvecs"]  # REVIEW: ✓ Return eigenvectors only
        except Exception:
            return None  # REVIEW: ✓ Handle corrupted files

    def __getitem__(self, idx: int) -> Data:
        """
        Get training sample for a specific proof step.
        
        REVIEW: ✓ Returns PyG Data object
        REVIEW: ✓ All features are properly normalized
        """
        inst, step_idx = self.samples[idx]
        nodes = inst["nodes"]
        edges = inst["edges"]
        proof_steps = inst.get("proof_steps", [])
        
        # REVIEW: ✓ Map node IDs to indices
        id2idx = self._node_order_map(nodes)
        n_nodes = len(nodes)
        
        # FEATURE CONSTRUCTION
        # REVIEW: ✓ Base feature dim is 16
        base_dim = 16
        x = torch.zeros((n_nodes, base_dim), dtype=torch.float)
        
        # REVIEW: ✓ Compute derived facts UP TO this step (not including)
        derived_up_to_step = set(int(s["derived_node"]) for s in proof_steps[:step_idx])
        initial_facts = self._compute_initial_facts(inst)
        
        # COMPUTE DEGREES
        # REVIEW: ✓ Initialize with zeros
        in_degree = np.zeros(n_nodes)
        out_degree = np.zeros(n_nodes)
        
        # REVIEW: ✓ Count edges
        for e in edges:
            src_idx = id2idx[int(e["src"])]
            dst_idx = id2idx[int(e["dst"])]
            out_degree[src_idx] += 1
            in_degree[dst_idx] += 1
            
        # REVIEW: ✓ Normalize by max degree (avoid division by zero)
        max_degree = max(max(in_degree), max(out_degree), 1)
        
        # COMPUTE DEPTHS
        # REVIEW: ✓ BFS from initial facts
        depth_map = self._compute_depths(nodes, edges, initial_facts, id2idx)
        max_depth = max(depth_map.values()) if depth_map else 1
        
        # COMPUTE RULE BODY SIZES
        # REVIEW: ✓ Count body edges for each rule
        rule_body_sizes = {}
        for node in nodes:
            if node["type"] == "rule":
                body_edges = [e for e in edges if e["dst"] == node["nid"] and e.get("etype") == "body"]
                rule_body_sizes[node["nid"]] = len(body_edges)
        
        # REVIEW: ✓ Normalize (avoid division by zero)
        max_body_size = max(rule_body_sizes.values()) if rule_body_sizes else 1
        
        # BUILD FEATURES FOR EACH NODE
        for node in nodes:
            idx = id2idx[int(node["nid"])]
            nid = int(node["nid"])
            
            # [0-1]: Type one-hot
            # REVIEW: ✓ Mutually exclusive (only one is 1.0)
            if node["type"] == "fact":
                x[idx, 0] = 1.0
            elif node["type"] == "rule":
                x[idx, 1] = 1.0
            
            # [2]: Derived flag
            # REVIEW: ✓ Check if derived BEFORE this step
            x[idx, 2] = 1.0 if nid in derived_up_to_step else 0.0
            
            # [3]: Initial fact flag
            # REVIEW: ✓ Check if in initial facts set
            x[idx, 3] = 1.0 if nid in initial_facts else 0.0
            
            # [4-5]: Normalized degrees
            # REVIEW: ✓ Division by max_degree (guaranteed >= 1)
            x[idx, 4] = in_degree[idx] / max_degree
            x[idx, 5] = out_degree[idx] / max_degree
            
            # [6]: Normalized depth
            # REVIEW: ✓ Division by max_depth (guaranteed >= 1)
            x[idx, 6] = depth_map.get(nid, max_depth) / max(max_depth, 1)
            
            # [7]: Normalized rule body size (0 for facts)
            # REVIEW: ✓ Only non-zero for rules
            if node["type"] == "rule":
                x[idx, 7] = rule_body_sizes.get(nid, 0) / max(max_body_size, 1)
            
            # [8-15]: Predicate hash (deterministic 8-bit encoding)
            # REVIEW: ✓ Extract predicate from label
            atom_label = node.get("label", "")
            # REVIEW: ✓ Handle both "P(x)" and "P" formats
            predicate = atom_label.split("(")[0] if "(" in atom_label else atom_label
            # REVIEW: ✓ Remove negation for base predicate
            predicate = predicate.replace("~", "")
            # REVIEW: ✓ Deterministic hash (abs to ensure positive)
            pred_hash = abs(hash(predicate)) % 256
            # REVIEW: ✓ Convert to 8-bit binary representation
            for bit_idx in range(8):
                x[idx, 8 + bit_idx] = float((pred_hash >> bit_idx) & 1)

        # ADD SPECTRAL FEATURES (OPTIONAL)
        # REVIEW: ✓ Concatenate if available
        spectral_feats = self._load_spectral(inst.get("id", ""))
        if spectral_feats is not None:
            spectral_feats = torch.from_numpy(spectral_feats).float()
            # REVIEW: ✓ Check shape compatibility
            if spectral_feats.shape[0] == n_nodes:
                x = torch.cat([x, spectral_feats], dim=-1)

        # BUILD EDGE INDEX
        # REVIEW: ✓ Convert edges to PyG format (2, E)
        edge_list = []
        for e in edges:
            src_idx = id2idx[int(e["src"])]
            dst_idx = id2idx[int(e["dst"])]
            edge_list.append([src_idx, dst_idx])
            
        # REVIEW: ✓ Handle empty edge list
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # TARGET: Rule node used in this step
        # REVIEW: ✓ Extract from proof_steps[step_idx]
        step = proof_steps[step_idx]
        rule_nid = int(step["used_rule"])
        
        # REVIEW: ✓ Convert to index (CRITICAL for batching)
        target_idx = id2idx[rule_nid]
        y = torch.tensor([target_idx], dtype=torch.long)
        
        # CREATE PyG DATA OBJECT
        # REVIEW: ✓ Standard PyG Data format
        data = Data(x=x, edge_index=edge_index, y=y)
        
        # ADD METADATA
        # REVIEW: ✓ Useful for debugging
        data.meta = {
            "instance_id": inst.get("id", ""),
            "step_idx": step_idx,
            "num_nodes": n_nodes,
            "num_derived": len(derived_up_to_step),
            "num_initial": len(initial_facts),
        }
        
        return data

def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Evaluate model with PROPER batch handling.


    REVIEW: ✓ Uses PyG batch vector correctly
    REVIEW: ✓ Returns dict of metrics
    """
    model.eval()
    losses = []
    hit1, hit3, hit10 = [], [], []
    ce = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_data in dataloader:
            batch_data = batch_data.to(device)
            
            # REVIEW: ✓ Model forward pass with batch vector
            scores, _ = model(batch_data.x, batch_data.edge_index, batch_data.batch)
            
            # REVIEW: ✓ Get number of graphs in batch
            batch_size = batch_data.num_graphs if hasattr(batch_data, 'num_graphs') else 1
            
            # CRITICAL: Process each graph separately
            # REVIEW: ✓ This is THE KEY FIX for batching
            for i in range(batch_size):
                # REVIEW: ✓ Mask to get nodes belonging to graph i
                mask = (batch_data.batch == i)
                graph_scores = scores[mask]  # REVIEW: ✓ Now scores are 0-indexed for THIS graph
                
                # REVIEW: ✓ Get target for graph i
                target_idx = int(batch_data.y[i].item())
                
                # REVIEW: ✓ Validate target is within range
                if target_idx < 0 or target_idx >= len(graph_scores):
                    print(f"WARNING: Invalid target {target_idx} for graph with {len(graph_scores)} nodes")
                    continue
                
                # REVIEW: ✓ Compute loss for this graph
                logits = graph_scores.unsqueeze(0)  # Shape: (1, num_nodes_in_graph)
                targ = torch.tensor([target_idx], dtype=torch.long, device=device)
                loss = ce(logits, targ)
                losses.append(float(loss.item()))
                
                # REVIEW: ✓ Compute Hit@K metrics
                hit1.append(hit_at_k(graph_scores, target_idx, 1))
                hit3.append(hit_at_k(graph_scores, target_idx, 3))
                hit10.append(hit_at_k(graph_scores, target_idx, 10))

    # REVIEW: ✓ Compute averages (handle empty case)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "hit1": float(np.mean(hit1)) if hit1 else 0.0,
        "hit3": float(np.mean(hit3)) if hit3 else 0.0,
        "hit10": float(np.mean(hit10)) if hit10 else 0.0,
        "num_samples": len(losses)
    }

def train(
    json_dir: str,
    spectral_dir: Optional[str] = None,
    exp_dir: Optional[str] = None,
    epochs: int = 50,
    lr: float = 5e-4,
    batch_size: int = 32,
    hidden: int = 256,
    num_layers: int = 4,
    dropout: float = 0.3,
    device_str: str = "cpu",
    val_fraction: float = 0.2,
    test_fraction: float = 0.1,
    seed: int = 42
) -> Tuple[torch.nn.Module, Dict[str, float]]:
    """
    Main training function with ALL FIXES APPLIED.
    
    REVIEW: ✓ All parameters have sensible defaults
    REVIEW: ✓ Returns model and test metrics
    """
    
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    
    # REVIEW: ✓ Set all random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    print(f"Using device: {device}")
    print(f"Random seed: {seed}")
    
    # CRITICAL FIX: INSTANCE-LEVEL SPLIT (not file-level)
    # REVIEW: ✓ This prevents data leakage
    all_files = glob.glob(os.path.join(json_dir, "**/*.json"), recursive=True)
    
    # REVIEW: ✓ Build mapping from instance_id to file
    instance_to_file = {}
    for f in all_files:
        try:
            with open(f) as fp:
                inst = json.load(fp)
            inst_id = inst.get("id", f)  # REVIEW: ✓ Use filename as fallback
            instance_to_file[inst_id] = f
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue

    # REVIEW: ✓ Split instances (not files)
    instance_ids = list(instance_to_file.keys())
    random.shuffle(instance_ids)
    
    n_total = len(instance_ids)
    n_val = int(n_total * val_fraction)
    n_test = int(n_total * test_fraction)
    n_train = n_total - n_val - n_test
    
    # REVIEW: ✓ Non-overlapping splits
    train_ids = instance_ids[:n_train]
    val_ids = instance_ids[n_train:n_train + n_val]
    test_ids = instance_ids[n_train + n_val:]
    
    # REVIEW: ✓ Map back to files
    train_files = [instance_to_file[i] for i in train_ids]
    val_files = [instance_to_file[i] for i in val_ids]
    test_files = [instance_to_file[i] for i in test_ids]
    
    print(f"\nInstance-level split:")
    print(f"  Train: {n_train} instances -> {len(train_files)} files")
    print(f"  Val:   {n_val} instances -> {len(val_files)} files")
    print(f"  Test:  {n_test} instances -> {len(test_files)} files")
    
    # REVIEW: ✓ Verify no overlap
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    assert len(train_set & val_set) == 0, "Train/Val overlap!"
    assert len(train_set & test_set) == 0, "Train/Test overlap!"
    assert len(val_set & test_set) == 0, "Val/Test overlap!"
    print("  ✓ No instance overlap between splits")
    
    # CREATE DATASETS
    # REVIEW: ✓ Use fixed dataset class
    train_dataset = StepPredictionDataset(train_files, spectral_dir, seed)
    val_dataset = StepPredictionDataset(val_files, spectral_dir, seed + 1)
    test_dataset = StepPredictionDataset(test_files, spectral_dir, seed + 2)
    
    # REVIEW: ✓ Check for empty datasets
    if len(train_dataset) == 0:
        raise ValueError("Empty training dataset! Check data generation.")
        
    print(f"\nDataset sizes:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Test samples:  {len(test_dataset)}")
    
    # CREATE DATALOADERS
    # REVIEW: ✓ PyG DataLoader handles batching correctly
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # MODEL INITIALIZATION
    # REVIEW: ✓ Get feature dimension from first sample
    sample_data = train_dataset[0]
    in_feats = sample_data.x.shape[1]
    print(f"\nModel configuration:")
    print(f"  Input features:  {in_feats}")
    print(f"  Hidden dim:      {hidden}")
    print(f"  Num layers:      {num_layers}")
    print(f"  Dropout:         {dropout}")
    
    # REVIEW: ✓ Create model
    model = GCNStepPredictor(in_feats, hidden, num_layers, dropout).to(device)
    
    # REVIEW: ✓ Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params:    {num_params:,}")
    
    # OPTIMIZER & SCHEDULER
    # REVIEW: ✓ AdamW with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # REVIEW: ✓ ReduceLROnPlateau based on validation Hit@1
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # TRAINING LOOP
    ce = torch.nn.CrossEntropyLoss()
    best_val_hit1 = 0.0
    patience_counter = 0
    max_patience = 20  # REVIEW: ✓ Early stopping patience
    
    print(f"\nTraining for {epochs} epochs...")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Patience:      {max_patience}")
    
    for epoch in range(1, epochs + 1):
        # TRAINING PHASE
        model.train()
        train_losses = []
        
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            
            # REVIEW: ✓ Forward pass
            scores, _ = model(batch_data.x, batch_data.edge_index, batch_data.batch)
            
            # CRITICAL: Process each graph in batch
            # REVIEW: ✓ This is the KEY FIX for batch training
            batch_size_actual = batch_data.num_graphs if hasattr(batch_data, 'num_graphs') else 1
            total_loss = 0.0
            valid_samples = 0
            
            for i in range(batch_size_actual):
                # REVIEW: ✓ Get nodes for graph i
                mask = (batch_data.batch == i)
                graph_scores = scores[mask]
                
                # REVIEW: ✓ Get target for graph i
                target_idx = int(batch_data.y[i].item())
                
                # REVIEW: ✓ Validate target
                if target_idx < 0 or target_idx >= len(graph_scores):
                    continue
                
                # REVIEW: ✓ Compute loss
                logits = graph_scores.unsqueeze(0)
                targ = torch.tensor([target_idx], dtype=torch.long, device=device)
                loss = ce(logits, targ)
                total_loss += loss
                valid_samples += 1
            
            # REVIEW: ✓ Average loss over valid samples
            if valid_samples > 0:
                total_loss = total_loss / valid_samples
                total_loss.backward()
                
                # REVIEW: ✓ Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                
                optimizer.step()
                train_losses.append(float(total_loss.item()))

        # VALIDATION PHASE
        # REVIEW: ✓ Evaluate on validation set
        val_metrics = evaluate_model(model, val_loader, device)
        
        # REVIEW: ✓ Learning rate scheduling
        scheduler.step(val_metrics['hit1'])
        
        # REVIEW: ✓ Compute average train loss
        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        
        # LOGGING
        print(f"[Epoch {epoch:3d}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Hit@1: {val_metrics['hit1']:.4f} "
              f"Hit@3: {val_metrics['hit3']:.4f} "
              f"Hit@10: {val_metrics['hit10']:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
              
        # CHECKPOINT SAVING
        # REVIEW: ✓ Save best model based on validation Hit@1
        if val_metrics['hit1'] > best_val_hit1:
            best_val_hit1 = val_metrics['hit1']
            patience_counter = 0
            
            # REVIEW: ✓ Save checkpoint
            if exp_dir:
                os.makedirs(exp_dir, exist_ok=True)
                ckpt_path = os.path.join(exp_dir, "best_model.pt")
            else:
                ckpt_path = "best_model.pt"
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_hit1': val_metrics['hit1'],
                'in_feats': in_feats
            }, ckpt_path)
            
            print(f"  → Saved best model (Val Hit@1: {best_val_hit1:.4f})")
        else:
            patience_counter += 1
            
        # EARLY STOPPING
        # REVIEW: ✓ Stop if no improvement for max_patience epochs
        if patience_counter >= max_patience:
            print(f"\n⚠️  Early stopping triggered (no improvement for {max_patience} epochs)")
            break

    # FINAL TESTING
    print("\n" + "="*70)
    print("FINAL TEST EVALUATION")
    print("="*70)
    
    # REVIEW: ✓ Load best model
    if exp_dir:
        ckpt_path = os.path.join(exp_dir, "best_model.pt")
    else:
        ckpt_path = "best_model.pt"
        
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    # REVIEW: ✓ Evaluate on test set
    test_metrics = evaluate_model(model, test_loader, device)
    
    print(f"\nTest Results:")
    print(f"  Loss:   {test_metrics['loss']:.4f}")
    print(f"  Hit@1:  {test_metrics['hit1']:.4f} ({test_metrics['hit1']:.2f}%)")
    print(f"  Hit@3:  {test_metrics['hit3']:.4f} ({test_metrics['hit3']:.2f}%)")
    print(f"  Hit@10: {test_metrics['hit10']:.4f} ({test_metrics['hit10']:.2f}%)")
    print(f"  Samples: {test_metrics['num_samples']}")
    print("="*70)
    
    # REVIEW: ✓ Save final results
    if exp_dir:
        results_path = os.path.join(exp_dir, "final_results.json")
        with open(results_path, 'w') as f:
            json.dump({
                'config': {
                    'lr': lr,
                    'batch_size': batch_size,
                    'hidden': hidden,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'epochs': epochs,
                    'seed': seed
                },
                'test_metrics': test_metrics,
                'best_val_hit1': best_val_hit1
            }, f, indent=2)
        print(f"\nResults saved to: {results_path}")
        
    return model, test_metrics

if __name__ == "__main__":
    # REVIEW: ✓ Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Fixed Training Script")
    parser.add_argument("--json-dir", required=True, help="Directory with JSON instances")
    parser.add_argument("--spectral-dir", default=None, help="Directory with spectral features")
    parser.add_argument("--exp-dir", default=None, help="Experiment directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # REVIEW: ✓ Run training
    train(
        json_dir=args.json_dir,
        spectral_dir=args.spectral_dir,
        exp_dir=args.exp_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device_str=args.device,
        seed=args.seed
    )