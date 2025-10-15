# src/train_tptp.py
"""
Training pipeline specifically for TPTP dataset.

Key differences from synthetic training:
- No proof steps in raw TPTP (need to generate or use theorem prover output)
- Focus on conjecture entailment rather than step prediction
- Different evaluation metrics (provability, entailment)

Two training modes:
1. Axiom Selection: Predict which axioms are relevant for proving conjecture
2. Formula Entailment: Predict if conjecture follows from axioms
"""

import os
import json
import glob
import argparse
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from torch_geometric.data import Data
except ImportError:
    raise RuntimeError("torch_geometric required")

from src.models.gcn_step_predictor import GCNStepPredictor


class TPTPAxiomSelectionDataset(Dataset):
    """
    Dataset for axiom selection task on TPTP problems.
    
    Task: Given axioms and conjecture, predict which axioms are relevant.
    
    Since TPTP doesn't provide proof steps, we create synthetic training signal:
    - Positive samples: axioms that share symbols with conjecture
    - Negative samples: axioms that don't share symbols
    """
    
    def __init__(self, json_files: List[str], mode: str = "axiom_selection", seed: int = 42):
        self.files = json_files
        self.mode = mode
        self.samples = []
        
        np.random.seed(seed)
        
        for file_path in self.files:
            try:
                with open(file_path) as f:
                    inst = json.load(f)
                
                # Filter: only problems with conjecture
                if not inst["metadata"].get("has_conjecture", False):
                    continue
                
                # Create samples based on mode
                if mode == "axiom_selection":
                    self._create_axiom_selection_samples(inst)
                elif mode == "entailment":
                    self._create_entailment_sample(inst)
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
    
    def _create_axiom_selection_samples(self, inst: Dict):
        """
        Create axiom selection samples.
        For each problem, create multiple samples marking different axioms as target.
        """
        # Find axiom nodes and conjecture node
        axiom_nodes = [n for n in inst["nodes"] if n["type"] == "axiom"]
        conjecture_nodes = [n for n in inst["nodes"] if n["type"] == "conjecture"]
        
        if not conjecture_nodes or not axiom_nodes:
            return
        
        conjecture_nid = conjecture_nodes[0]["nid"]
        
        # Heuristic: axioms connected to same predicates as conjecture are relevant
        # Get predicates used by conjecture
        conj_predicates = set()
        for edge in inst["edges"]:
            if edge["src"] == conjecture_nid:
                dst_node = next(n for n in inst["nodes"] if n["nid"] == edge["dst"])
                if dst_node["type"] == "predicate":
                    conj_predicates.add(dst_node["label"])
        
        # For each axiom, determine if it's relevant
        for axiom in axiom_nodes:
            # Get predicates in this axiom
            axiom_predicates = set()
            for edge in inst["edges"]:
                if edge["src"] == axiom["nid"]:
                    dst_node = next(n for n in inst["nodes"] if n["nid"] == edge["dst"])
                    if dst_node["type"] == "predicate":
                        axiom_predicates.add(dst_node["label"])
            
            # Label: relevant if shares predicates with conjecture
            is_relevant = len(conj_predicates & axiom_predicates) > 0
            
            self.samples.append({
                "instance": inst,
                "target_axiom_nid": axiom["nid"],
                "is_relevant": is_relevant
            })
    
    def _create_entailment_sample(self, inst: Dict):
        """
        Create binary entailment sample: do axioms entail conjecture?
        This is simpler: one sample per problem.
        """
        self.samples.append({
            "instance": inst,
            "has_conjecture": True  # Always True since we filtered
        })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        inst = sample["instance"]
        
        # Build graph
        nodes = inst["nodes"]
        edges = inst["edges"]
        
        # Node ordering
        nodes_sorted = sorted(nodes, key=lambda x: x["nid"])
        nid_to_idx = {n["nid"]: i for i, n in enumerate(nodes_sorted)}
        n_nodes = len(nodes_sorted)
        
        # Node features: one-hot type encoding
        types = [n["type"] for n in nodes_sorted]
        unique_types = sorted(list(set(types)))
        type2idx = {t: i for i, t in enumerate(unique_types)}
        
        node_features = np.zeros((n_nodes, len(unique_types)), dtype=np.float32)
        for i, node_type in enumerate(types):
            node_features[i, type2idx[node_type]] = 1.0
        
        # Edge index
        src, dst = [], []
        for edge in edges:
            s, d = edge["src"], edge["dst"]
            if s in nid_to_idx and d in nid_to_idx:
                src.append(nid_to_idx[s])
                dst.append(nid_to_idx[d])
        
        edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)
        
        # Create PyG Data object
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index
        )
        
        # Add task-specific labels
        if self.mode == "axiom_selection":
            target_idx = nid_to_idx[sample["target_axiom_nid"]]
            data.y = torch.tensor([target_idx], dtype=torch.long)
            data.is_relevant = torch.tensor([sample["is_relevant"]], dtype=torch.float)
        elif self.mode == "entailment":
            # Binary classification: provable or not
            # For now, assume all are provable (since they're from TPTP)
            data.y = torch.tensor([1], dtype=torch.long)
        
        data.meta = {"id": inst["id"]}
        
        return data


class TPTPEntailmentModel(nn.Module):
    """
    Model for TPTP entailment prediction.
    Uses GCN backbone with graph-level classification head.
    """
    
    def __init__(self, in_feats: int, hidden: int = 128, num_layers: int = 2):
        super().__init__()
        
        # GCN backbone
        self.gcn = GCNStepPredictor(in_feats, hidden, num_layers)
        
        # Graph-level classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 2)  # Binary: provable/not provable
        )
    
    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: node features (N, F)
            edge_index: edge indices (2, E)
            batch: batch assignment (N,) for graph-level pooling
        
        Returns:
            logits: (B, 2) for binary classification
            node_emb: (N, hidden) node embeddings
        """
        # Get node embeddings
        _, node_emb = self.gcn(x, edge_index)
        
        # Graph-level pooling
        if batch is not None:
            # Mean pooling over nodes in each graph
            num_graphs = batch.max().item() + 1
            graph_emb = torch.zeros(num_graphs, node_emb.size(1), device=x.device)
            for i in range(num_graphs):
                mask = batch == i
                graph_emb[i] = node_emb[mask].mean(dim=0)
        else:
            # Single graph
            graph_emb = node_emb.mean(dim=0, keepdim=True)
        
        # Classify
        logits = self.classifier(graph_emb)
        
        return logits, node_emb


def train_tptp(
    json_dir: str,
    mode: str = "axiom_selection",
    epochs: int = 30,
    lr: float = 1e-3,
    hidden: int = 128,
    device_str: str = "cpu",
    seed: int = 42
):
    """
    Train on TPTP dataset.
    
    Args:
        json_dir: Directory with converted TPTP JSON files
        mode: "axiom_selection" or "entailment"
        epochs: Number of training epochs
        lr: Learning rate
        hidden: Hidden dimension
        device_str: Device (cpu/cuda)
        seed: Random seed
    """
    device = torch.device(device_str)
    
    # Load data
    files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    files = [f for f in files if not f.endswith("conversion_stats.json")]
    
    if not files:
        print(f"No JSON files found in {json_dir}")
        return
    
    print(f"Found {len(files)} TPTP problems")
    
    # Split data
    np.random.seed(seed)
    np.random.shuffle(files)
    
    n_test = max(1, len(files) // 10)
    n_val = max(1, len(files) // 10)
    
    train_files = files[: -(n_val + n_test)]
    val_files = files[-(n_val + n_test): -n_test]
    test_files = files[-n_test:]
    
    print(f"Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    # Create datasets
    train_ds = TPTPAxiomSelectionDataset(train_files, mode=mode, seed=seed)
    val_ds = TPTPAxiomSelectionDataset(val_files, mode=mode, seed=seed + 1)
    test_ds = TPTPAxiomSelectionDataset(test_files, mode=mode, seed=seed + 2)
    
    print(f"Samples: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")
    
    if len(train_ds) == 0:
        print("No training samples created. Check that TPTP files have conjectures.")
        return
    
    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # Create model
    sample = train_ds[0]
    in_feats = sample.x.size(1)
    
    if mode == "axiom_selection":
        model = GCNStepPredictor(in_feats, hidden, num_layers=2).to(device)
        criterion = nn.BCEWithLogitsLoss()  # Binary classification per axiom
    else:  # entailment
        model = TPTPEntailmentModel(in_feats, hidden, num_layers=2).to(device)
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Mode: {mode}")
    print(f"Training for {epochs} epochs...\n")
    
    best_val_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            if mode == "axiom_selection":
                scores, _ = model(data.x, data.edge_index)
                target = data.is_relevant.view(-1)
                
                # Sigmoid + BCE loss
                pred = torch.sigmoid(scores[data.y.item()])
                loss = criterion(pred.unsqueeze(0), target)
                
                # Accuracy
                pred_binary = (pred > 0.5).float()
                train_correct += (pred_binary == target).sum().item()
                train_total += 1
                
            else:  # entailment
                logits, _ = model(data.x, data.edge_index)
                loss = criterion(logits, data.y)
                
                pred_class = logits.argmax(dim=1)
                train_correct += (pred_class == data.y).sum().item()
                train_total += 1
            
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        train_acc = train_correct / max(1, train_total)
        
        # Validate
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                
                if mode == "axiom_selection":
                    scores, _ = model(data.x, data.edge_index)
                    target = data.is_relevant.view(-1)
                    pred = torch.sigmoid(scores[data.y.item()])
                    loss = criterion(pred.unsqueeze(0), target)
                    
                    pred_binary = (pred > 0.5).float()
                    val_correct += (pred_binary == target).sum().item()
                    val_total += 1
                    
                else:  # entailment
                    logits, _ = model(data.x, data.edge_index)
                    loss = criterion(logits, data.y)
                    
                    pred_class = logits.argmax(dim=1)
                    val_correct += (pred_class == data.y).sum().item()
                    val_total += 1
                
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        val_acc = val_correct / max(1, val_total)
        
        # Print progress
        print(f"[Epoch {epoch}/{epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "in_feats": in_feats,
                "mode": mode,
                "epoch": epoch,
                "val_acc": val_acc
            }, "checkpoints/tptp_best.pt")
            print(f"  → Saved best model (val_acc={val_acc:.4f})")
    
    # Final test
    print("\n" + "="*70)
    print("Final Test Evaluation")
    print("="*70)
    
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            
            if mode == "axiom_selection":
                scores, _ = model(data.x, data.edge_index)
                target = data.is_relevant.view(-1)
                pred = torch.sigmoid(scores[data.y.item()])
                pred_binary = (pred > 0.5).float()
                test_correct += (pred_binary == target).sum().item()
            else:
                logits, _ = model(data.x, data.edge_index)
                pred_class = logits.argmax(dim=1)
                test_correct += (pred_class == data.y).sum().item()
            
            test_total += 1
    
    test_acc = test_correct / max(1, test_total)
    
    print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test Correct: {test_correct}/{test_total}")
    print("="*70)
    
    return model, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on TPTP dataset")
    parser.add_argument("--json-dir", required=True, help="Directory with converted TPTP JSON files")
    parser.add_argument("--mode", default="axiom_selection", 
                       choices=["axiom_selection", "entailment"],
                       help="Training mode")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    train_tptp(
        args.json_dir,
        mode=args.mode,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        device_str=args.device,
        seed=args.seed
    )