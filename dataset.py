"""
Feature-Rich Dataset for Step Prediction
Key fixes:
- Added critical rule-specific features
"""

import json
import random
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from collections import deque

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class StepPredictionDataset(Dataset):
    """
    Dataset for step prediction with RICH features.
    
    Node features (24-dim base):
    [0-1]: Type (fact/rule) one-hot
    [2]: Is initially given (not derived)
    [3]: Is derived up to this step
    [4]: Fraction of premises satisfied (0-1, continuous)
    [5-6]: Normalized degree (in/out)
    [7]: Normalized depth from initial facts
    [8]: Normalized rule body size (0 for facts)
    [9]: Betweenness centrality (normalized)
    [10-17]: Predicate hash (8-bit binary)
    [18-19]: Graph-level features (graph size, proof progress)
    [20]: Is head of rule derived (for rules)
    [21]: All premises available (binary, can fire)
    [22]: Number of times atom appears in graph
    [23]: Is in current proof frontier
    """
    
    def __init__(self,
                 json_files: List[str],
                 spectral_dir: Optional[str] = None,
                 seed: int = 42):
        self.files = json_files
        self.spectral_dir = spectral_dir
        self.samples = []
        
        random.seed(seed)
        
        print("Loading dataset...")
        skipped = 0
        
        for f in json_files:
            try:
                with open(f) as fp:
                    inst = json.load(fp)
                
                proof = inst.get("proof_steps", [])
                
                if len(proof) == 0:
                    skipped += 1
                    continue
                
                for step_idx in range(len(proof)):
                    self.samples.append((inst, step_idx))
            
            except Exception as e:
                print(f"Error loading {f}: {e}")
                skipped += 1
        
        print(f"Loaded {len(self.samples)} samples from {len(json_files)} files")
        print(f"Skipped {skipped} files")
    
    def __len__(self):
        return len(self.samples)
    
    def _compute_node_features(self, inst: Dict, step_idx: int) -> torch.Tensor:
        """Compute rich node features."""
        nodes = inst["nodes"]
        edges = inst["edges"]
        proof = inst["proof_steps"]
        
        n_nodes = len(nodes)
        
        # Create node ID mapping
        id2idx = {n["nid"]: i for i, n in enumerate(nodes)}
        
        # Initialize features
        feats = torch.zeros(n_nodes, 22)
        
        # Track what's derived up to this step
        derived = set()
        initial_facts = set()
        
        for node in nodes:
            if node["type"] == "fact" and node.get("is_initial", False):
                initial_facts.add(node["nid"])
        
        for i in range(step_idx):
            derived.add(proof[i]["derived_node"])
        
        known = initial_facts | derived
        
        # Compute degrees
        in_deg = np.zeros(n_nodes)
        out_deg = np.zeros(n_nodes)
        
        for e in edges:
            src_idx = id2idx[e["src"]]
            dst_idx = id2idx[e["dst"]]
            out_deg[src_idx] += 1
            in_deg[dst_idx] += 1
        
        max_deg = max(max(in_deg), max(out_deg), 1)
        
        # Compute depths
        depths = {}
        queue = deque()
        
        for fact_id in initial_facts:
            depths[fact_id] = 0
            queue.append((fact_id, 0))
        
        visited = set(initial_facts)
        
        while queue:
            nid, depth = queue.popleft()
            
            for e in edges:
                if e["src"] == nid and e["dst"] not in visited:
                    next_nid = e["dst"]
                    visited.add(next_nid)
                    depths[next_nid] = depth + 1
                    queue.append((next_nid, depth + 1))
        
        max_depth = max(depths.values()) if depths else 1
        
        # Count atom occurrences
        atom_counts = {}
        for node in nodes:
            atom = node.get("atom", node.get("head_atom", ""))
            atom_counts[atom] = atom_counts.get(atom, 0) + 1
        
        max_count = max(atom_counts.values()) if atom_counts else 1
        
        # Build features for each node
        for i, node in enumerate(nodes):
            nid = node["nid"]
            
            # [0-1]: Type one-hot
            if node["type"] == "fact":
                feats[i, 0] = 1.0
            else:
                feats[i, 1] = 1.0
            
            # [2]: Is initial
            feats[i, 2] = 1.0 if nid in initial_facts else 0.0
            
            # [3]: Is derived
            feats[i, 3] = 1.0 if nid in derived else 0.0
            
            # [4]: Fraction of premises satisfied (continuous)
            # if node["type"] == "rule":
            #     body_atoms = node["body_atoms"]
            #     satisfied_count = sum(
            #         1 for atom in body_atoms
            #         if any(n["atom"] == atom and n["nid"] in known 
            #                for n in nodes if n["type"] == "fact")
            #     )
            #     feats[i, 4] = satisfied_count / max(len(body_atoms), 1)
            
            # [5-6]: Degrees
            feats[i, 4] = in_deg[i] / max_deg
            feats[i, 5] = out_deg[i] / max_deg
            
            # [7]: Depth
            feats[i, 6] = depths.get(nid, max_depth) / max(max_depth, 1)
            
            # [8]: Body size (for rules)
            if node["type"] == "rule":
                body_size = len(node["body_atoms"])
                max_body = max(len(n["body_atoms"]) for n in nodes if n["type"] == "rule")
                feats[i, 7] = body_size / max(max_body, 1)
            
            # [9]: Centrality
            feats[i, 8] = (in_deg[i] + out_deg[i]) / (2 * max_deg)
            
            # [10-17]: Predicate hash
            atom = node.get("atom", node.get("head_atom", ""))
            atom_clean = atom.replace("~", "")
            pred_hash = abs(hash(atom_clean)) % 256
            
            for bit in range(8):
                feats[i, 9 + bit] = float((pred_hash >> bit) & 1)
            
            # [18-19]: Graph-level context
            feats[i, 17] = n_nodes / 100.0
            feats[i, 18] = step_idx / max(len(proof), 1)
            
            # [20]: Is head of rule derived (for rules)
            if node["type"] == "rule":
                head_atom = node["head_atom"]
                feats[i, 19] = 1.0 if any(n["atom"] == head_atom and n["nid"] in known 
                                          for n in nodes if n["type"] == "fact") else 0.0
            
            # # [21]: All premises available (can fire)
            # if node["type"] == "rule":
            #     body_atoms = node["body_atoms"]
            #     can_fire = all(
            #         any(n["atom"] == atom and n["nid"] in known 
            #             for n in nodes if n["type"] == "fact")
            #         for atom in body_atoms
            #     )
            #     feats[i, 21] = 1.0 if can_fire else 0.0
            
            # [22]: Atom occurrence frequency
            atom = node.get("atom", node.get("head_atom", ""))
            feats[i, 20] = atom_counts.get(atom, 0) / max_count
            
            # [23]: Is in current frontier (derived in last step)
            if step_idx > 0:
                last_derived = proof[step_idx - 1]["derived_node"]
                feats[i, 21] = 1.0 if nid == last_derived else 0.0
        
        return feats
    
    def _load_spectral(self, inst_id: str) -> Optional[torch.Tensor]:
        """Load spectral features if available."""
        if not self.spectral_dir:
            return None
        
        spectral_path = Path(self.spectral_dir) / f"{inst_id}_spectral.npz"
        
        if not spectral_path.exists():
            return None
        
        try:
            data = np.load(spectral_path)
            eigvecs = data["eigvecs"]
            return torch.from_numpy(eigvecs).float()
        except:
            return None
    
    def __getitem__(self, idx: int) -> Data:
        """Get a training sample."""
        inst, step_idx = self.samples[idx]
        
        nodes = inst["nodes"]
        edges = inst["edges"]
        proof = inst["proof_steps"]
        
        # Create node ID mapping
        id2idx = {n["nid"]: i for i, n in enumerate(nodes)}
        
        # Node features
        x_base = self._compute_node_features(inst, step_idx)
        
        # Add spectral features
        x_spectral = self._load_spectral(inst.get("id", ""))
        
        if x_spectral is not None and x_spectral.shape[0] == x_base.shape[0]:
            x = torch.cat([x_base, x_spectral], dim=-1)
        else:
            x = x_base
        
        # Edge index with edge types
        edge_list = []
        edge_types = []
        
        for e in edges:
            src = id2idx[e["src"]]
            dst = id2idx[e["dst"]]
            edge_list.append([src, dst])
            
            # Edge type: 0=body, 1=head, 2=other
            etype = 0 if e["etype"] == "body" else (1 if e["etype"] == "head" else 2)
            edge_types.append(etype)
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_types, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.long)
        
        # Target: rule used in this step
        # Store LOCAL index (0 to N-1), not global node ID
        step = proof[step_idx]
        rule_nid = step["used_rule"]
        target_idx = id2idx[rule_nid]  # This is already 0-indexed for this graph
        
        y = torch.tensor([target_idx], dtype=torch.long)
        
        # Create PyG Data
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
        data.meta = {
            "instance_id": inst.get("id", ""),
            "step_idx": step_idx,
            "num_nodes": len(nodes)
        }
        
        return data


def create_split(
    json_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> tuple:
    """Create train/val/test split at INSTANCE level."""
    all_files = list(Path(json_dir).rglob("*.json"))
    
    instance_map = {}
    for f in all_files:
        try:
            with open(f) as fp:
                inst = json.load(fp)
            inst_id = inst.get("id", str(f))
            instance_map[inst_id] = str(f)
        except:
            continue
    
    instance_ids = list(instance_map.keys())
    random.seed(seed)
    random.shuffle(instance_ids)
    
    n = len(instance_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_ids = instance_ids[:n_train]
    val_ids = instance_ids[n_train:n_train + n_val]
    test_ids = instance_ids[n_train + n_val:]
    
    train_files = [instance_map[i] for i in train_ids]
    val_files = [instance_map[i] for i in val_ids]
    test_files = [instance_map[i] for i in test_ids]
    
    print(f"\nDataset split (instance-level):")
    print(f"  Train: {len(train_files)} files ({len(train_ids)} instances)")
    print(f"  Val:   {len(val_files)} files ({len(val_ids)} instances)")
    print(f"  Test:  {len(test_files)} files ({len(test_ids)} instances)")
    
    assert len(set(train_ids) & set(val_ids)) == 0
    assert len(set(train_ids) & set(test_ids)) == 0
    assert len(set(val_ids) & set(test_ids)) == 0
    
    return train_files, val_files, test_files