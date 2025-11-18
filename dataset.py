"""
Feature-Rich Dataset for Step Prediction (FIXED)
================================================

This file now contains the *single*, definitive ProofStepDataset
AND the 'fixed_collate_fn' directly inside it.

CRITICAL FIX (Issue 3):
- _compute_node_features now correctly computes feature [21]
  (rule applicability fraction) instead of hardcoding it to 0.

INTEGRATION:
- Replaced the old `_compute_node_features` with the new
  `FeatureComputer` class from `dataset_new.py` for
  correct 29-dimensional feature calculation.
"""

import copy
import json
import random
import re
import numpy as np
from typing import Any, List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict, deque
import networkx as nx

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch.utils.data import DataLoader as GeoDataLoader
import torch_geometric.utils as pyg_utils  # Add import
import torch.nn.functional as F

import time 
import logging

from tqdm import tqdm
from data_generator import ProofVerifier
from math import log

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ADD THESE TWO FUNCTIONS to dataset.py

def compute_derived_mask(is_derived_column: torch.Tensor) -> torch.Tensor:
    """
    Converts the 'is_derived' feature column into the boolean mask.
    The call in __getitem__ was `compute_derived_mask(features[:, 3])`.
    This function simply implements the intent of that call.
    """
    # Use .bool() for a proper mask, or .long() if other code expects 0/1
    return is_derived_column.bool() 

def compute_step_numbers(
    derived_mask: torch.Tensor, 
    proof_steps: List[Dict], 
    current_step_idx: int, 
    id2idx: Dict
) -> torch.Tensor:
    """
    This is the function your __getitem__ is trying to call.
    It builds a tensor where each entry is the step number
    at which that node was derived.
    """
    num_nodes = len(derived_mask)
    step_num_tensor = torch.zeros(num_nodes, dtype=torch.long)
    
    # Iterate through proof history *up to the current step*
    for step_info in proof_steps[:current_step_idx + 1]:
        derived_nid = step_info.get('derived_node')
        if derived_nid in id2idx:
            node_idx = id2idx[derived_nid]
            if node_idx < num_nodes:
                # Store the step number (step_id + 1, since step_id is 0-indexed)
                step_num_tensor[node_idx] = step_info.get('step_id', 0) + 1
                
    return step_num_tensor

# ==============================================================================
# SELF-CONTAINED COLLATE FUNCTION (MOVED FROM dataset_utils.py)
# ==============================================================================
def fixed_collate_fn(batch_list: List[Data]) -> Batch:
    """
    Custom collation that preserves graph-level metadata.
    This is now part of dataset.py to avoid import errors.
    
    FIXED:
    1. Correctly offsets 'data.y' *before* batching.
    2. Calls Batch.from_data_list *only once*.
    3. Attaches 'batch.node_offsets' for use in training loop.
    """
    
    # Filter None samples
    batch_list = [b for b in batch_list if b is not None]
    
    if len(batch_list) == 0:
        return None

    # === THIS IS THE CRITICAL FIX ===
    # 1. Calculate offsets and modify data.y *before* batching
    node_offsets_list = [0] * len(batch_list)
    cumulative_offset = 0
    for i, data in enumerate(batch_list):
        # Save the offset for this graph
        node_offsets_list[i] = cumulative_offset
        
        # Modify data.y to be the global index
        # This is what Batch.from_data_list will use
        data.y = data.y + cumulative_offset
        
        cumulative_offset += data.x.shape[0]
    # === END FIX ===

    # 2. Call Batch.from_data_list *once*
    # PyG's Batch.from_data_list will now automatically create 
    # a 'batch.y' tensor with the *correctly offset* global indices.
    try:
        follow_attrs = ['x', 'eigvecs', 'derived_mask', 'step_numbers', 'applicable_mask']
        existing_follow_attrs = [attr for attr in follow_attrs if hasattr(batch_list[0], attr)]
        
        batch = Batch.from_data_list(batch_list, follow_batch=existing_follow_attrs)
    except Exception as e:
        logger.error(f"ERROR during Batch.from_data_list: {e}")
        return None
    
    # 3. Add critical metadata for index mapping
    try:
        # This creates batch.num_nodes_per_graph as a TENSOR
        batch.num_nodes_per_graph = torch.tensor(
            [data.x.shape[0] for data in batch_list],
        dtype=torch.long
        )
        num_nodes_per_graph = torch.bincount(batch.batch)
        node_offsets = torch.cat([
            torch.tensor([0], device='cpu'),
            num_nodes_per_graph.cumsum(0)[:-1]
        ])

        # Store in batch
        batch.node_offsets = node_offsets
    except Exception as e:
        logger.error(f"ERROR adding metadata: {e}")
    
    # Add custom attributes (str/int lists)
    batch.difficulties = [data.difficulty for data in batch_list]
    batch.step_indices = [data.step_idx for data in batch_list]
    batch.proof_lengths = [data.proof_length for data in batch_list]
    batch.meta_list = [
        {
            'difficulty': data.difficulty,
            'step_idx': data.step_idx,
            'proof_length': data.proof_length
        }
        for data in batch_list
    ]

    return batch


# ==============================================================================
# NEW: CORRECT FEATURE COMPUTER (from dataset_new.py)
# ==============================================================================
class FeatureComputer:
    """
    FAST Feature Computation.
    
    OPTIMIZATIONS:
    1. REMOVED all O(N^3) NetworkX algorithms (Centrality, Clustering, Cycles).
    2. Uses direct edge counting for degrees (O(E)).
    3. Calculates rule applicability efficiently.
    """
    
    def __init__(self):
        self.feature_dim = 29
    
    def compute_features(self, nodes, edges, proof_steps, step_idx, id2idx):
        N = len(nodes)
        features = torch.zeros((N, self.feature_dim), dtype=torch.float32)
        
        # 1. Fast Degree Calculation (No NetworkX)
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        
        # Build adjacency for local neighborhood stats (fast)
        adj = defaultdict(list)
        
        for e in edges:
            if e['src'] in id2idx and e['dst'] in id2idx:
                src_idx = id2idx[e['src']]
                dst_idx = id2idx[e['dst']]
                
                out_degree[src_idx] += 1
                in_degree[dst_idx] += 1
                adj[src_idx].append(dst_idx)
                adj[dst_idx].append(src_idx) # Undirected view for neighbors

        max_degree = 1
        if in_degree or out_degree:
            max_degree = max(max(in_degree.values(), default=0), 
                             max(out_degree.values(), default=0))
            max_degree = max(max_degree, 1)

        # 2. Identify Derived/Known Set
        derived_indices = set()
        known_atoms = set()
        
        # Initial facts
        for i, node in enumerate(nodes):
            if node.get('type') == 'fact' and node.get('is_initial', False):
                known_atoms.add(node['atom'])
        
        # Derived facts
        for step in proof_steps[:step_idx + 1]:
            derived_nid = step.get('derived_node')
            if derived_nid in id2idx:
                idx = id2idx[derived_nid]
                derived_indices.add(idx)
                if idx < N and nodes[idx].get('type') == 'fact':
                    known_atoms.add(nodes[idx]['atom'])

        # 3. Compute Node Features
        for i, node in enumerate(nodes):
            node_type = node.get('type', 'unknown')
            
            # --- Basic Features [0-4] ---
            features[i, 0] = 1.0 if node_type == 'fact' else 0.0
            features[i, 1] = 1.0 if node_type == 'rule' else 0.0
            features[i, 2] = min(len(node.get('atom', '')) / 100.0, 1.0)
            features[i, 3] = 1.0 if i in derived_indices else 0.0
            features[i, 4] = 1.0 if node.get('is_initial', False) else 0.0
            
            # --- Fast Structure Features [5-7] ---
            d_in = in_degree.get(i, 0)
            d_out = out_degree.get(i, 0)
            features[i, 5] = (d_in + d_out) / max_degree
            features[i, 6] = d_in / max_degree
            features[i, 7] = d_out / max_degree
            
            # --- Disabled Expensive Metrics [8-11] ---
            # These are O(N^3) and kill performance. Model learns them via GAT.
            features[i, 8] = 0.0  # Clustering
            features[i, 9] = 0.0  # Betweenness
            features[i, 10] = 0.0 # Closeness
            features[i, 11] = 0.0 # PageRank
            
            # --- Rule Features [12, 18-21] ---
            if node_type == 'rule':
                body = node.get('body_atoms', [])
                features[i, 12] = len(body)
                features[i, 18] = len(body) / 10.0
                features[i, 19] = 1.0 if node.get('head_atom') in known_atoms else 0.0
                features[i, 20] = len(body)
                
                # Rule Applicability
                if body:
                    satisfied = sum(1 for a in body if a in known_atoms)
                    features[i, 21] = satisfied / len(body)
                else:
                    features[i, 21] = 1.0 # Axiom rule
            
            # --- Fact Features [13] ---
            elif node_type == 'fact':
                # Approximation: Just check out-degree (facts -> rules)
                features[i, 13] = d_out / 10.0
            
            # --- Topology [14-17] ---
            features[i, 14] = 0.0 # Depth (O(N) BFS, skipped for speed)
            features[i, 15] = 1.0 if d_out == 0 else 0.0 # Leaf
            features[i, 16] = 1.0 if d_in == 0 else 0.0  # Root
            
            # Avg neighbor degree
            neighbors = adj.get(i, [])
            if neighbors:
                avg_d = sum(in_degree[n] + out_degree[n] for n in neighbors) / len(neighbors)
                features[i, 17] = avg_d / max_degree
            
            # --- Advanced/Temporal [22-28] ---
            features[i, 23] = (d_in + d_out) / max_degree # Laplacian diag proxy
            
            # Temporal Recency
            max_s = len(proof_steps)
            if max_s > 0:
                features[i, 28] = 1.0 - (step_idx / max_s)
            else:
                features[i, 28] = 1.0

        return features

class NormalizedFeatureComputer(FeatureComputer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Precomputed statistics from training set
        self.feature_mean = None
        self.feature_std = None
    
    def fit_normalizer(self, train_dataset):
        """Compute normalization statistics from training set"""
        all_features = []
        for data in tqdm(train_dataset, desc="Computing feature stats"):
            all_features.append(data.x.cpu())
        
        all_features = torch.cat(all_features, dim=0)
        self.feature_mean = all_features.mean(dim=0)
        self.feature_std = all_features.std(dim=0) + 1e-6
    
    def normalize(self, features):
        """Apply z-score normalization"""
        if self.feature_mean is None:
            return features  # First epoch, no stats yet
        
        return (features - self.feature_mean) / self.feature_std

# ==============================================================================
# MAIN DATASET CLASS (NOW USES FeatureComputer)
# ==============================================================================

class ProofStepDataset(Dataset):
    """
    Dataset for proof step prediction.
    Each item is a graph state at a specific proof step, with target rule index.
    """
    
    def __init__(self, file_paths: List[str], spectral_dir: Optional[str] = None, seed: int = 42):
        super().__init__()
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Convert to Path objects to fix 'str' no 'stem'
        self.file_paths = [Path(p) for p in file_paths]
        self.spectral_dir = Path(spectral_dir) if spectral_dir else None
        self.samples = []  # List of (inst_id, step_idx, original_inst_id, original_step_idx)
        self.instances = {}  # Dict of instance data by inst_id
        
        # --- INTEGRATION: Initialize the new feature computer ---
        self.feature_computer = FeatureComputer()
        
        self._load_samples()
        
        # Validate all samples to prevent runtime errors
        valid_samples = []
        for sample in self.samples:
            inst_id, step_idx, _, _ = sample
            inst = self.instances.get(inst_id)
            if inst is None:
                logger.warning(f"Skipping missing instance {inst_id}")
                continue
            proof_steps = inst.get('proof_steps', [])
            if not isinstance(proof_steps, list) or step_idx >= len(proof_steps) or step_idx < 0:
                logger.warning(f"Invalid sample {inst_id} step {step_idx}, proof_steps len={len(proof_steps)}")
                continue
            valid_samples.append(sample)
        self.samples = valid_samples
        
        logger.info(f"✅ Loaded {len(self.samples)} proof steps from {len(self.instances)} instances")
    
    def _load_samples(self):
        for file_path in tqdm(self.file_paths, desc="Loading files"):
            try:
                with open(file_path, 'r') as f:
                    inst = json.load(f)
                inst_id = inst.get('id', file_path.stem)  # Now file_path is Path, .stem works
                self.instances[inst_id] = inst
                proof_steps = inst.get('proof_steps', [])
                if not isinstance(proof_steps, list) or len(proof_steps) == 0:
                    logger.warning(f"Skipping invalid/empty instance {inst_id}")
                    continue
                for step_idx in range(len(proof_steps)):
                    self.samples.append((inst_id, step_idx, inst_id, step_idx))
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
    
    def _build_graph(self, inst: Dict, step_idx: int) -> Tuple[List[Dict], List[Dict]]:
        """
        Build cumulative graph up to step_idx: initial nodes/edges + derived up to step.
        """
        nodes = copy.deepcopy(inst.get('nodes', []))  # All nodes (facts, rules)
        edges = copy.deepcopy(inst.get('edges', []))  # All edges
        
        proof_steps = inst.get('proof_steps', [])
        
        # Add derived facts up to step_idx
        for s in range(step_idx + 1):  # Cumulative
            step = proof_steps[s]
            derived_nid = step.get('derived_node')
            # Add if not present (though should be in initial nodes)
            if not any(n['nid'] == derived_nid for n in nodes):
                nodes.append({'nid': derived_nid, 'type': 'fact', 'atom': step.get('derived_atom', 'unknown'), 'is_derived': True})
        
        # Filter edges to only connect present nodes (optional for robustness)
        present_nids = {n['nid'] for n in nodes}
        edges = [e for e in edges if e['src'] in present_nids and e['dst'] in present_nids]
        
        return nodes, edges
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Optional[Data]:
        try:
            inst_id, step_idx, _, _ = self.samples[idx]
            inst = self.instances[inst_id]
            proof_steps = inst.get('proof_steps', [])
            
            # Build graph up to this step
            nodes, edges = self._build_graph(inst, step_idx)
            
            if not nodes or not edges:
                raise ValueError("Empty graph")
            
            # ID to index mapping
            id2idx = {n['nid']: i for i, n in enumerate(nodes)}
            
            # Edge index and attributes
            src_indices = [id2idx[edge['src']] for edge in edges if edge['src'] in id2idx and edge['dst'] in id2idx]
            dst_indices = [id2idx[edge['dst']] for edge in edges if edge['src'] in id2idx and edge['dst'] in id2idx]
            edge_index = torch.tensor([[src, dst] for src, dst in zip(src_indices, dst_indices)], dtype=torch.long).t()
            edge_attr = torch.tensor([self._encode_edge_type(edge.get('etype', 'unknown')) for edge in edges if edge['src'] in id2idx and edge['dst'] in id2idx], dtype=torch.long) # <-- Change dtype to torch.long
            
            # --- INTEGRATION: Call the new FeatureComputer ---
            # Node features (29 dimensions)
            features = self.feature_computer.compute_features(
                nodes, edges, proof_steps, step_idx, id2idx
            )
            # --- END INTEGRATION ---
            
            # Derived mask and step numbers
            derived_mask = compute_derived_mask(features[:, 3])  # Assuming feature [3] is is_derived
            step_numbers = compute_step_numbers(derived_mask, proof_steps, step_idx, id2idx)
            
            # Applicable rules mask
            # THIS IS THE CRITICAL LOGIC FIX from the previous step
            applicable_mask, _ = self.compute_applicable_rules_for_step(
                nodes, edges, step_idx, step_numbers, id2idx
            )
            
            # Ground truth target (rule node index)
            gt_node_idx = self.get_ground_truth(proof_steps, step_idx, id2idx)
            
            # Load spectral features if available
            num_nodes = len(nodes) # Get num_nodes
            eigvecs, eigvals, eig_mask = self._load_spectral_features(inst_id, num_nodes)
            metadata = inst.get('metadata', {})
            proof_length = metadata.get('proof_length', len(proof_steps))
            current_proof_length = max(proof_length, 1.0) # Avoid division by zero
            value_target_float = 1.0 - (step_idx / current_proof_length)
            
            value_target_tensor = torch.tensor([value_target_float], dtype=torch.float)
            # Create PyG Data object
            data = Data(
                x=features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([gt_node_idx], dtype=torch.long),
                applicable_mask=applicable_mask,
                derived_mask=derived_mask,
                step_numbers=step_numbers,
                eigvecs=eigvecs,
                eigvals=eigvals,
                eig_mask=eig_mask,
                difficulty=metadata.get('difficulty', 'medium'), 
                step_idx=step_idx,  # int
                proof_length=proof_length,# <-- ALSO FIX
                value_target=value_target_tensor
            )
            
            return data
        
        except Exception as e:
            logger.error(f"[Dataset Error] Failed __getitem__ for instance {inst_id}, step {step_idx}: {e}")
            return None
    
    def _encode_edge_type(self, etype: str) -> int:
        """
        Encode edge type to float (e.g., for GATv2).
        """
        type_map = {
            'head': 1.0,
            'body': 0.5,
            'unknown': 0.0
            # Add more as per your data
        }
        return type_map.get(etype, 0.0)
    
    def get_ground_truth(self, proof_steps: List[Dict], step_idx: int, id2idx: Dict) -> int:
        """
        Get ground truth rule node index.
        """
        if step_idx >= len(proof_steps):
            return -1
        step = proof_steps[step_idx]
        gt_nid = step.get('used_rule')
        return id2idx.get(gt_nid, -1)  # -1 if not found
    
    def _get_dummy_spectral_features(self, n_nodes: int = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates and returns placeholder dummy tensors for spectral features.
        We use k=16 to match the default model.py configuration.
        """
        k = 16 # Default k_dim from model.py
        
        # Return all zeros, including a mask of zeros (False) to indicate no valid eigenvalues.
        return (
            torch.zeros((n_nodes, k), dtype=torch.float32),
            torch.zeros(k, dtype=torch.float32),
            torch.zeros(k, dtype=torch.bool)
        )

    def _load_spectral_features(self, inst_id: str, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load precomputed spectral features or dummy.
        """
        # Case 1: Spectral processing is disabled
        if self.spectral_dir is None:
            return self._get_dummy_spectral_features(num_nodes)
        
        # Case 2: A call with inst_id=None (should no longer happen, but safe)
        if inst_id is None:
             logger.warning("Spectral loader called with inst_id=None; using dummy")
             return self._get_dummy_spectral_features(num_nodes)

        cache_path = self.spectral_dir / f"{inst_id}_spectral.npz"
        
        # Case 3: Cache file exists
        if cache_path.exists():
            try:
                data = np.load(cache_path)
                eigvecs = torch.from_numpy(data['eigenvectors']).float()
                eigvals = torch.from_numpy(data['eigenvalues']).float()
                
                # Pad eigenvectors if graph is smaller than precomputed k
                k_dim = 16 # Should match model default
                
                # Pad/truncate eigenvectors
                if eigvecs.shape[0] != num_nodes:
                     # This indicates a mismatch between the graph and its cache file.
                    #  logger.warning(f"Node count mismatch for {inst_id} ({eigvecs.shape[0]} vs {num_nodes}). Using dummy.")
                     return self._get_dummy_spectral_features(num_nodes)
                
                if eigvecs.shape[1] < k_dim:
                    pad = (0, k_dim - eigvecs.shape[1])
                    eigvecs = F.pad(eigvecs, pad, "constant", 0)
                elif eigvecs.shape[1] > k_dim:
                    eigvecs = eigvecs[:, :k_dim]

                # Pad/truncate eigenvalues and create mask
                eig_mask = torch.zeros(k_dim, dtype=torch.bool)
                valid_len = min(len(eigvals), k_dim)
                eig_mask[:valid_len] = True
                
                if len(eigvals) < k_dim:
                    pad = (0, k_dim - len(eigvals))
                    eigvals = F.pad(eigvals, pad, "constant", 0)
                elif len(eigvals) > k_dim:
                    eigvals = eigvals[:k_dim]
                    
                return eigvecs, eigvals, eig_mask
            
            except Exception as e:
                logger.error(f"Error loading {cache_path} for {inst_id}: {e}. Using dummy.")
                return self._get_dummy_spectral_features(num_nodes)
        
        # Case 4: Cache file is missing (The original problem)
        else:
            logger.warning(f"Spectral cache missing for {inst_id}; using dummy")
            # --- THIS IS THE FIX ---
            # Instead of recursive call, return dummy data directly
            return self._get_dummy_spectral_features(num_nodes)
    
    # --- DELETED OLD `_compute_node_features` METHOD ---
    
    def compute_applicable_rules_for_step(self, nodes, edges, step_idx, step_numbers, id2idx):
        """
        CRITICAL FIX: This function now correctly uses `step_numbers` to find
        atoms known *before* the current `step_idx` is applied.
        """
        known_atoms = set()
        current_step_number = step_idx + 1 
        
        for i, node in enumerate(nodes):
            if node['type'] == 'fact':
                is_initial = node.get('is_initial', False)
                
                # A node is known if it was derived in a *previous* step
                # i.e., its step number is > 0 and *less than* the current step number
                is_derived_previously = (step_numbers[i] > 0) and (step_numbers[i] < current_step_number)
                
                if is_initial or is_derived_previously:
                    known_atoms.add(node['atom'])
        
        # The rest of the function remains the same
        applicable_mask = torch.zeros(len(nodes), dtype=torch.bool)
        for i, node in enumerate(nodes):
            if node['type'] == 'rule':
                body = set(node.get('body_atoms', []))
                head = node.get('head_atom')
                if body.issubset(known_atoms) and head not in known_atoms:
                    applicable_mask[i] = True
        return applicable_mask, known_atoms


class AugmentedProofStepDataset(ProofStepDataset):
    """
    Augmented version with on-the-fly data augmentation.
    """
    
    def __init__(self, file_paths: List[str], spectral_dir: Optional[str] = None, 
                 augment_prob: float = 0.4, seed: int = 42, enable_instrumentation: bool = False):
        super().__init__(file_paths, spectral_dir, seed)
        self.augment_prob = augment_prob
        self.enable_instrumentation = enable_instrumentation
        
        logger.info(f"AugmentedProofStepDataset active with {augment_prob*100:.1f}% augmentation prob.")
    
    def __getitem__(self, idx: int) -> Optional[Data]:
        data = super().__getitem__(idx)
        if data is None:
            return None
        
        if random.random() < self.augment_prob:
            # Mixup augmentation (input only, preserve label)
            other_idx = random.randint(0, len(self) - 1)
            other_data = super().__getitem__(other_idx)
            if other_data is None:
                return data  # Skip if invalid
            
            alpha = random.uniform(0.2, 0.8)
            
            # Mix adjacency
            adj1 = pyg_utils.to_scipy_sparse_matrix(data.edge_index)
            adj2 = pyg_utils.to_scipy_sparse_matrix(other_data.edge_index)
            adj_mix = alpha * adj1 + (1 - alpha) * adj2
            edge_index_mix, _ = pyg_utils.from_scipy_sparse_matrix(adj_mix > 0.5)
            
            # Mix features
            data.x = alpha * data.x + (1 - alpha) * other_data.x
            
            data.edge_index = edge_index_mix
            
            # Recompute derived/step (as structure changed)
            data.derived_mask = compute_derived_mask(data.x[:, 3]) # Re-compute from mixed features
            # Note: Re-computing step_numbers is non-trivial and may not make sense for mixup
            # We'll accept the mixed step_numbers
            data.step_numbers = (alpha * data.step_numbers + (1 - alpha) * other_data.step_numbers).long()

        
        # Gumbel noise (if augment)
        if random.random() < 0.1:
            features = data.x
            noise = torch.distributions.Gumbel(0, 1).sample(features.shape).to(features.device)
            data.x += 0.01 * noise
        
        return data

# ==============================================================================
# HELPER: File Split with Path Handling
# ==============================================================================
def create_split(data_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42) -> Tuple[List[Path], List[Path], List[Path]]:
    data_dir = Path(data_dir)  # Ensure Path
    all_files = list(data_dir.rglob('*.json'))  # Paths
    random.seed(seed)
    random.shuffle(all_files)
    
    n_total = len(all_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    test_files = all_files[n_train + n_val:]
    
    # ... (logging unchanged)
    
    return train_files, val_files, test_files

# ==============================================================================
# HELPER: Dataloader Creation (Use this in train.py)
# ==============================================================================
def create_properly_split_dataloaders(
    data_dir: str, 
    spectral_dir: Optional[str] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory = False
) -> Tuple[GeoDataLoader, GeoDataLoader, GeoDataLoader]:
    """
    Creates the DataLoaders using the *correct* cleaned dataset and collate function.
    
    FIX: This function now correctly imports 'fixed_collate_fn'
         from 'dataset_utils.py' and passes it to the DataLoader.
    """
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # --- THIS IS THE CRITICAL FIX ---
    # Import the one, true collate function
    try:
        # NOTE: fixed_collate_fn is now defined *in this file*
        logger.info("Using self-contained fixed_collate_fn from dataset.py.")
    except ImportError:
        logger.error("FATAL: Could not import fixed_collate_fn from dataset_utils.py!")
        raise
    # --- END FIX ---
    
    # 1. Create file splits
    train_files, val_files, test_files = create_split(
        data_dir, train_ratio, val_ratio, seed
    )
    
    # 2. Create datasets
    train_dataset = ProofStepDataset(
        train_files, spectral_dir=spectral_dir, seed=seed
    )
    val_dataset = ProofStepDataset(
        val_files, spectral_dir=spectral_dir, seed=seed + 1
    )
    test_dataset = ProofStepDataset(
        test_files, spectral_dir=spectral_dir, seed=seed + 2
    )
    
    # 3. Create loaders with FIXED collation
    train_loader = GeoDataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=fixed_collate_fn,
        pin_memory=pin_memory  # <-- PASSING THE CORRECT FUNCTION
    )
    val_loader = GeoDataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=fixed_collate_fn,
        pin_memory=pin_memory  # <-- PASSING THE CORRECT FUNCTION
    )
    test_loader = GeoDataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=fixed_collate_fn,
        pin_memory=pin_memory  # <-- PASSING THE CORRECT FUNCTION
    )
    
    logger.info("\n✅ DataLoaders created with fixed_collate_fn (using torch.utils.data.DataLoader).")    
    logger.info(f"   Train batches: {len(train_loader)}")
    logger.info(f"   Val batches: {len(val_loader)}")
    logger.info(f"   Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader